"""AlphaZeroBeta — CHEAP edition. Identical architecture, vectorized environment.

Why this is fast (and why the previous trainer was not):
  1. The multiscale (daily/weekly/monthly) observation is precomputed ONCE per
     fold as a single strided tensor. A window is then a view, not a numpy
     rebuild. The old trainer rebuilt (N, 3F, W) in Python at every step, which
     dominated everything else.
  2. Many parallel environments (the source paper uses 32-64; the old trainer
     used 1), so each forward pass carries 48x more data per launch.
  3. The test-window rollout is a SINGLE batched forward over all test days,
     because the observation is market-only and therefore has no sequential
     dependence. The old trainer stepped day by day.
  4. Minibatch/epoch structure now matches the source paper (minibatch 256,
     10 epochs) instead of a 16-step rollout with 3 epochs.

Architecture is unchanged from the reproduction: Conv1d 32/64/64, kernels 8/4/3,
strides 4/2/1, GRU 512, shared FC 512, per-head 512 ReLU, tanh policy, Eq. 8
reward, dollar-neutral L1-ball projection.

Numerical fix (D12): the source floors sigma_p at 1e-8, which does not bound
(r_p - r_m)/sigma_p when the book is near-flat, and made the value loss diverge
to 1e25 in the old trainer. sigma_p is floored at kappa * trailing market vol
instead, and the ratio is clipped. Without this the null portfolio is a
degenerate optimum: 0/1e-8 style ratios dominate the gradient.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# --- frozen, identical to the reproduction (AZB Table D5) ---
GAMMA, LR, CLIP, LAM_GAE = 0.99, 3e-4, 0.20, 0.95
ENT_COEF, VF_COEF, GRAD_CLIP = 0.01, 0.5, 0.5
LAMBDA1, LAMBDA2 = 0.5, 0.001
AEON_WINDOW, ROLL_WIN = 100, 60
HORIZON = 200          # paper: rollout horizon 200
MINIBATCH = 256        # paper: 256 trajectories per update
PPO_EPOCHS = 10        # paper: 10
N_ENVS = 48            # paper: 32-64 parallel environments
KAPPA = 0.25           # D12: sigma_p floor as a fraction of trailing market vol
REWARD_CLIP = 10.0
ADV_CLIP = 100.0       # D13: bounds the GAE advantage before PPO consumes it
VALUE_CLIP = 50.0      # D13: bounds the value target (regressed out of the old trainer)


# ------------------------------------------------------------------ model
def project(w):
    """Dollar-neutral projection onto the L1 ball of radius 1 (AZB Listing D.4.2)."""
    w = w - w.mean(dim=-1, keepdim=True)
    g = w.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (w * torch.where(g > 1.0, 1.0 / g, torch.ones_like(g))).clamp(-1.0, 1.0)


class ActorCritic(nn.Module):
    """Identical to the reproduction: shared per-asset CNN-GRU trunk."""

    def __init__(self, feat_dim, hidden=512):
        super().__init__()
        in_ch = 3 * feat_dim
        self.c1 = nn.Conv1d(in_ch, 32, 8, stride=4)
        self.c2 = nn.Conv1d(32, 64, 4, stride=2)
        self.c3 = nn.Conv1d(64, 64, 3, stride=1)
        self.gru = nn.GRU(64, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, hidden)
        self.pi = nn.Sequential(nn.Linear(hidden, 512), nn.ReLU(), nn.Linear(512, 1), nn.Tanh())
        self.v = nn.Sequential(nn.Linear(hidden, 512), nn.ReLU(), nn.Linear(512, 1))
        self.logstd = nn.Parameter(torch.tensor(-2.0))   # sigma in [e^-4, 1]

    def forward(self, x):                      # x: (B, N, 3F, W)
        B, N, C, W = x.shape
        h = x.reshape(B * N, C, W)
        h = F.relu(self.c1(h)); h = F.relu(self.c2(h)); h = F.relu(self.c3(h))
        o, _ = self.gru(h.transpose(1, 2))
        e = F.relu(self.fc(o[:, -1])).reshape(B, N, -1)
        return self.pi(e).squeeze(-1), self.v(e).squeeze(-1)

    def dist(self, x):
        mu, v = self.forward(x)
        return torch.distributions.Normal(mu, self.logstd.clamp(-4, 0).exp()), v


# ------------------------------------------------------------------ features
def build_features(panel, ret):
    """Past-only per-asset features -> (T, N, F) float32 tensor on device."""
    close = panel["Close"].astype("float64")
    vol = panel["Volume"].astype("float64")
    cols = list(close.columns)
    r = ret[cols].fillna(0.0).astype("float64")
    feats = [
        r,                                                   # 1-day momentum
        close.pct_change(5, fill_method=None),                # 1-week
        close.pct_change(20, fill_method=None),               # 1-month
        -close.pct_change(5, fill_method=None),               # reversal
        r.rolling(20, min_periods=10).std(),                  # realised vol
        r.rolling(60, min_periods=30).skew(),                 # skew
        (vol - vol.rolling(60, min_periods=30).mean())
        / vol.rolling(60, min_periods=30).std(),              # volume z-score
        100 - 100 / (1 + r.clip(lower=0).rolling(14, min_periods=7).mean()
                     / (-r.clip(upper=0)).rolling(14, min_periods=7).mean()
                     .replace(0, np.nan)),                    # RSI
    ]
    arr = np.nan_to_num(np.stack([f.replace([np.inf, -np.inf], np.nan).values
                                  for f in feats], axis=1),
                        nan=0.0, posinf=0.0, neginf=0.0)      # (T, F, N)
    mu = arr.mean(axis=2, keepdims=True)
    sd = arr.std(axis=2, keepdims=True) + 1e-8
    arr = (arr - mu) / sd
    return close.index, cols, torch.from_numpy(arr.transpose(0, 2, 1).astype(np.float32)).to(DEV), len(feats)


def multiscale_tensor(feat_t):
    """(T, N, F) on device -> (T, N, 3F) with daily/weekly/monthly channels.
    Weekly and monthly are forward-filled by integer division, i.e. end-of-period
    values carried forward, which is causal."""
    T = feat_t.shape[0]
    idx = torch.arange(T, device=feat_t.device)
    w = feat_t.index_select(0, (idx // 5).clamp_max(T - 1))
    m = feat_t.index_select(0, (idx // 21).clamp_max(T - 1))
    return torch.cat([feat_t, w, m], dim=2)


# ------------------------------------------------------------------ batched env
class VecEnv:
    """Vectorized env over precomputed observation views.

    obs_win[t] is the model input for a decision made at time t, built entirely
    from bars < t. Envs differ only by their current time index.
    """

    def __init__(self, feat_t, ret_t, bench_t, window=AEON_WINDOW, n_envs=N_ENVS,
                 lo=None, hi=None):
        T, N, F = feat_t.shape
        ms = multiscale_tensor(feat_t)                       # (T, N, 3F)
        # unfold(0, W, 1) on (T, N, 3F) gives (T-W+1, N, 3F, W) directly
        self.obs_win = ms.unfold(0, window, 1).contiguous()  # materialise once
        self.W, self.N, self.F = window, N, F
        self.ret = ret_t                                      # (T, N)
        self.bench = bench_t                                  # (T,)
        self.T = T
        self.n_envs = n_envs
        self.lo = (window if lo is None else lo)
        self.hi = (T - 1 if hi is None else hi)
        self.reset()

    def obs(self, t):
        """(B,) decision-time indices -> (B, N, 3F, W), built from bars < t."""
        i = (t - self.W).clamp(0, self.obs_win.shape[0] - 1)
        return self.obs_win[i]

    def reset(self):
        self.t = torch.randint(self.lo + 1, max(self.lo + 2, self.hi),
                               (self.n_envs,), device=DEV)
        self.w = torch.zeros(self.n_envs, self.N, device=DEV)
        self.hrp = torch.zeros(self.n_envs, ROLL_WIN, device=DEV)
        self.hrm = torch.zeros(self.n_envs, ROLL_WIN, device=DEV)
        self.filled = 0
        return self.obs(self.t)

    def step(self, w):
        """Advance every env one day. Returns reward, rp, rm, turnover, done."""
        turn = (w - self.w).abs().sum(-1)
        rp = (w * self.ret[self.t]).sum(-1)
        rm = self.bench[self.t]
        self.w = w
        self.t = self.t + 1
        done = self.t >= self.hi
        if bool(done.any()):
            self.t = torch.where(done, torch.full_like(self.t, self.lo),
                                 self.t)
            self.w = torch.where(done[:, None], torch.zeros_like(self.w), self.w)
        return rp, rm, turn, done

    def push(self, rp, rm):
        self.hrp = torch.cat([self.hrp[:, 1:], rp[:, None]], dim=1)
        self.hrm = torch.cat([self.hrm[:, 1:], rm[:, None]], dim=1)
        self.filled = min(self.filled + 1, ROLL_WIN)


def reward_from(rp, rm, hrp, hrm, market_vol, turn, filled):
    """AZB Eq. 8 with the D12 volatility floor and ratio clip."""
    sd_p = hrp.std(dim=1) if filled > 1 else torch.full_like(rp, 1e-3)
    mu_p = hrp.mean(dim=1)
    mu_m = hrm.mean(dim=1)
    sd_m = hrm.std(dim=1)
    cov = ((hrp - mu_p[:, None]) * (hrm - mu_m[:, None])).mean(dim=1)
    corr = torch.where((sd_p > 0) & (sd_m > 0), cov / (sd_p * sd_m + 1e-8),
                       torch.zeros_like(cov))
    sigma = torch.clamp(sd_p, min=(KAPPA * market_vol))
    ra = torch.clamp((rp - rm) / sigma, -REWARD_CLIP, REWARD_CLIP)
    return ra - LAMBDA1 * corr - LAMBDA2 * turn


def market_vol_series(bench_t, win=60):
    """Trailing market volatility, used as the D12 floor base. Fully vectorised:
    cumulative sums, so no Python loop over the timeline."""
    b = bench_t
    T = len(b)
    c = torch.cat([torch.zeros(1, device=b.device, dtype=b.dtype), torch.cumsum(b, 0)])
    c2 = torch.cat([torch.zeros(1, device=b.device, dtype=b.dtype), torch.cumsum(b * b, 0)])
    i = torch.arange(T, device=b.device)
    w = (i + 1).clamp_max(win)
    j = (i + 1 - w).clamp_min(0)
    cnt = w.to(b.dtype)
    s = c[i + 1] - c[j]
    s2 = c2[i + 1] - c2[j]
    var = (s2 / cnt) - (s / cnt) ** 2
    sd = torch.sqrt(var.clamp_min(1e-12))
    sd = torch.where(w < 5, torch.full_like(sd, 1e-3), sd)
    return sd.clamp_min(1e-4)


# ------------------------------------------------------------------ train
def train_fold(feat_t, ret_t, bench_t, seed, updates, ppo_epochs, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    T, N, nfeat = feat_t.shape
    model = ActorCritic(nfeat).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    env = VecEnv(feat_t, ret_t, bench_t)
    mv = market_vol_series(bench_t)
    losses = []
    n_nan = 0
    t0 = time.time()

    for it in range(updates):
        O, A, LP, R, V, D = [], [], [], [], [], []
        for _ in range(HORIZON):
            obs_t = env.obs(env.t)                 # decision-time observation
            with torch.no_grad():
                d, v = model.dist(obs_t)
                raw = d.sample()                   # density is defined here
                lp = d.log_prob(raw).sum(-1)       # ... so evaluate it here
                a = project(raw)                   # deterministic action map
            rp, rm, turn, done = env.step(a)
            env.push(rp, rm)
            rew = reward_from(rp, rm, env.hrp, env.hrm, mv[env.t.clamp_max(T - 1)],
                              turn, env.filled)
            O.append(obs_t)
            A.append(a); LP.append(lp); R.append(rew)
            V.append(v.mean(-1)); D.append(done.float())
        with torch.no_grad():
            _, last_v = model.dist(env.obs(env.t))
            last_v = last_v.mean(-1)
        R = torch.stack(R); V = torch.stack(V); D = torch.stack(D)   # (H, E)
        adv = torch.zeros_like(R)
        last = torch.zeros(env.n_envs, device=DEV)
        for t in reversed(range(HORIZON)):
            nv = last_v if t == HORIZON - 1 else V[t + 1]
            delta = R[t] + GAMMA * nv * (1 - D[t]) - V[t]
            last = delta + GAMMA * LAM_GAE * (1 - D[t]) * last
            adv[t] = last
        adv = adv.clamp(-ADV_CLIP, ADV_CLIP)
        ret_ = (adv + V).clamp(-VALUE_CLIP, VALUE_CLIP)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        Ob = torch.stack(O).reshape(-1, N, 3 * nfeat, env.W)
        Ab = torch.stack(A).reshape(-1, N)
        LPb = torch.stack(LP).reshape(-1)
        advb = adv.reshape(-1); retb = ret_.reshape(-1)
        n = Ob.shape[0]
        idx = torch.randperm(n, device=DEV)
        for _ in range(ppo_epochs):
            for s in range(0, n, MINIBATCH):
                b = idx[s:s + MINIBATCH]
                d, v = model.dist(Ob[b])
                lp = d.log_prob(Ab[b]).sum(-1)     # Ab holds RAW samples
                ratio = (lp - LPb[b]).exp()
                pi = -torch.min(ratio * advb[b],
                                ratio.clamp(1 - CLIP, 1 + CLIP) * advb[b]).mean()
                # value head is per asset, the return target is per (step, env)
                vtgt = retb[b].unsqueeze(-1).expand_as(v)
                loss = (pi + VF_COEF * F.mse_loss(v, vtgt)
                        - ENT_COEF * d.entropy().sum(-1).mean())
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                if torch.isfinite(loss) and all(torch.isfinite(p.grad).all()
                                                for p in model.parameters() if p.grad is not None):
                    opt.step()
                if not torch.isfinite(loss):
                    n_nan += 1
                    losses.append(float("nan"))
                else:
                    losses.append(float(loss))
        if verbose:
            print(f"  update {it+1}/{updates} loss {np.mean(losses[-200:]):.4f} "
                  f"gross {env.w.abs().sum(-1).mean().item():.3f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
    finite = [x for x in losses if np.isfinite(x)]
    return (model, sum(p.numel() for p in model.parameters()),
            float(np.median(finite)) if finite else float("nan"),
            float(max(finite)) if finite else float("nan"),
            n_nan, time.time() - t0, env)


def test_weights(model, feat_t, ret_t, dates_te):
    """Single batched forward over the whole test window: obs is market-only,
    so there is no sequential dependence to step through."""
    T, N, F = feat_t.shape
    env = VecEnv(feat_t, ret_t, torch.zeros(T, device=DEV))
    ts = torch.arange(env.W, T, device=DEV)
    with torch.no_grad():
        mu, _ = model(env.obs(ts))
        W = project(mu)
    return pd.DataFrame(W.cpu().numpy(), index=pd.DatetimeIndex(dates_te[env.W:T], name="date"),
                       columns=[f"a{i}" for i in range(N)])


def folds_from_index(dates, train=756, val=126, test=126, start="2014-01-01"):
    first = dates.searchsorted(pd.Timestamp(start))
    out, k, t0 = [], 0, first
    while t0 + test <= len(dates):
        out.append({"fold": k, "tr": (t0 - train - val, t0 - val), "te": (t0, t0 + test)})
        k += 1; t0 += test
    return out


def load_all(root):
    panel = pd.read_parquet(os.path.join(root, "panel.parquet"))
    if set(map(str, panel.columns.get_level_values(0))) != {"Close", "Open", "High", "Low", "Volume"}:
        panel = panel.swaplevel(0, 1, axis=1).sort_index(axis=1)
    ret = pd.read_parquet(os.path.join(root, "ret.parquet"))
    bench = pd.read_parquet(os.path.join(root, "bench.parquet"))["ret"]
    liq = pd.read_parquet(os.path.join(root, "liq_rank.parquet"))
    return panel, ret, bench, liq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("KVAB_ROOT", "/opt/data/kvab"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--updates", type=int, default=8)     # 8 x 200 = 1600 steps
    ap.add_argument("--epochs", type=int, default=PPO_EPOCHS)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--topn", type=int, default=50)
    ap.add_argument("--tag", default="azbfast")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "windows_" + args.tag)
    os.makedirs(out, exist_ok=True)

    panel, ret, bench, liq = load_all(args.root)
    elig = liq.le(args.topn).mean(axis=0)
    cols = [c for c in panel["Close"].columns if c in ret.columns and elig.get(c, 0) >= 0.5]
    print("universe", len(cols), flush=True)
    panel = {k: panel[k][cols] for k in ["Open", "High", "Low", "Close", "Volume"]}
    ret = ret[cols]
    dates, tickers, feat, F_ = build_features(panel, ret)
    rmat = torch.from_numpy(ret.reindex(dates)[tickers].fillna(0.0).values.astype(np.float32)).to(DEV)
    bvec = torch.from_numpy(bench.reindex(dates).fillna(0.0).values.astype(np.float32)).to(DEV)
    dates = pd.DatetimeIndex(dates)

    fs = folds_from_index(dates)
    if args.folds != "all":
        keep = {int(x) for x in args.folds.split(",")}
        fs = [f for f in fs if f["fold"] in keep]
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"folds {len(fs)} seeds {seeds} updates {args.updates} "
          f"(= {args.updates*HORIZON} steps) envs {N_ENVS}", flush=True)

    for f in fs:
        a, b = f["te"]
        ftr, rtr, btr = feat[:a], rmat[:a], bvec[:a]
        fte = feat[a - AEON_WINDOW - 1:b]
        dte = dates[a - AEON_WINDOW - 1:b]
        for sd in seeds:
            model, npar, loss, loss_max, n_nan, rt, env = train_fold(
                ftr, rtr, btr, sd, args.updates, args.epochs, args.verbose)
            wdf = test_weights(model, fte, rmat[a - AEON_WINDOW - 1:b], dte)
            wdf.columns = tickers
            wdf.to_parquet(os.path.join(out, f"w_fold{f['fold']:02d}_seed{sd}.parquet"))
            rec = {"exp": f"{args.tag.upper()}-SPX-{f['fold']:03d}-S{sd}", "model": args.tag,
                   "fold": f["fold"], "seed": sd, "params": int(npar),
                   "loss_median": None if not np.isfinite(loss) else round(loss, 5),
                   "loss_max": None if not np.isfinite(loss_max) else float(f"{loss_max:.4g}"),
                   "n_nonfinite_updates": int(n_nan), "updates": args.updates,
                   "steps": args.updates * HORIZON, "test_start": str(dates[a].date()),
                   "test_end": str(dates[b - 1].date()),
                   "runtime_s": round(rt, 1), "n_assets": len(tickers),
                   "gross_exposure_mean_test": round(float(wdf.abs().sum(axis=1).mean()), 4),
                   "device": DEV}
            with open(os.path.join(args.root, "experiment_log.jsonl"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
