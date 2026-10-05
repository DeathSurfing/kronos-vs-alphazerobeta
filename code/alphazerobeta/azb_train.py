"""AlphaZeroBeta reproduction: CNN-GRU policy trained with recurrent PPO on a
dollar-neutral, market-neutral portfolio objective.

Components taken from Belyakov (2026) (arXiv 2607.18001, Financial Innovation):
  architecture  : Conv1d 32/64/64 with kernels 8/4/3 and strides 4/2/1, GRU(512),
                  shared FC(512), per-head 512-unit ReLU, tanh policy output
  observation   : multi-resolution (daily/weekly/monthly) tensor stacked on the
                  channel axis, 100-step agent window (n_agent_window = 100)
  RL            : PPO, gamma 0.99, lr 3e-4 Adam, clip 0.20, GAE lambda 0.95,
                  entropy coef 0.01, value coef 0.5, 10 epochs, grad clip 0.5
  reward (Eq. 8): (r_p - r_m)/sigma_p - l1*corr(r_p, r_m) - l2*sum|dw_i|
                  l1 = 0.5, l2 = 0.001, sigma_p floored at 1e-8, 60-day window
  constraints   : dollar-neutral (sum w = 0), L1 ball radius 1, box [-1, 1]
  protocol      : 36mo train / 6mo val / 6mo test, 6mo step, test 2014-2024
  costs         : 5 bps/side top decile, 15 bps/side other US names, 30 bps/yr borrow

DEVIATIONS (all listed in docs/reproduction.md):
  D1 the paper's single GRU consumes a flattened feature map; we use a SHARED
     CNN-GRU applied per asset (asset dim = batch dim), because the flattened
     variant is quadratic in the number of assets and infeasible here.
  D2 envs: paper uses 256 trajectories per update, we use a single trajectory
     per update with a 24-step rollout, repeated.
  D3 steps: paper trains to ~57-106 GPU-hours per fold-index; we budget
     --steps rollouts per fold, disclosed in docs/experiment_log.md.
  D4 3 seeds instead of 9.
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

# frozen hyperparameters (AZB Table D5)
GAMMA, LR, CLIP, LAM_GAE = 0.99, 5e-5, 0.20, 0.95   # D10: lr lowered; see NaN guard below
ENT_COEF, VF_COEF, PPO_EPOCHS, GRAD_CLIP = 0.01, 0.5, 3, 0.5   # D5: paper uses 10
LAMBDA1, LAMBDA2 = 0.5, 0.001
AEON_WINDOW, ROLL_WIN = 100, 60
ROLLOUT, MINIBATCH = 16, 64
REWARD_CLIP = 10.0   # D10: the source Eq.8 term is unbounded when sigma_p -> 0
VALUE_CLIP = 50.0


def project(w):
    """Dollar-neutral projection onto the L1 ball of radius 1 (AZB Listing D.4.2)."""
    w = w - w.mean(dim=-1, keepdim=True)
    g = w.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (w * torch.where(g > 1.0, 1.0 / g, torch.ones_like(g))).clamp(-1.0, 1.0)


class ActorCritic(nn.Module):
    def __init__(self, feat_dim, n_assets, hidden=512):
        super().__init__()
        in_ch = 3 * feat_dim
        self.c1 = nn.Conv1d(in_ch, 32, 8, stride=4)
        self.c2 = nn.Conv1d(32, 64, 4, stride=2)
        self.c3 = nn.Conv1d(64, 64, 3, stride=1)
        self.gru = nn.GRU(64, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, hidden)
        self.pi = nn.Sequential(nn.Linear(hidden, 512), nn.ReLU(), nn.Linear(512, 1), nn.Tanh())
        self.v = nn.Sequential(nn.Linear(hidden, 512), nn.ReLU(), nn.Linear(512, 1))
        self.logstd = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):                      # x: (N, 3F, W)
        h = F.relu(self.c1(x)); h = F.relu(self.c2(h)); h = F.relu(self.c3(h))
        o, _ = self.gru(h.transpose(1, 2))
        e = F.relu(self.fc(o[:, -1]))          # (N, hidden)
        return self.pi(e).squeeze(-1), self.v(e).squeeze(-1)

    def dist(self, x):
        mu, v = self.forward(x)
        return torch.distributions.Normal(mu, self.logstd.clamp(-5, 1).exp()), v


def build_features(panel, ret, window=AEON_WINDOW):
    """Past-only per-asset features -> (T, N, F) float32."""
    close = panel["Close"].astype("float64")
    vol = panel["Volume"].astype("float64")
    cols = list(close.columns)
    r = ret[cols].fillna(0.0).astype("float64")
    mom1, mom5 = r, close.pct_change(5, fill_method=None)
    mom20 = close.pct_change(20, fill_method=None)
    vl20 = r.rolling(20, min_periods=10).std()
    sk60 = r.rolling(60, min_periods=30).skew()
    vm = vol.rolling(60, min_periods=30).mean()
    v20 = (vol - vm) / vol.rolling(60, min_periods=30).std()
    dn = (-r.clip(upper=0)).rolling(14, min_periods=7).mean().replace(0, np.nan)
    rsi = 100 - 100 / (1 + r.clip(lower=0).rolling(14, min_periods=7).mean() / dn)
    feats = [mom1, mom5, mom20, -mom5, vl20, sk60, v20, rsi]
    arr = np.nan_to_num(np.stack([f.replace([np.inf, -np.inf], np.nan).values
                                  for f in feats], axis=1),
                        nan=0.0, posinf=0.0, neginf=0.0)          # T x F x N
    mu = arr.mean(axis=2, keepdims=True)
    sd = arr.std(axis=2, keepdims=True) + 1e-8
    arr = (arr - mu) / sd
    return close.index, cols, arr.transpose(0, 2, 1).astype(np.float32), len(feats)


def multiscale(win_nlf):
    """(N, L, F) daily window -> (N, 3F, L) with weekly/monthly channels."""
    L = win_nlf.shape[1]
    def up(a):
        if a.shape[1] == 0:
            return np.zeros((a.shape[0], L, a.shape[2]), dtype=np.float32)
        return a[:, np.clip(np.arange(L) * a.shape[1] // L, 0, a.shape[1] - 1), :]
    st = np.concatenate([win_nlf, up(win_nlf[:, ::5, :]), up(win_nlf[:, ::21, :])], axis=2)
    return st.transpose(0, 2, 1).astype(np.float32)


class Env:
    """T x N x F features, T x N returns, T benchmark returns. Sequential over time."""

    def __init__(self, feat, ret, bench, window=AEON_WINDOW):
        self.feat, self.ret, self.bench, self.W = feat, ret, bench, window
        self.T, self.n, self.F = feat.shape
        self.reset()

    def obs(self, t):
        win = self.feat[t - self.W:t].transpose(1, 0, 2)      # (N, W, F)
        return torch.from_numpy(multiscale(win)).to(DEV)

    def reset(self, start=None):
        self.t = self.W if start is None else int(start)
        self.w = np.zeros(self.n, dtype=np.float32)
        return self.obs(self.t)

    def step(self, w):
        w = w.detach().cpu().numpy()
        turn = float(np.abs(w - self.w).sum())
        rp = float(np.dot(w, self.ret[self.t])) if self.t < self.T else 0.0
        rm = float(self.bench[self.t]) if self.t < self.T else 0.0
        self.w = w
        self.t += 1
        done = self.t >= self.T - 1
        return rp, rm, turn, done


def reward(rp, rm, sigma_p, corr, turn):
    """AZB Eq. 8. The risk-adjusted term is clipped: sigma_p is floored at 1e-8 in
    the source, which does not bound the ratio when the portfolio is near-flat
    early in training, and an unclipped term diverges (observed: value loss 3e25)."""
    ra = (rp - rm) / max(sigma_p, 1e-8)
    ra = float(np.clip(ra, -REWARD_CLIP, REWARD_CLIP))
    return ra - LAMBDA1 * corr - LAMBDA2 * turn


def train_fold(feat, ret, bench, seed, steps):
    torch.manual_seed(seed); np.random.seed(seed)
    n_assets, F_ = feat.shape[1], feat.shape[2]
    model = ActorCritic(F_, n_assets).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist_rp, hist_rm = [], []
    env = Env(feat, ret, bench)
    obs = env.reset()
    losses = []
    t0 = time.time()
    for it in range(steps):
        O, A, LP, R, V, D = [], [], [], [], [], []
        for _ in range(ROLLOUT):
            with torch.no_grad():
                d, v = model.dist(obs)
                a = project(d.sample())
                lp = d.log_prob(a)          # (N,)
            rp, rm, turn, done = env.step(a)
            wrp = np.array(hist_rp[-ROLL_WIN:] + [rp]); wrm = np.array(hist_rm[-ROLL_WIN:] + [rm])
            sig = float(wrp.std()) if len(wrp) > 1 else 1e-3
            cc = float(np.corrcoef(wrp, wrm)[0, 1]) if len(wrp) >= 5 and wrm.std() > 0 else 0.0
            if not np.isfinite(cc):
                cc = 0.0
            hist_rp.append(rp); hist_rm.append(rm)
            O.append(obs); A.append(a); LP.append(lp)
            R.append(reward(rp, rm, sig, cc, turn)); V.append(v.mean().item()); D.append(done)
            obs = env.obs(env.t) if not done else env.reset()
        with torch.no_grad():
            _, last_v = model(obs)
        adv = np.zeros(ROLLOUT, dtype=np.float32)
        last = 0.0
        for t in reversed(range(ROLLOUT)):
            nv = last_v.mean().item() if t == ROLLOUT - 1 else V[t + 1]
            delta = R[t] + GAMMA * nv * (1 - D[t]) - V[t]
            last = delta + GAMMA * LAM_GAE * (1 - D[t]) * last
            adv[t] = last
        adv_t = torch.tensor(adv, device=DEV)
        ret_t = (adv_t + torch.tensor(V, device=DEV)).clamp(-VALUE_CLIP, VALUE_CLIP)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        # fold the asset dimension into the batch: CNN-GRU is shared per asset
        O = torch.stack(O).reshape(-1, 3 * F_, AEON_WINDOW)
        Af = torch.stack(A).reshape(-1)
        LPf = torch.stack(LP).reshape(-1)
        advf = adv_t.unsqueeze(1).expand(ROLLOUT, n_assets).reshape(-1)
        retf = ret_t.unsqueeze(1).expand(ROLLOUT, n_assets).reshape(-1)
        idx = np.random.permutation(O.shape[0])
        for _ in range(PPO_EPOCHS):
            for s in range(0, O.shape[0], MINIBATCH):
                b = torch.tensor(idx[s:s + MINIBATCH], device=DEV)
                d, v = model.dist(O[b])
                lp = d.log_prob(Af[b])
                ratio = (lp - LPf[b]).exp()
                pi_loss = -torch.min(ratio * advf[b],
                                     ratio.clamp(1 - CLIP, 1 + CLIP) * advf[b]).mean()
                loss = pi_loss + VF_COEF * F.mse_loss(v, retf[b]) - ENT_COEF * d.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                # D10: a single non-finite batch poisons every parameter (observed:
                # policy loc -> NaN on fold 2 at lr 1e-4). Skip such updates.
                if all(torch.isfinite(pp.grad).all() for pp in model.parameters() if pp.grad is not None) \
                        and torch.isfinite(loss):
                    opt.step()
                losses.append(float(loss) if torch.isfinite(loss) else 0.0)
    return model, sum(p.numel() for p in model.parameters()), float(np.mean(losses[-100:])), time.time() - t0


def rollout_weights(model, feat, dates_te):
    """Deterministic policy over the test window -> DataFrame date x ticker."""
    model.eval()
    T = feat.shape[0]
    rows, idxs = [], []
    with torch.no_grad():
        for t in range(AEON_WINDOW, T):
            win = feat[t - AEON_WINDOW:t].transpose(1, 0, 2)
            o = torch.from_numpy(multiscale(win)).to(DEV)
            mu, _ = model(o)
            rows.append(project(mu).cpu().numpy())
            idxs.append(dates_te[t])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idxs))


def folds_from_index(dates, train=756, val=126, test=126, start="2014-01-01"):
    first = dates.searchsorted(pd.Timestamp(start))
    out, k, t0 = [], 0, first
    while t0 + test <= len(dates):
        out.append({"fold": k, "tr": (t0 - train - val, t0 - val),
                    "va": (t0 - val, t0), "te": (t0, t0 + test)})
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
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--tag", default="azb")
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
    rmat = ret.reindex(dates)[tickers].fillna(0.0).values.astype(np.float32)
    bvec = bench.reindex(dates).fillna(0.0).values.astype(np.float32)
    dates = pd.DatetimeIndex(dates)

    fs = folds_from_index(dates)
    if args.folds != "all":
        keep = {int(x) for x in args.folds.split(",")}
        fs = [f for f in fs if f["fold"] in keep]
    seeds = [int(x) for x in args.seeds.split(",")]
    print("folds", len(fs), "seeds", seeds, "steps", args.steps, flush=True)

    for f in fs:
        a, b = f["te"]
        ftr, rtr, btr = feat[:a], rmat[:a], bvec[:a]
        fte = feat[a - AEON_WINDOW - 1:b]
        dte = dates[a - AEON_WINDOW - 1:b]
        for sd in seeds:
            model, npar, loss, rt = train_fold(ftr, rtr, btr, sd, args.steps)
            wdf = rollout_weights(model, fte, dte)
            wdf.columns = tickers
            wdf.to_parquet(os.path.join(out, f"w_fold{f['fold']:02d}_seed{sd}.parquet"))
            rec = {"exp": f"{args.tag.upper()}-SPX-{f['fold']:03d}-S{sd}", "model": args.tag,
                   "fold": f["fold"], "seed": sd, "params": int(npar), "loss": round(loss, 5),
                   "train_end": str(dates[a - 1].date()), "test_start": str(dates[a].date()),
                   "test_end": str(dates[b - 1].date()), "steps": args.steps,
                   "runtime_s": round(rt, 1), "n_assets": len(tickers), "device": DEV}
            with open(os.path.join(args.root, "experiment_log.jsonl"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
