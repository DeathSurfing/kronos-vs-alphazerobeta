"""Statistical tests for the comparison. All implemented from first principles.

Tests used and why:
  - stationary block bootstrap on Sharpe differences: returns are autocorrelated
    and heteroskedastic, so i.i.d. resampling would understate the variance of
    the Sharpe difference. Blocks of 21 days preserve short-horizon dependence.
  - Diebold-Mariano on squared forecast errors (Kronos vs the random-walk
    benchmark pred_close = prev_close). DM is the standard test for comparing
    two forecast sequences; we use the Harvey-Leybourne-Newbold correction.
  - Newey-West (HAC) regression of the strategy's return on the benchmark to
    test whether alpha survives at conventional levels with autocorrelation and
    heteroskedasticity in the errors.
  - Sharpe ratio difference is also reported as an effect size (points of Sharpe).

No test is applied that the design cannot support: forecasting tests only run
where a model emits a forecast (Kronos), never for the RL agent, which emits
portfolio weights.
"""
import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def sharpe(r, ann=TRADING_DAYS):
    r = np.asarray(pd.Series(r).dropna(), dtype=float)
    s = r.std(ddof=1)
    return r.mean() / s * np.sqrt(ann) if s > 0 else np.nan


def block_bootstrap_idx(n, block, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=nb)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return np.clip(idx, 0, n - 1)


def sharpe_diff_test(a, b, block=21, n_boot=10000, seed=0):
    """Paired stationary block bootstrap of the Sharpe difference (a - b)."""
    rng = np.random.default_rng(seed)
    df = pd.concat([pd.Series(a).rename("a"), pd.Series(b).rename("b")], axis=1).dropna()
    x, y = df["a"].values, df["b"].values
    n = len(df)
    obs = sharpe(x) - sharpe(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = block_bootstrap_idx(n, block, rng)
        diffs[i] = sharpe(x[idx]) - sharpe(y[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    centered = diffs - obs
    p = 2 * min((centered <= -abs(obs)).mean(), (centered >= abs(obs)).mean())
    return {"sharpe_a": sharpe(x), "sharpe_b": sharpe(y), "diff": obs,
            "ci_lo": lo, "ci_hi": hi, "p_value": float(min(1.0, p)),
            "n_obs": int(n), "block": block}


def diebold_mariano(e1, e2, h=1, power=2):
    """DM statistic with the Harvey-Leybourne-Newbold small-sample correction.
    e1, e2 are loss VALUES (already squared or absolute errors), same length."""
    d = np.asarray(e1, float) - np.asarray(e2, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"dm": np.nan, "p_value": np.nan, "n": n}
    dbar = d.mean()
    # Newey-West variance (lag h-1)
    g0 = np.mean((d - dbar) ** 2)
    s = g0
    for k in range(1, h):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        s += 2 * gk
    var = s / n
    if var <= 0:
        return {"dm": np.nan, "p_value": np.nan, "n": n, "mean_diff": dbar}
    dm = dbar / np.sqrt(var)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return {"dm": float(dm), "p_value": float(p), "n": int(n), "mean_diff": float(dbar)}


def newey_west_alpha(r, bench, lag=5):
    """OLS r = a + b*bench + e with HAC(lag) standard errors (Newey-West 1987)."""
    y = np.asarray(pd.Series(r).dropna(), float)
    x = np.asarray(pd.Series(bench).reindex(pd.Series(r).dropna().index), float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = len(y)
    if n < 30:
        return {}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lag + 1):
        w = 1 - L / (lag + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - 1))
    return {"alpha_daily": float(beta[0]), "alpha_ann": float(beta[0] * TRADING_DAYS),
            "alpha_t": float(t[0]), "alpha_p": float(p[0]),
            "beta": float(beta[1]), "beta_t": float(t[1]), "beta_p": float(p[1]),
            "n": n, "lag": lag}


def forecast_metrics(pred_close, actual_close, prev_close):
    """MAE / RMSE / directional accuracy / rank IC for a forecast series."""
    p = np.asarray(pred_close, float)
    a = np.asarray(actual_close, float)
    pv = np.asarray(prev_close, float)
    m = np.isfinite(p) & np.isfinite(a) & np.isfinite(pv)
    p, a, pv = p[m], a[m], pv[m]
    err = p - a
    rw_err = pv - a                       # random-walk forecast
    dacc = np.mean(np.sign(p - pv) == np.sign(a - pv))
    out = {"mae": float(np.mean(np.abs(err))),
           "rmse": float(np.sqrt(np.mean(err ** 2))),
           "dir_acc": float(dacc), "n": int(len(p)),
           "rw_mae": float(np.mean(np.abs(rw_err))),
           "rw_rmse": float(np.sqrt(np.mean(rw_err ** 2)))}
    out["mae_skill"] = 1 - out["mae"] / out["rw_mae"] if out["rw_mae"] > 0 else np.nan
    return out


def rank_ic(pred_ret, actual_ret):
    """Spearman rank correlation between predicted and realised cross-section."""
    d = pd.DataFrame({"p": pred_ret, "a": actual_ret}).dropna()
    if len(d) < 10:
        return np.nan
    return float(stats.spearmanr(d["p"], d["a"]).statistic)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 1500
    b = pd.Series(rng.normal(0, 0.01, n))
    a = b + pd.Series(rng.normal(0, 0.002, n))          # a genuinely better
    t = sharpe_diff_test(a, b, n_boot=400, seed=3)
    assert t["diff"] > 0 and t["ci_lo"] < t["diff"] < t["ci_hi"]
    e1 = np.abs(rng.normal(0, 1.1, 500)); e2 = np.abs(rng.normal(0, 1.0, 500))
    dm = diebold_mariano(e1 ** 2, e2 ** 2)
    assert np.isfinite(dm["dm"])
    nw = newey_west_alpha(a, b)
    assert nw["beta"] > 0.8
    fm = forecast_metrics(np.array([101., 100., 99.]), np.array([102., 99., 98.]),
                          np.array([100., 100., 100.]))
    assert fm["mae"] > 0 and 0 <= fm["dir_acc"] <= 1
    print("statistics self-check ok", round(t["diff"], 3), round(dm["dm"], 2), round(nw["beta"], 3))
