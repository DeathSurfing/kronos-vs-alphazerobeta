"""Turn each model's output into target weights, in one common frame.

Market-neutral arms (dollar-neutral, gross<=1): Kronos, AlphaZeroBeta, Momentum, Ridge.
Net-long reference arms (sum w = 1): Index B&H, Equal-Weight, Max-Sharpe, Min-Correlation.

All signal construction uses information available up to and including date t-1;
weights formed at t-1 are applied to the return of t (see harness.simulate).
Nothing here is tuned on the test window: the rules are fixed below.
"""
import numpy as np
import pandas as pd

TOP_K = 20                # names per side (fixed a priori)
LOOKBACK_MOM = 252        # 12-month momentum
SKIP = 21                 # 1-month skip (12-1 momentum)
MAXSHARPE_WIN = 252


def _neutralize(sig: pd.Series, top_k: int = TOP_K) -> pd.Series:
    """Rank -> top/bottom k -> dollar-neutral, L1 budget 1, equal risk budget."""
    s = sig.dropna()
    if len(s) < 2 * top_k:
        return pd.Series(0.0, index=sig.index)
    order = s.sort_values(ascending=False)
    longs, shorts = order.index[:top_k], order.index[-top_k:]
    w = pd.Series(0.0, index=sig.index)
    w[longs] = 0.5 / top_k
    w[shorts] = -0.5 / top_k
    return w


# ------------------------------------------------------------------- Kronos
def kronos_weights(pred: pd.DataFrame, universe: pd.DataFrame,
                   top_k: int = TOP_K) -> pd.DataFrame:
    """pred: (date, ticker, pred_close, prev_close). Signal = expected next-day return."""
    p = pred.copy()
    p["sig"] = p["pred_close"] / p["prev_close"] - 1.0
    # shift: prediction dated t is formed from data < t, already leakage-free
    rows = {}
    for d, g in p.groupby("date"):
        rows[d] = _neutralize(g.set_index("ticker")["sig"], top_k)
    W = pd.DataFrame(rows).T.fillna(0.0)
    W.index = pd.DatetimeIndex(W.index)
    return W.sort_index()


# ------------------------------------------------------------------ Momentum
def momentum_weights(close: pd.DataFrame, universe: pd.DataFrame,
                     top_k: int = TOP_K) -> pd.DataFrame:
    """12-1 cross-sectional momentum, dollar-neutral, same L1 budget."""
    sig = close.shift(SKIP) / close.shift(LOOKBACK_MOM) - 1.0
    rows = {}
    for d in sig.index:
        elig = universe.loc[d] if d in universe.index else None
        s = sig.loc[d]
        if elig is not None:
            s = s[elig.reindex(s.index).fillna(False)]
        w = _neutralize(s, top_k).reindex(close.columns).fillna(0.0)
        if w.abs().sum() > 0:
            rows[d] = w
    W = pd.DataFrame(rows).T.fillna(0.0)
    return W.sort_index()


# ---------------------------------------------------------------------- Ridge
def ridge_weights(close: pd.DataFrame, feat_sig: pd.DataFrame, universe: pd.DataFrame,
                  top_k: int = TOP_K, alpha: float = 1.0, refit: int = 63,
                  lookback: int = 756) -> pd.DataFrame:
    """Walk-forward cross-sectional ridge of next-day return on past-only features.
    Refit every `refit` days on the trailing `lookback` window. No test peeking."""
    X = feat_sig                          # DataFrame date x ticker (a single feature)
    y = close.pct_change(fill_method=None).shift(-1)
    dates = close.index
    rows, coef = {}, None
    for i, d in enumerate(dates):
        if i < LOOKBACK_MOM + 5:
            continue
        if (i % refit) == 0:
            tr = dates[max(0, i - lookback):i]
            xv = X.loc[tr].values.ravel()
            yv = y.loc[tr].values.ravel()
            m = np.isfinite(xv) & np.isfinite(yv)
            xv, yv = xv[m], yv[m]
            if len(xv) > 100:
                G = xv @ xv + alpha
                coef = float((xv @ yv) / G) if G > 0 else 0.0
        if coef is None:
            continue
        s = X.loc[d] * coef
        elig = universe.loc[d] if d in universe.index else None
        if elig is not None:
            s = s[elig.reindex(s.index).fillna(False)]
        w = _neutralize(s, top_k).reindex(close.columns).fillna(0.0)
        if w.abs().sum() > 0:
            rows[d] = w
    return pd.DataFrame(rows).T.fillna(0.0).sort_index()


# ------------------------------------------------------- net-long references
def equal_weight(universe: pd.DataFrame) -> pd.DataFrame:
    n = universe.sum(axis=1).replace(0, np.nan)
    W = universe.div(n, axis=0).fillna(0.0)
    return W


def _opt_long(cov: np.ndarray, mu: np.ndarray, mode: str) -> np.ndarray:
    from scipy.optimize import minimize
    n = len(mu)
    x0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        v = np.sqrt(max(w @ cov @ w, 1e-12))
        return -(w @ mu) / v

    def neg_mincorr(w):
        # minimise average pairwise correlation contribution of the held book
        v = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        C = cov / np.outer(v, v)
        return float(w @ C @ w)

    f = neg_sharpe if mode == "maxsharpe" else neg_mincorr
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 1.0)] * n
    r = minimize(f, x0, method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 60, "ftol": 1e-6})
    w = r.x if r.success else x0
    return w / max(w.sum(), 1e-9)


def optimized_weights(close: pd.DataFrame, universe: pd.DataFrame, mode: str,
                      win: int = MAXSHARPE_WIN, step: int = 21) -> pd.DataFrame:
    """Rolling max-Sharpe / min-correlation long-only, refit every `step` days
    on the trailing `win` window (past-only)."""
    ret = close.pct_change(fill_method=None)
    dates = close.index
    rows, last = {}, None
    for i, d in enumerate(dates):
        if i < win + 1:
            continue
        if last is None or (i % step) == 0:
            tr = ret.iloc[i - win:i]
            elig = universe.iloc[i].reindex(tr.columns).fillna(False)
            cols = list(elig[elig].index)
            if len(cols) < 10:
                continue
            sub = tr[cols].fillna(0.0)
            cov = sub.cov().values * 252
            mu = sub.mean().values * 252
            last = _opt_long(cov, mu, mode)
            last = pd.Series(last, index=cols)
        if last is None:
            continue
        s = last.reindex(close.columns).fillna(0.0)
        if s.sum() > 0:
            rows[d] = s
    return pd.DataFrame(rows).T.sort_index()


if __name__ == "__main__":
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    cols = [f"T{i:02d}" for i in range(80)]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .01, (300, 80)), 0)), index=idx, columns=cols)
    uni = pd.DataFrame(True, index=idx, columns=cols)
    for fn in (momentum_weights(close, uni),):
        w = fn.fillna(0.0)
        assert abs(w.sum(axis=1)).max() < 1e-9, "momentum not neutral"
        assert w.abs().sum(axis=1).max() <= 1.0 + 1e-9, "momentum gross"
        assert (w.abs().sum(axis=1) > 0).any()
    ew = equal_weight(uni)
    assert abs(ew.sum(axis=1) - 1).max() < 1e-9
    pred = pd.DataFrame({"date": idx[-20:], "ticker": cols[0],
                         "pred_close": 101.0, "prev_close": 100.0})
    pred = pd.concat([pred.assign(ticker=c, pred_close=100 + i)
                      for i, c in enumerate(cols)], ignore_index=True)
    kw = kronos_weights(pred, uni)
    assert abs(kw.sum(axis=1)).max() < 1e-9
    print("signals self-check ok", kw.shape, float(kw.abs().sum(axis=1).max()))
