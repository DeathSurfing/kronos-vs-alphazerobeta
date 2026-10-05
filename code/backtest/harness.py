"""Common portfolio harness: costs, constraints, metrics, fold generation.

Every model is converted to a target-weight matrix (date x ticker) and pushed
through `simulate`. Nothing model-specific lives here, so the comparison is
mechanically identical across models.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------- constraints
def dollar_neutral(w):
    """Centre then project onto the L1 ball of radius gross_cap (AZB eq. D.4.2)."""
    w = np.nan_to_num(w, nan=0.0)
    w = w - w.mean()
    g = np.abs(w).sum()
    if g > 1.0:
        w = w / g
    return np.clip(w, -1.0, 1.0)


# ---------------------------------------------------------------- simulation
def simulate(weights: pd.DataFrame, ret: pd.DataFrame, cost_bps: pd.Series,
             borrow_bps_annual: pd.Series, gross_cap: float = 1.0):
    """weights[t] = target exposure formed from information <= t, held over t+1.

    cost_bps[t] / borrow_bps_annual[t] are per-name cost rates for the name's
    liquidity bucket on date t (bps per side / bps per year).
    """
    idx = weights.index.intersection(ret.index)
    w = weights.loc[idx].copy()
    r = ret.loc[idx].reindex(columns=w.columns).fillna(0.0)
    cb = cost_bps.reindex(idx).fillna(cost_bps.median())
    bb = borrow_bps_annual.reindex(idx).fillna(borrow_bps_annual.median())

    # gross book: 0.85 long + -0.85 short by construction of the L1 ball
    gross_w = w.abs().sum(axis=1).replace(0, np.nan)

    dw = w.diff()
    dw.iloc[0] = w.iloc[0]
    turnover = dw.abs().sum(axis=1)                       # sum |dw_i|

    trade_cost = (dw.abs().mul(cb, axis=0)).sum(axis=1) / 1e4
    borrow_cost = (w.abs().clip(lower=0).mul(bb, axis=0)).sum(axis=1) / 1e4 / TRADING_DAYS

    gross_ret = (w.shift(0) * r.shift(-1)).sum(axis=1)     # w_t applied to r_{t+1}
    net_ret = gross_ret - trade_cost - borrow_cost
    net_ret.iloc[-1] = 0.0

    lw = w.clip(lower=0).sum(axis=1)
    sw = (-w.clip(upper=0)).sum(axis=1)
    return pd.DataFrame({
        "gross": gross_ret, "net": net_ret, "turnover": turnover,
        "trade_cost": trade_cost, "borrow_cost": borrow_cost,
        "long": lw, "short": sw, "gross_exposure": gross_w.fillna(0.0),
        "net_exposure": w.sum(axis=1),
    })


# ---------------------------------------------------------------- metrics
def _ann(x, periods=TRADING_DAYS):
    return x.mean() * periods


def drawdown(r):
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1


def metrics(net: pd.Series, bench: pd.Series | None = None) -> dict:
    net = net.dropna()
    if len(net) < 30:
        return {}
    eq = (1 + net).cumprod()
    yrs = len(net) / TRADING_DAYS
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = net.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = _ann(net) / vol if vol > 0 else np.nan
    dn = net[net < 0]
    dstd = dn.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(dn) > 1 else np.nan
    sortino = _ann(net) / dstd if dstd and dstd > 0 else np.nan
    dd = drawdown(net)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    var95 = np.percentile(net, 5)
    es95 = net[net <= var95].mean() if (net <= var95).any() else np.nan
    out = {
        "n_days": len(net), "total_return": eq.iloc[-1] - 1, "cagr": cagr,
        "ann_return": _ann(net), "ann_vol": vol, "sharpe": sharpe,
        "sortino": sortino, "max_dd": mdd, "calmar": calmar,
        "downside_dev": dstd, "var95": var95, "es95": es95,
        "skew": net.skew(), "kurtosis": net.kurtosis(),
    }
    if bench is not None:
        b = bench.reindex(net.index).fillna(0.0)
        v = b.var(ddof=1)
        out["beta"] = net.cov(b) / v if v > 0 else np.nan
        out["corr"] = net.corr(b)
        out["ann_alpha"] = _ann(net) - out["beta"] * _ann(b)
    return out


def trade_stats(sim: pd.DataFrame) -> dict:
    d = sim["turnover"]
    active = d > 1e-8
    return {
        "turnover_ann": d.mean() * TRADING_DAYS,
        "n_rebalances": int(active.sum()),
        "avg_gross_exposure": sim["gross_exposure"].mean(),
        "avg_net_exposure": sim["net_exposure"].abs().mean(),
        "avg_long": sim["long"].mean(), "avg_short": sim["short"].mean(),
        "cost_drag_ann": (sim["trade_cost"] + sim["borrow_cost"]).mean() * TRADING_DAYS,
    }


# ---------------------------------------------------------------- folds
def folds(index: pd.DatetimeIndex, train=756, val=126, test=126, start="2014-01-01"):
    """AZB §3: rolling 3y train / 6mo val / 6mo test, 6mo step."""
    first = index.searchsorted(pd.Timestamp(start))
    out, k = [], 0
    t0 = first
    while t0 + test <= len(index):
        tr = index[t0 - train - val: t0 - val]
        va = index[t0 - val: t0]
        te = index[t0: t0 + test]
        if len(tr) < train or len(va) < val:
            break
        out.append({"fold": k, "train": tr, "val": va, "test": te})
        k += 1
        t0 += test
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = pd.date_range("2020-01-01", periods=400, freq="B")
    cols = [f"T{i}" for i in range(20)]
    ret = pd.DataFrame(rng.normal(0, 0.01, (400, 20)), index=d, columns=cols)
    raw = rng.normal(0, 1, (400, 20))
    wdf = pd.DataFrame([dollar_neutral(r) for r in raw], index=d, columns=cols)
    cb = pd.Series(5.0, index=d)
    bb = pd.Series(30.0, index=d)
    sim = simulate(wdf, ret, cb, bb)
    m = metrics(sim["net"], ret.mean(axis=1))
    ts = trade_stats(sim)
    assert abs(wdf.sum(axis=1)).max() < 1e-9, "not dollar neutral"
    assert wdf.abs().sum(axis=1).max() <= 1.0 + 1e-9, "gross cap violated"
    assert np.isfinite(list(m.values())).all(), "non-finite metric"
    assert sim["net"].std() > 0
    f = folds(d)
    assert all(len(x["train"]) == 756 for x in f), "bad train window"
    assert all(len(x["test"]) == 126 for x in f), "bad test window"
    print("harness self-check ok", {k: round(v, 4) for k, v in list(m.items())[:6]})
