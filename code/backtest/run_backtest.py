"""Run every arm through the common harness on the frozen protocol.

Arms:
  market-neutral (dollar-neutral, L1 budget 1)
    kronos      zero-shot Kronos forecasts -> cross-sectional long/short
    azb         AlphaZeroBeta CNN-GRU Recurrent PPO policy
    momentum    12-1 cross-sectional momentum
    ridge       walk-forward cross-sectional ridge on past-only features
  net-long reference (sum w = 1)
    index       ^GSPC buy-and-hold
    equal       equal-weight S&P 500 members
    maxsharpe   rolling max-Sharpe long-only
    mincorr     rolling minimum-correlation long-only
    rw          random walk naive (per-asset drift = 0) -> equal weight, kept
                as the forecast baseline, not a portfolio arm

Writes:
  results/processed/daily_net.parquet   (date x arm, net returns)
  results/processed/daily_gross.parquet (date x arm, gross returns)
  results/processed/metrics.csv
  results/processed/trade_stats.csv
  results/raw/fold_windows.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/opt/data/kvab/code/kronos")
from harness import metrics, simulate, trade_stats  # noqa: E402
from signals import (TOP_K, equal_weight, kronos_weights, momentum_weights,  # noqa: E402
                     optimized_weights, ridge_weights)

ROOT = "/opt/data/kvab"
PROC = f"{ROOT}/results/processed"
RAW = f"{ROOT}/results/raw"
WIN = f"{ROOT}/windows"
COST_LEVELS = {"0x": 0.0, "1x": 1.0, "2x": 2.0}
TRAIN, VAL, TEST = 756, 126, 126
START = "2014-01-01"


def cost_series(liq, dates, ret, mult=1.0):
    """Per-name bps/side from the AZB schedule (US: 5 bps top decile, 15 bps other),
    scaled by the sensitivity multiplier. Returns (cost_bps_df, borrow_bps_df)."""
    n = liq.notna().sum(axis=1).replace(0, np.nan)
    thr = (n * 0.10).round()
    is_top = liq.le(thr, axis=0)
    c = pd.DataFrame(np.where(is_top, 5.0, 15.0), index=liq.index, columns=liq.columns)
    c = c.reindex(dates).reindex(columns=ret.columns) * mult
    b = pd.DataFrame(30.0, index=liq.index, columns=liq.columns)
    b = b.reindex(dates).reindex(columns=ret.columns) * mult
    return c, b


def fold_bounds(dates):
    first = dates.searchsorted(pd.Timestamp(START))
    out, k, t0 = [], 0, first
    while t0 + TEST <= len(dates):
        out.append({"fold": int(k), "a": int(t0), "b": int(t0 + TEST),
                    "test_start": str(dates[t0].date()),
                    "test_end": str(dates[t0 + TEST - 1].date()),
                    "train_start": str(dates[t0 - TRAIN - VAL].date()),
                    "train_end": str(dates[t0 - VAL - 1].date())})
        k += 1
        t0 += TEST
    return out


def azb_continuous(dates, seeds=(42,)):
    """Stitch AZB fold windows into one continuous weight matrix (per seed)."""
    allw = {}
    for sd in seeds:
        frames = []
        for f in sorted(os.listdir(WIN)):
            if not f.startswith(f"w_fold") or not f.endswith(f"_seed{sd}.parquet"):
                continue
            w = pd.read_parquet(f"{WIN}/{f}")
            frames.append(w)
        if not frames:
            continue
        W = pd.concat(frames).sort_index()
        W = W[~W.index.duplicated(keep="first")]
        allw[sd] = W
    return allw


def main():
    panel = pd.read_parquet(f"{PROC}/panel.parquet")
    if set(map(str, panel.columns.get_level_values(0))) != {"Close", "Open", "High", "Low", "Volume"}:
        panel = panel.swaplevel(0, 1, axis=1).sort_index(axis=1)
    ret = pd.read_parquet(f"{PROC}/ret.parquet")
    bench = pd.read_parquet(f"{PROC}/bench.parquet")
    liq = pd.read_parquet(f"{PROC}/liq_rank.parquet")
    uni = pd.read_parquet(f"{PROC}/universe.parquet")
    colset = list(ret.columns)
    dates = pd.DatetimeIndex(ret.index)

    folds = fold_bounds(dates)
    json.dump(folds, open(f"{RAW}/fold_windows.json", "w"), indent=1)
    print("folds", len(folds), flush=True)

    # top-100 by liquidity -> investable set applied to all arms
    top100 = liq.le(TOP_K * 5).reindex(columns=colset)
    uni100 = (uni.reindex(columns=colset).fillna(False)
              & top100.reindex(index=uni.index).fillna(False))

    arms = {}
    arms["momentum"] = momentum_weights(panel["Close"], uni100)
    print("momentum done", arms["momentum"].shape, flush=True)
    arms["ridge"] = ridge_weights(panel["Close"], panel["Close"].pct_change(5, fill_method=None), uni100)
    print("ridge done", arms["ridge"].shape, flush=True)
    arms["equal"] = equal_weight(uni100)
    arms["maxsharpe"] = optimized_weights(panel["Close"], uni100, "maxsharpe")
    print("maxsharpe done", flush=True)
    arms["mincorr"] = optimized_weights(panel["Close"], uni100, "mincorr")
    print("mincorr done", flush=True)

    kp = pd.read_parquet(f"{RAW}/kronos_pred.parquet")
    arms["kronos"] = kronos_weights(kp, uni100)
    print("kronos weights", arms["kronos"].shape, flush=True)

    azb = azb_continuous(dates)
    print("azb seeds available", list(azb), flush=True)

    # index B&H: 100% in ^GSPC
    idxw = pd.DataFrame(0.0, index=dates, columns=colset)
    idxw["__IDX__"] = 1.0
    arms["index_ref"] = idxw

    out = {"gross": {}, "net": {}, "trade": {}, "foldrows": []}
    for level, mult in COST_LEVELS.items():
        cb, bb = cost_series(liq, dates, ret, mult)
        for name, W in list(arms.items()) + [(f"azb_s{s}", w) for s, w in azb.items()]:
            if name.startswith("azb"):
                tag = f"{name}"
            else:
                tag = name
            # index arm: use benchmark return directly
            if tag == "index_ref":
                r = bench["ret"].reindex(dates).fillna(0.0)
                net = r.copy()
                sim = pd.DataFrame({"net": net, "gross": net, "turnover": 0.0,
                                    "trade_cost": 0.0, "borrow_cost": 0.0,
                                    "long": 1.0, "short": 0.0,
                                    "gross_exposure": 1.0, "net_exposure": 1.0})
            else:
                Wf = W.reindex(columns=colset).fillna(0.0)
                sim = simulate(Wf, ret, cb, bb)
            g, n = {}, {}
            per_fold = {}
            for f in folds:
                sl = sim.iloc[f["a"]:f["b"]]
                per_fold[f["fold"]] = sl
            net_all = pd.concat([per_fold[f["fold"]]["net"] for f in folds])
            gross_all = pd.concat([per_fold[f["fold"]]["gross"] for f in folds])
            out["net"][f"{tag}_{level}"] = net_all
            out["gross"][f"{tag}_{level}"] = gross_all
            if tag != "index_ref":
                ts = trade_stats(pd.concat([per_fold[f["fold"]] for f in folds]))
                ts["cost_level"] = level
                out["trade"][f"{tag}_{level}"] = ts
            m = metrics(net_all, bench["ret"].reindex(net_all.index))
            m["model"], m["cost_level"] = tag, level
            out["foldrows"].append(m)

    netdf = pd.DataFrame(out["net"])
    grossdf = pd.DataFrame(out["gross"])
    netdf.to_parquet(f"{PROC}/daily_net.parquet")
    grossdf.to_parquet(f"{PROC}/daily_gross.parquet")
    met = pd.DataFrame(out["foldrows"]).set_index(["model", "cost_level"])
    met.to_csv(f"{PROC}/metrics.csv")
    pd.DataFrame(out["trade"]).T.to_csv(f"{PROC}/trade_stats.csv")
    print(met[["cagr", "sharpe", "sortino", "max_dd"]].round(3).to_string(), flush=True)
    print("period", netdf.index[0].date(), netdf.index[-1].date(), "days", len(netdf))


if __name__ == "__main__":
    main()
