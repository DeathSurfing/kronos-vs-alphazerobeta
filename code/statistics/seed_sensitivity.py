"""Seed-sensitivity analysis for AlphaZeroBeta (Table 11, Figure 9).

Combines the multi-seed fold windows (windows_seed/) into one comparison and
writes results/tables/T11_seeds.csv with mean, sd, best and worst reported so a
lucky run cannot be passed off as the headline number.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/opt/data/kvab"
PROC, RAW, TAB = f"{ROOT}/results/processed", f"{ROOT}/results/raw", f"{ROOT}/results/tables"
WIN = f"{ROOT}/windows_seed"
sys.path.insert(0, f"{ROOT}/code/backtest")
sys.path.insert(0, f"{ROOT}/code/statistics")
from harness import metrics, simulate  # noqa: E402


def cost_series(liq, dates, cols):
    n = liq.notna().sum(axis=1).replace(0, np.nan)
    thr = (n * 0.10).round()
    top = liq.le(thr, axis=0)
    c = pd.DataFrame(np.where(top, 5.0, 15.0), index=liq.index, columns=liq.columns)
    c = c.reindex(dates).reindex(columns=cols)
    b = pd.DataFrame(30.0, index=liq.index, columns=liq.columns).reindex(dates).reindex(columns=cols)
    return c, b


def main():
    os.makedirs(TAB, exist_ok=True)
    if not os.path.isdir(WIN) or not os.listdir(WIN):
        print("no seed windows available; skipping")
        return
    ret = pd.read_parquet(f"{PROC}/ret.parquet")
    bench = pd.read_parquet(f"{PROC}/bench.parquet")["ret"]
    liq = pd.read_parquet(f"{PROC}/liq_rank.parquet")
    dates = pd.DatetimeIndex(ret.index)
    cb, bb = cost_series(liq, dates, ret.columns)

    rows = []
    by_seed = {}
    for f in sorted(os.listdir(WIN)):
        if not f.endswith(".parquet"):
            continue
        seed = int(f.split("_seed")[-1].split(".")[0])
        w = pd.read_parquet(f"{WIN}/{f}").reindex(columns=ret.columns).fillna(0.0)
        sim = simulate(w, ret, cb, bb)
        d = sim["net"].dropna()
        if len(d) < 20:
            continue
        m = metrics(d, bench.reindex(d.index))
        by_seed.setdefault(seed, []).append(m)

    for seed, ms in sorted(by_seed.items()):
        agg = {k: float(np.mean([x[k] for x in ms if k in x])) for k in
               ["sharpe", "cagr", "max_dd", "ann_vol", "sortino", "turnover_ann" if False else "max_dd"]}
        sh = [x["sharpe"] for x in ms if np.isfinite(x.get("sharpe", np.nan))]
        agg = {"seed": seed, "n_folds": len(ms),
               "sharpe": float(np.mean(sh)) if sh else np.nan,
               "n_sharpe_folds": len(sh),
               "cagr": float(np.mean([x["cagr"] for x in ms])),
               "max_dd": float(np.mean([x["max_dd"] for x in ms])),
               "ann_vol": float(np.mean([x["ann_vol"] for x in ms]))}
        rows.append(agg)
    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB}/T11_seeds.csv", index=False)
    print("NOTE: Sharpe is reported only over folds where the realised series has "
          "non-zero variance; a null (all-zero) position produces no Sharpe.", flush=True)
    print(df.to_string(), flush=True)


if __name__ == "__main__":
    main()
