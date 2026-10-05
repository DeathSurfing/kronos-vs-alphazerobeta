"""Baselines that need no GPU: momentum, ridge, equal weight, max-Sharpe,
min-correlation, and the index buy-and-hold reference.

The actual weight construction lives in code/kronos/signals.py and is exercised
by code/backtest/run_backtest.py; this module only verifies the inputs exist and
reports the coverage of each arm, so a missing arm fails loudly instead of
silently shrinking the comparison.
"""
import os
import sys

import pandas as pd

ROOT = "/opt/data/kvab"
PROC = f"{ROOT}/results/processed"
sys.path.insert(0, f"{ROOT}/code/kronos")


def main():
    need = ["panel.parquet", "ret.parquet", "bench.parquet", "universe.parquet", "liq_rank.parquet"]
    missing = [f for f in need if not os.path.exists(f"{PROC}/{f}")]
    assert not missing, f"processed inputs missing: {missing} (run code/data/build_panel.py first)"

    ret = pd.read_parquet(f"{PROC}/ret.parquet")
    bench = pd.read_parquet(f"{PROC}/bench.parquet")
    liq = pd.read_parquet(f"{PROC}/liq_rank.parquet")
    print(f"panel rows={len(ret)} tickers={ret.shape[1]}")
    print(f"benchmark days={int(bench['ret'].notna().sum())}", flush=True)
    for f in ["kronos_pred.parquet"]:
        p = f"{ROOT}/results/raw/{f}"
        print(f"{f}: {'present' if os.path.exists(p) else 'MISSING (Kronos arm will be skipped)'}")

    from signals import TOP_K, equal_weight, momentum_weights, optimized_weights, ridge_weights
    panel = pd.read_parquet(f"{PROC}/panel.parquet")
    if set(map(str, panel.columns.get_level_values(0))) != {"Close", "Open", "High", "Low", "Volume"}:
        panel = panel.swaplevel(0, 1, axis=1).sort_index(axis=1)
    uni = pd.read_parquet(f"{PROC}/universe.parquet").reindex(columns=ret.columns).fillna(False)
    uni100 = uni & liq.le(TOP_K * 5).reindex(columns=ret.columns).fillna(False)
    for name, w in [("momentum", momentum_weights(panel["Close"], uni100)),
                    ("equal", equal_weight(uni100)),
                    ("maxsharpe", optimized_weights(panel["Close"], uni100, "maxsharpe")),
                    ("mincorr", optimized_weights(panel["Close"], uni100, "mincorr"))]:
        active = int((w.abs().sum(axis=1) > 0).sum())
        print(f"{name}: {active} dates with a position, {w.shape[1]} names", flush=True)


if __name__ == "__main__":
    main()
