"""Generate every table in the paper from the experiment outputs.

Inputs : results/processed/{daily_net,daily_gross,metrics,trade_stats}.parquet|csv
         results/raw/kronos_pred.parquet, results/raw/fold_windows.json
Outputs: results/tables/*.csv  (consumed by the LaTeX build; never hand-typed)

Table map
  T1 model_architecture      params/context/config comparison
  T2 dataset_config          universe, period, windows, costs, seeds
  T3 published               reported values from the two source papers
  T4 reproduction            published vs reproduced (S&P 500)
  T5 forecasting             MAE/RMSE/dir-acc/rank-IC + Diebold-Mariano
  T6 portfolio_main          headline metrics per model, baseline costs
  T7 significance            bootstrap Sharpe-difference CIs + NW alpha
  T8 regimes                 vol terciles and bull/bear
  T9 cost_sensitivity       0x/1x/2x
  T10 computational          params, latency, throughput, training time
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/opt/data/kvab"
PROC, RAW, OUT = f"{ROOT}/results/processed", f"{ROOT}/results/raw", f"{ROOT}/results/tables"
sys.path.insert(0, f"{ROOT}/code/statistics")
sys.path.insert(0, f"{ROOT}/code/backtest")
from significance import (diebold_mariano, forecast_metrics, newey_west_alpha,  # noqa: E402
                          rank_ic, sharpe_diff_test)

os.makedirs(OUT, exist_ok=True)
BASE = "1x"
NEUTRAL = ["kronos", "azb_s42", "momentum", "ridge"]
LONG = ["index_ref", "equal", "maxsharpe", "mincorr"]


def load():
    net = pd.read_parquet(f"{PROC}/daily_net.parquet")
    gross = pd.read_parquet(f"{PROC}/daily_gross.parquet")
    met = pd.read_csv(f"{PROC}/metrics.csv", index_col=["model", "cost_level"])
    tr = pd.read_csv(f"{PROC}/trade_stats.csv", index_col=0)
    bench = pd.read_parquet(f"{PROC}/bench.parquet")["ret"]
    return net, gross, met, tr, bench


def t1_architecture():
    rows = [
        {"model": "Kronos-small", "type": "financial foundation model (decoder-only transformer)",
         "params": 24741376, "context": 512, "inputs": "OHLCV + amount",
         "objective": "next-bar K-line generation (self-supervised pre-training)",
         "decision_use": "forecast -> cross-sectional ranking -> weights",
         "training_here": "zero-shot (no fine-tuning)"},
        {"model": "AlphaZeroBeta", "type": "CNN-GRU actor-critic (recurrent PPO)",
         "params": "measured per run", "context": 100, "inputs": "8 past-only price/volume features x 3 resolutions",
         "objective": "reward Eq.8: risk-adjusted excess return - corr penalty - turnover",
         "decision_use": "policy emits dollar-neutral weights directly",
         "training_here": "trained per walk-forward fold"},
    ]
    log = f"{ROOT}/experiment_log.jsonl"
    if os.path.exists(log):
        recs = [json.loads(l) for l in open(log) if l.strip()]
        if recs:
            rows[1]["params"] = int(np.median([r["params"] for r in recs]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/T1_model_architecture.csv", index=False)
    return df


def t2_dataset():
    import yaml
    cfg = yaml.safe_load(open(f"{ROOT}/experiments/configs/base.yaml"))
    folds = json.load(open(f"{RAW}/fold_windows.json"))
    rows = [
        ("market", "S&P 500 (^GSPC)"),
        ("data source", "Yahoo Finance daily bars via yfinance, retrieved 2026-10-05"),
        ("price adjustment", "OHLC scaled by AdjClose/Close (splits+dividends), same-day factor"),
        ("panel", f"{cfg['dataset']['n_tickers']} tickers x {cfg['dataset']['n_days']} trading days"),
        ("sample", f"{cfg['dataset']['start_date']} .. {cfg['dataset']['end_date']}"),
        ("investable set", f"top {cfg['universe']['top_n_liquidity']} by 60d mean dollar volume, identical for all models"),
        ("train window", f"{cfg['walk_forward']['train_days']} trading days (~36 months)"),
        ("validation window", f"{cfg['walk_forward']['val_days']} trading days (~6 months)"),
        ("test window", f"{cfg['walk_forward']['test_days']} trading days (~6 months)"),
        ("step", f"{cfg['walk_forward']['step_days']} trading days (6 months)"),
        ("out-of-sample folds", f"{len(folds)} non-overlapping, {folds[0]['test_start']} .. {folds[-1]['test_end']}"),
        ("rebalance", "daily"),
        ("execution", "weights formed from information <= t-1, applied to the return of t"),
        ("transaction costs", "5 bps/side top liquidity decile, 15 bps/side otherwise (AZB Table D4)"),
        ("borrow cost", "30 bps/yr on short notional, accrued per holding day"),
        ("sensitivity", "0x / 1x / 2x the baseline schedule"),
        ("constraints (neutral arms)", "sum w = 0, sum|w| <= 1, w_i in [-1,1], 20 names per side"),
        ("constraints (long arms)", "sum w = 1, w_i >= 0"),
        ("seeds", "AZB: 1 seed all 22 folds; 3 seeds on folds {0,7,14} for seed sensitivity"),
        ("hardware", "NVIDIA RTX 4080 SUPER 16 GB, Windows 11; CPU pipeline on 4-core container"),
    ]
    df = pd.DataFrame(rows, columns=["item", "value"])
    df.to_csv(f"{OUT}/T2_dataset_config.csv", index=False)
    return df


def t3_published():
    per_market = [1.63, 0.94, 0.86, 1.61, 1.04, 1.48, 1.20]   # published, Table 4
    rows = [
        {"source": "AlphaZeroBeta (Belyakov 2026)", "market": "^GSPC", "metric": "Sharpe", "value": 1.61, "uncertainty": "0.48"},
        {"source": "AlphaZeroBeta (Belyakov 2026)", "market": "^NDX", "metric": "Sharpe", "value": 1.48, "uncertainty": "0.41"},
        {"source": "AlphaZeroBeta (Belyakov 2026)", "market": "^DJI", "metric": "Sharpe", "value": 1.20, "uncertainty": "0.28"},
        {"source": "AlphaZeroBeta (Belyakov 2026)", "market": "^FTSE", "metric": "Sharpe", "value": 0.94, "uncertainty": "0.19"},
        {"source": "AlphaZeroBeta (Belyakov 2026)", "market": "^GDAXI", "metric": "Sharpe", "value": 0.86, "uncertainty": "0.23"},
        {"source": "AlphaZeroBeta (Belyakov 2026)", "market": "^HSI", "metric": "Sharpe", "value": 1.04, "uncertainty": "0.33"},
        {"source": "AlphaZeroBeta (Belyakov 2026)", "market": "000001.SS", "metric": "Sharpe", "value": 1.63, "uncertainty": "0.38"},
        {"source": "AlphaZeroBeta (Belyakov 2026)", "market": "^GSPC", "metric": "max drawdown", "value": -0.26, "uncertainty": "0.15"},
        {"source": "AlphaZeroBeta (Belyakov 2026)", "market": "^GSPC", "metric": "corr to index", "value": 0.15, "uncertainty": "0.09"},
        {"source": "AlphaZeroBeta (Belyakov 2026)", "market": "all 7", "metric": "average Sharpe",
         "value": float(np.mean(per_market)), "uncertainty": "0.30"},
        {"source": "Kronos (Shi et al. 2025)", "market": "various", "metric": "zero-shot forecasting", "value": np.nan,
         "uncertainty": "-"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/T3_published.csv", index=False)
    return df


def t4_reproduction(met):
    """Published vs reproduced for S&P 500, with percentage difference."""
    try:
        pub = float(met.loc[("azb_s42", BASE), "sharpe"])
        pub_mdd = float(met.loc[("azb_s42", BASE), "max_dd"])
        pub_corr = float(met.loc[("azb_s42", BASE), "corr"])
        idx_sharpe = float(met.loc[("index_ref", BASE), "sharpe"])
    except KeyError:
        return pd.DataFrame()
    ref = {"sharpe": (1.61, pub), "max_dd": (-0.26, pub_mdd), "corr": (0.15, pub_corr),
           "index buy-and-hold sharpe": (0.72, idx_sharpe)}
    rows = []
    for k, (p, r) in ref.items():
        rows.append({"metric": k, "published": p, "reproduced": round(r, 4),
                     "abs_diff": round(r - p, 4),
                     "pct_diff": round((r - p) / abs(p) * 100, 1) if p else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/T4_reproduction.csv", index=False)
    return df


def t5_forecasting(bench):
    p = pd.read_parquet(f"{RAW}/kronos_pred.parquet")
    close = pd.read_parquet(f"{PROC}/panel.parquet")["Close"]
    if "Close" not in close.columns.get_level_values(0):
        pass
    px = close.stack() if isinstance(close, pd.DataFrame) else close
    px.index.names = ["date", "ticker"]
    p = p.set_index(["date", "ticker"]).join(px.rename("actual")).reset_index()
    p["actual"] = p["actual"].astype(float)
    m = forecast_metrics(p["pred_close"], p["actual"], p["prev_close"])
    m["model"] = "Kronos-small (zero-shot)"
    m2 = {"model": "Random walk (pred = prev close)",
          "mae": m["rw_mae"], "rmse": m["rw_rmse"],
          "dir_acc": float(np.mean(np.sign(p["actual"] - p["prev_close"]) != 0)),
          "mae_skill": 0.0, "n": m["n"]}
    # DM on squared errors, Kronos vs random walk, aggregated per date
    p["e_k"] = (p["pred_close"] - p["actual"]) ** 2
    p["e_rw"] = (p["prev_close"] - p["actual"]) ** 2
    g = p.groupby("date")[["e_k", "e_rw"]].mean()
    dm = diebold_mariano(g["e_k"], g["e_rw"], h=1)
    # rank IC: cross-sectional spearman of predicted vs realised next-day return
    p["pr"] = p["pred_close"] / p["prev_close"] - 1
    p["ar"] = p["actual"] / p["prev_close"] - 1
    ic = p.groupby("date").apply(lambda d: rank_ic(d["pr"], d["ar"]), include_groups=False)
    m["rank_ic_mean"] = float(ic.mean()); m["rank_ic_std"] = float(ic.std())
    m["rank_ic_t"] = float(ic.mean() / (ic.std() / np.sqrt(len(ic.dropna())))) if ic.std() > 0 else np.nan
    m2["rank_ic_mean"] = np.nan; m2["rank_ic_std"] = np.nan; m2["rank_ic_t"] = np.nan
    df = pd.DataFrame([m, m2])
    df.attrs["dm"] = dm
    df.to_csv(f"{OUT}/T5_forecasting.csv", index=False)
    json.dump(dm, open(f"{OUT}/T5_diebold_mariano.json", "w"), indent=1)
    return df, dm


def t6_portfolio(met, tr):
    df = met.loc[(slice(None), BASE), :].copy()
    df = df.reset_index()
    keep = ["model", "cagr", "ann_vol", "sharpe", "sortino", "calmar", "max_dd",
            "var95", "es95", "beta", "corr", "skew", "n_days"]
    df = df[[c for c in keep if c in df.columns]]
    tt = tr[tr["cost_level"] == BASE] if "cost_level" in tr.columns else tr
    df = df.merge(tt.reset_index().rename(columns={"index": "model"}),
                  on="model", how="left")
    df.to_csv(f"{OUT}/T6_portfolio_main.csv", index=False)
    return df


def t7_significance(net, bench):
    rows = []
    for a, b in [("kronos", "azb_s42"), ("kronos", "index_ref"), ("azb_s42", "index_ref"),
                 ("kronos", "momentum"), ("azb_s42", "momentum"), ("kronos", "ridge"),
                 ("azb_s42", "ridge"), ("kronos", "equal")]:
        ca, cb = f"{a}_{BASE}", f"{b}_{BASE}"
        if ca not in net.columns or cb not in net.columns:
            continue
        t = sharpe_diff_test(net[ca], net[cb], block=21, n_boot=5000, seed=7)
        nw = newey_west_alpha(net[ca], bench.reindex(net.index))
        rows.append({"model_a": a, "model_b": b, "sharpe_a": round(t["sharpe_a"], 3),
                     "sharpe_b": round(t["sharpe_b"], 3), "diff": round(t["diff"], 3),
                     "ci_lo": round(t["ci_lo"], 3), "ci_hi": round(t["ci_hi"], 3),
                     "p_value": round(t["p_value"], 4),
                     "alpha_ann": round(nw.get("alpha_ann", np.nan), 4),
                     "alpha_t": round(nw.get("alpha_t", np.nan), 3),
                     "alpha_p": round(nw.get("alpha_p", np.nan), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/T7_significance.csv", index=False)
    return df


def t8_regimes(net, bench):
    b = bench.reindex(net.index)
    vol = b.rolling(60, min_periods=40).std() * np.sqrt(252)
    q1, q2 = vol.quantile([1 / 3, 2 / 3])
    dd = (1 + b.fillna(0)).cumprod()
    dd = dd / dd.cummax() - 1
    regimes = {"low vol": vol <= q1, "mid vol": (vol > q1) & (vol <= q2),
               "high vol": vol > q2, "bull (dd > -10%)": dd > -0.10,
               "bear (dd <= -10%)": dd <= -0.10}
    rows = []
    for name, mask in regimes.items():
        mask = mask.reindex(net.index).fillna(False)
        for c in net.columns:
            r = net.loc[mask, c].dropna()
            if len(r) < 20:
                continue
            rows.append({"regime": name, "model": c.rsplit("_", 1)[0],
                         "cost_level": c.rsplit("_", 1)[1],
                         "n_days": len(r), "ann_return": r.mean() * 252,
                         "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(252) if r.std() > 0 else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/T8_regimes.csv", index=False)
    return df


def t9_cost(met):
    df = met.reset_index()
    piv = df.pivot_table(index="model", columns="cost_level",
                         values=["sharpe", "cagr", "turnover_ann"] if "turnover_ann" in df.columns
                         else ["sharpe", "cagr"], aggfunc="mean")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv.to_csv(f"{OUT}/T9_cost_sensitivity.csv", index=False)
    return piv


def t10_compute():
    rows = []
    log = f"{ROOT}/experiment_log.jsonl"
    azb_rt = []
    if os.path.exists(log):
        for l in open(log):
            if l.strip():
                r = json.loads(l)
                azb_rt.append(r["runtime_s"])
    lat = f"{RAW}/latency.json"
    kronos = json.load(open(lat)) if os.path.exists(lat) else {}
    rows.append({"model": "Kronos-small", "params": kronos.get("params", 24741376),
                 "device": "RTX 4080 SUPER", "mean_latency_ms": kronos.get("mean_ms"),
                 "median_latency_ms": kronos.get("median_ms"), "p95_latency_ms": kronos.get("p95_ms"),
                 "throughput_names_per_s": kronos.get("throughput"),
                 "training_time_s": 0.0, "notes": "zero-shot; no training"})
    rows.append({"model": "AlphaZeroBeta", "params": 1703587, "device": "RTX 4080 SUPER",
                 "mean_latency_ms": np.nan, "median_latency_ms": np.nan, "p95_latency_ms": np.nan,
                 "throughput_names_per_s": np.nan,
                 "training_time_s": round(float(np.mean(azb_rt)), 1) if azb_rt else np.nan,
                 "notes": f"mean per fold over {len(azb_rt)} fold runs"})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/T10_computational.csv", index=False)
    return df


def main():
    net, gross, met, tr, bench = load()
    print("models", list(net.columns)[:8], flush=True)
    t1_architecture(); t2_dataset(); t3_published()
    print(t4_reproduction(met).to_string(), flush=True)
    f5, dm = t5_forecasting(bench)
    print(f5.to_string(), flush=True)
    print("DM", dm, flush=True)
    print(t6_portfolio(met, tr).to_string(), flush=True)
    print(t7_significance(net, bench).to_string(), flush=True)
    t8_regimes(net, bench); t9_cost(met); t10_compute()
    print("tables written to", OUT, flush=True)


if __name__ == "__main__":
    main()
