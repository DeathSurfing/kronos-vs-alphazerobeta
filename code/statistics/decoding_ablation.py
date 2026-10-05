"""Ablation: effect of Kronos decoding configuration on the forecast signal.

Compares the reported run (T=1.0, single sample) against the model's own
recommended price-forecasting setting (T=0.6, 5 averaged samples) over the
overlapping window, so the reported configuration is measured rather than
merely confessed.

Writes results/tables/T12_decoding_ablation.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/opt/data/kvab"
RAW, PROC, TAB = f"{ROOT}/results/raw", f"{ROOT}/results/processed", f"{ROOT}/results/tables"
sys.path.insert(0, f"{ROOT}/code/statistics")
from significance import forecast_metrics, rank_ic  # noqa: E402


def evaluate(pred_path, close, label):
    p = pd.read_parquet(pred_path)
    px = close.stack()
    px.index.names = ["date", "ticker"]
    d = p.set_index(["date", "ticker"]).join(px.rename("actual")).reset_index()
    d["actual"] = d["actual"].astype(float)
    m = forecast_metrics(d["pred_close"], d["actual"], d["prev_close"])
    d["pr"] = d["pred_close"] / d["prev_close"] - 1
    d["ar"] = d["actual"] / d["prev_close"] - 1
    ic = d.groupby("date").apply(lambda x: rank_ic(x["pr"], x["ar"]), include_groups=False)
    ic = ic.dropna()
    m["rank_ic_mean"] = float(ic.mean()) if len(ic) else np.nan
    m["rank_ic_t"] = (float(ic.mean() / (ic.std() / np.sqrt(len(ic))))
                      if len(ic) > 1 and ic.std() > 0 else np.nan)
    m["pred_ret_std"] = float(d["pr"].std())
    m["config"] = label
    m["n_days"] = int(d["date"].nunique())
    return m, d


def main():
    close = pd.read_parquet(f"{PROC}/panel.parquet")
    if set(map(str, close.columns.get_level_values(0))) != {"Close", "Open", "High", "Low", "Volume"}:
        close = close.swaplevel(0, 1, axis=1).sort_index(axis=1)
    close = close["Close"]

    a, da = evaluate(f"{RAW}/kronos_pred.parquet", close, "T=1.0, N=1 (reported run)")
    b, db = evaluate(f"{RAW}/kronos_pred_reco.parquet", close, "T=0.6, N=5 (model's recommendation)")

    # restrict the reported run to the same dates for a like-for-like comparison
    common = sorted(set(da["date"]) & set(db["date"]))
    dac = da[da["date"].isin(common)]
    dbc = db[db["date"].isin(common)]
    pa = pd.read_parquet(f"{RAW}/kronos_pred.parquet")
    pb = pd.read_parquet(f"{RAW}/kronos_pred_reco.parquet")
    shared = (pa.set_index(["date", "ticker"]).index
              .intersection(pb.set_index(["date", "ticker"]).index))
    rows = []
    for tag, d in [("T=1.0, N=1 (reported)", pa), ("T=0.6, N=5 (recommended)", pb)]:
        dd = d.set_index(["date", "ticker"]).loc[shared].reset_index()
        px = close.stack(); px.index.names = ["date", "ticker"]
        dd = dd.join(px.rename("actual"), on=["date", "ticker"]).dropna(subset=["actual"])
        m = forecast_metrics(dd["pred_close"], dd["actual"].astype(float), dd["prev_close"])
        dd["pr"] = dd["pred_close"] / dd["prev_close"] - 1
        dd["ar"] = dd["actual"] / dd["prev_close"] - 1
        ic = dd.groupby("date").apply(lambda x: rank_ic(x["pr"], x["ar"]), include_groups=False).dropna()
        m["rank_ic_mean"] = float(ic.mean()) if len(ic) else np.nan
        m["rank_ic_t"] = (float(ic.mean() / (ic.std() / np.sqrt(len(ic))))
                          if len(ic) > 1 and ic.std() > 0 else np.nan)
        m["pred_ret_std"] = float(dd["pr"].std())
        m["config"] = tag
        rows.append(m)
    df = pd.DataFrame(rows)
    df = df[["config", "n", "mae", "rmse", "dir_acc", "mae_skill", "rank_ic_mean",
             "rank_ic_t", "pred_ret_std"]]
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(f"{TAB}/T12_decoding_ablation.csv", index=False)
    print(f"compared on {len(shared)} shared (date,ticker) forecasts")
    print(df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
