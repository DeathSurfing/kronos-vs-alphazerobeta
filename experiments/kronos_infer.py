"""Kronos zero-shot inference: next-day close forecast for every name in the
S&P 500 panel, batched per test date. Runs on the remote GPU host.

Leakage: for date t we feed only bars up to and including t-1 (context window
ends at t-1), and predict the bar at t. So pred_close[t] is formed purely from
information available before t opens.

Output: results/raw/kronos_pred.parquet with (date, ticker, pred_close, prev_close)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

CTX_DAYS = 400
PRED_LEN = 1
N_SAMPLES = 1
TEMPERATURE = 1.0
TOPP = 0.9
BATCH = 128


def load_panel(path):
    df = pd.read_parquet(path)
    lv0 = set(map(str, df.columns.get_level_values(0)))
    if "Close" not in lv0:
        df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    df.columns.names = ["Price", "Ticker"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=r"C:\Users\vikk\kvab\panel.parquet")
    ap.add_argument("--liq-rank", default=r"C:\Users\vikk\kvab\liq_rank.parquet")
    ap.add_argument("--out", default=r"C:\Users\vikk\kvab\kronos_pred.parquet")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--topn", type=int, default=100)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=1)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(dev).eval()
    pred = KronosPredictor(model, tok, device=dev, max_context=512)
    print("device", dev, "params", sum(p.numel() for p in model.parameters()), flush=True)

    df = load_panel(args.panel)
    liq = pd.read_parquet(args.liq_rank)
    dates = df.index
    te = dates[(dates >= args.start) & (dates <= args.end)]

    rows = []
    if os.path.exists(args.out):
        prev = pd.read_parquet(args.out)
        rows = prev.to_dict("records")
        print("resuming with", len(rows), "rows", flush=True)

    t0 = time.time()
    for d in te:
        pos = dates.searchsorted(d)
        if pos < CTX_DAYS + 1:
            continue
        if d not in liq.index:
            continue
        cand = liq.loc[d]
        cand = cand[cand <= args.topn].index
        cand = [c for c in cand if c in df["Close"].columns]
        ctx = df.iloc[pos - CTX_DAYS:pos]              # bars up to and including t-1
        x_ts = pd.Series(ctx.index)
        y_ts = pd.Series(pd.DatetimeIndex([d]))
        frames, names = [], []
        for c in cand:
            x = pd.DataFrame({
                "open": ctx["Open"][c].values, "high": ctx["High"][c].values,
                "low": ctx["Low"][c].values, "close": ctx["Close"][c].values,
                "volume": ctx["Volume"][c].values,
                "amount": (ctx["Close"][c] * ctx["Volume"][c]).values,
            }).astype("float64")
            if x.isna().any().any() or (x["close"] <= 0).any():
                continue
            frames.append(x)
            names.append(c)
        for i in range(0, len(frames), BATCH):
            fb = frames[i:i + BATCH]
            nb = names[i:i + BATCH]
            try:
                out = pred.predict_batch(df_list=fb,
                                         x_timestamp_list=[x_ts] * len(fb),
                                         y_timestamp_list=[y_ts] * len(fb),
                                         pred_len=PRED_LEN, T=args.temp,
                                         top_p=TOPP, sample_count=args.samples,
                                         verbose=False)
            except Exception as e:
                print("BATCHERR", str(e)[:80], flush=True)
                continue
            for c, o in zip(nb, out):
                prev_close = float(df["Close"][c].iloc[pos - 1])
                rows.append({"date": d, "ticker": c,
                             "pred_close": float(o["close"].iloc[0]),
                             "prev_close": prev_close})
        if len(rows) and len(rows) % 5000 < BATCH * 2:
            pd.DataFrame(rows).drop_duplicates(["date", "ticker"]).to_parquet(args.out)
            print(f"{d.date()} rows={len(rows)} t={time.time()-t0:.0f}s", flush=True)

    res = pd.DataFrame(rows).drop_duplicates(["date", "ticker"])
    res.to_parquet(args.out)
    print("TOTAL", len(res), "elapsed", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
