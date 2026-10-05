"""Measure Kronos inference latency/throughput on the same GPU host, and record
it for Table 10. Per-name single-step forecasting, batch of 1, warm cache."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402


def main():
    root = os.environ.get("KVAB_ROOT", "/opt/data/kvab")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(dev).eval()
    pred = KronosPredictor(model, tok, device=dev, max_context=512)
    panel = pd.read_parquet(os.path.join(root, "panel.parquet"))
    if set(map(str, panel.columns.get_level_values(0))) != {"Close", "Open", "High", "Low", "Volume"}:
        panel = panel.swaplevel(0, 1, axis=1).sort_index(axis=1)
    close = panel["Close"]
    cols = [c for c in close.columns if close[c].notna().sum() > 2000][:8]
    ts = close.index
    pos = len(ts) - 1
    ctx = panel.iloc[pos - 400:pos]
    x_ts = pd.Series(ctx.index)
    y_ts = pd.Series(pd.DatetimeIndex([ts[-1]]))
    frames = [pd.DataFrame({"open": ctx["Open"][c].values, "high": ctx["High"][c].values,
                            "low": ctx["Low"][c].values, "close": ctx["Close"][c].values,
                            "volume": ctx["Volume"][c].values,
                            "amount": (ctx["Close"][c] * ctx["Volume"][c]).values}).astype("float64")
              for c in cols]
    for _ in range(3):                      # warm-up
        pred.predict_batch(frames[:1], [x_ts], [y_ts], pred_len=1, T=1.0, top_p=0.9,
                           sample_count=1, verbose=False)
    lat = []
    for i in range(60):
        t0 = time.perf_counter()
        pred.predict_batch([frames[i % len(frames)]], [x_ts], [y_ts], pred_len=1, T=1.0,
                           top_p=0.9, sample_count=1, verbose=False)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    import json
    out = {"params": sum(p.numel() for p in model.parameters()), "device": dev,
           "mean_ms": float(lat.mean()), "median_ms": float(np.median(lat)),
           "p95_ms": float(np.percentile(lat, 95)), "throughput": float(1000 / lat.mean()),
           "n": int(len(lat))}
    json.dump(out, open(os.path.join(root, "latency.json"), "w"), indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
