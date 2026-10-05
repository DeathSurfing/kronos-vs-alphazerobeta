"""Generate every figure in the paper from the experiment outputs.

Reads results/processed/daily_net.parquet and results/tables/T*.csv.
Writes vector PDFs to results/figures/ (included by the LaTeX build).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "/opt/data/kvab"
PROC, RAW, FIG, TAB = (f"{ROOT}/results/processed", f"{ROOT}/results/raw",
                       f"{ROOT}/results/figures", f"{ROOT}/results/tables")
os.makedirs(FIG, exist_ok=True)
BASE = "1x"
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 160, "savefig.bbox": "tight"})
LABEL = {"kronos": "Kronos (zero-shot)", "azb_s42": "AlphaZeroBeta", "momentum": "12-1 Momentum",
         "ridge": "Ridge (walk-forward)", "index_ref": "S&P 500 buy-and-hold",
         "equal": "Equal weight", "maxsharpe": "Max-Sharpe", "mincorr": "Min-correlation"}


def load():
    net = pd.read_parquet(f"{PROC}/daily_net.parquet")
    gross = pd.read_parquet(f"{PROC}/daily_gross.parquet")
    return net, gross


def pick(net):
    """Baseline-cost columns present in the run, in a stable display order."""
    order = ["kronos", "azb_s42", "momentum", "ridge", "index_ref", "equal",
             "maxsharpe", "mincorr"]
    cols = []
    for m in order:
        c = f"{m}_{BASE}"
        if c in net.columns:
            cols.append(c)
    return cols


def f1_architecture():
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    ax.axis("off")
    left = ["Kronos-small\n(foundation model)", "zero-shot\nnext-bar forecast",
            "expected-return\nsignal", "cross-sectional\nranking (20/20)",
            "dollar-neutral\nportfolio"]
    right = ["AlphaZeroBeta\n(CNN-GRU actor-critic)", "recurrent PPO\nover market state",
             "reward Eq.8\nrisk - corr - turnover", "policy head\ntanh -> weights",
             "dollar-neutral\nportfolio"]
    for i, (a, b) in enumerate(zip(left, right)):
        for j, txt in enumerate((a, b)):
            x = 0.03 + i * 0.20
            y = 0.62 if j == 0 else 0.10
            ax.add_patch(plt.Rectangle((x, 0.38 + j * 0.10), 0.17, 0.30,
                                       facecolor="#eef3fa" if j == 0 else "#fdf0e6",
                                       edgecolor="black", lw=0.6))
            ax.text(x + 0.085, 0.53 + j * 0.10, txt, ha="center", va="center", fontsize=6.4)
            if i < 4:
                ax.annotate("", xy=(x + 0.20, 0.53 + j * 0.10), xytext=(x + 0.17, 0.53 + j * 0.10),
                            arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.text(0.085, 0.075, "forecast-first pipeline", ha="center", fontsize=6.6, style="italic")
    ax.text(0.085, 0.005, "decision-first pipeline", ha="center", fontsize=6.6, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.02)
    fig.savefig(f"{FIG}/fig1_architecture.pdf")
    plt.close(fig)


def f2_walkforward():
    folds = json.load(open(f"{RAW}/fold_windows.json"))
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    for f in folds[:8]:
        # schematic: train / val / test bars
        ax.barh(f["fold"], 36, left=0, color="#4c72b0", alpha=.8, height=.6)
    ax.set_yticks([f["fold"] for f in folds[:8]])
    ax.set_ylabel("fold")
    ax.set_xlabel("months relative to fold start (3y train -> 6m val -> 6m test)")
    ax.barh(0, 6, left=36, color="#dd8452", height=.6)
    ax.barh(0, 6, left=42, color="#55a868", height=.6)
    ax.text(18, -0.75, "train 36m", color="#4c72b0", ha="center")
    ax.text(39, -0.75, "val 6m", color="#dd8452", ha="center")
    ax.text(45, -0.75, "test 6m", color="#55a868", ha="center")
    ax.set_title("Rolling walk-forward: 3y train / 6m validation / 6m test, 6m step")
    fig.savefig(f"{FIG}/fig2_walk_forward.pdf")
    plt.close(fig)


def f3_equity(net):
    cols = pick(net)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for c in cols:
        eq = (1 + net[c].fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.0, label=LABEL[c.rsplit("_", 1)[0]])
    ax.set_yscale("log")
    ax.set_ylabel("cumulative net value (log)")
    ax.set_title("Out-of-sample equity curves, S&P 500, 22 walk-forward folds (baseline costs)")
    ax.legend(fontsize=6.2, ncol=2)
    fig.savefig(f"{FIG}/fig3_equity_curves.pdf")
    plt.close(fig)


def f4_drawdown(net):
    cols = pick(net)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    for c in cols:
        r = net[c].fillna(0)
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values * 100, lw=0.9, label=LABEL[c.rsplit("_", 1)[0]])
    ax.set_ylabel("drawdown (%)")
    ax.set_title("Drawdown paths (baseline costs)")
    ax.legend(fontsize=6.2, ncol=2)
    fig.savefig(f"{FIG}/fig4_drawdown.pdf")
    plt.close(fig)


def f5_riskreturn():
    p = f"{TAB}/T6_portfolio_main.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for _, r in d.iterrows():
        lbl = LABEL.get(r["model"], r["model"])
        ax.scatter(r["ann_vol"] * 100, r["cagr"] * 100, s=28)
        ax.annotate(lbl, (r["ann_vol"] * 100, r["cagr"] * 100), fontsize=5.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("annualised volatility (%)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("Risk-return, baseline costs")
    fig.savefig(f"{FIG}/fig5_risk_return.pdf")
    plt.close(fig)


def f6_sharpe_by_model():
    p = f"{TAB}/T6_portfolio_main.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p).sort_values("sharpe")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.barh([LABEL.get(m, m) for m in d["model"]], d["sharpe"], color="#4c72b0")
    ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel("Sharpe ratio (net, baseline costs)")
    fig.savefig(f"{FIG}/fig6_sharpe_by_model.pdf")
    plt.close(fig)


def f7_regimes():
    p = f"{TAB}/T8_regimes.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d["cost_level"] == BASE]
    d = d[d["model"].isin(["kronos", "azb_s42", "momentum", "index_ref"])]
    piv = d.pivot_table(index="regime", columns="model", values="sharpe")
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    piv.plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452", "#55a868", "#8172b3"][:piv.shape[1]])
    ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Performance by prespecified regime (baseline costs)")
    ax.legend(fontsize=6.2)
    plt.xticks(rotation=20, ha="right")
    fig.savefig(f"{FIG}/fig7_regimes.pdf")
    plt.close(fig)


def f8_compute():
    p = f"{TAB}/T10_computational.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    axes[0].bar(d["model"], d["params"] / 1e6, color=["#4c72b0", "#dd8452"])
    axes[0].set_ylabel("parameters (millions)")
    axes[0].set_title("Model size")
    lat = f"{RAW}/latency.json"
    if os.path.exists(lat):
        j = json.load(open(lat))
        axes[1].bar(["mean", "median", "p95"], [j["mean_ms"], j["median_ms"], j["p95_ms"]],
                    color="#4c72b0")
        axes[1].set_ylabel("ms per name")
        axes[1].set_title("Kronos zero-shot latency")
    else:
        axes[1].text(.5, .5, "latency not measured", ha="center")
    fig.savefig(f"{FIG}/fig8_compute.pdf")
    plt.close(fig)


def f9_seed_sensitivity():
    p = f"{TAB}/T11_seeds.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.bar([str(s) for s in d["seed"]], d["sharpe"], color="#dd8452")
    ax.axhline(d["sharpe"].mean(), color="k", lw=.8, ls="--",
               label=f"mean {d['sharpe'].mean():.2f}")
    ax.set_xlabel("seed")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("AlphaZeroBeta seed sensitivity (3 folds)")
    ax.legend(fontsize=6.2)
    fig.savefig(f"{FIG}/fig9_seeds.pdf")
    plt.close(fig)


def main():
    net, gross = load()
    f1_architecture(); f2_walkforward(); f3_equity(net); f4_drawdown(net)
    f5_riskreturn(); f6_sharpe_by_model(); f7_regimes(); f8_compute(); f9_seed_sensitivity()
    print("figures written:", sorted(os.listdir(FIG)), flush=True)


if __name__ == "__main__":
    main()
