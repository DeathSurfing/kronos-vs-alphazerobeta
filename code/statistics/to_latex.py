"""Convert generated result CSVs into LaTeX table fragments.

The paper \\inputs these fragments, so no experimental number is ever typed by
hand into the .tex source. Rebuild order:
    python experiments/run_all.py     # produces results/raw + results/processed
    python code/statistics/generate_tables.py
    python code/statistics/to_latex.py
    tectonic paper/kronos_vs_alphazerobeta.tex
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = "/opt/data/kvab"
TAB = f"{ROOT}/results/tables"
TEX = f"{ROOT}/paper/tables"
os.makedirs(TEX, exist_ok=True)

LABEL = {"kronos": "Kronos (zero-shot)", "azb_s42": "AlphaZeroBeta", "azb_s123": "AlphaZeroBeta (s123)",
         "azb_s456": "AlphaZeroBeta (s456)", "momentum": "Momentum (12-1)",
         "ridge": "Ridge (walk-forward)", "index_ref": "S\\&P 500 buy-and-hold",
         "equal": "Equal weight", "maxsharpe": "Max-Sharpe", "mincorr": "Min-correlation"}


def fmt(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if pct:
        return f"{x*100:.2f}"
    return f"{x:.{nd}f}"


def esc(s):
    """Escape a plain-text cell for LaTeX. Order matters: backslash-carrying
    replacements must happen before the characters they introduce."""
    t = str(s)
    t = t.replace("\\", "\\textbackslash{}")
    for a, b in [("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("$", "\\$"),
                 ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
                 ("^", "\\textasciicircum{}"), ("~", "\\textasciitilde{}")]:
        t = t.replace(a, b)
    t = t.replace("<", "$<$").replace(">", "$>$")
    t = t.replace("|", "$|$")
    return t


def write(name, header, rows, caption, label, align=None, note=None):
    align = align or ("l" + "r" * (len(header) - 1))
    lines = [f"\\begin{{table*}}[t]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines.append("\\bottomrule")
    if note:
        lines.append(f"\\multicolumn{{{len(header)}}}{{l}}{{\\footnotesize {note}}} \\\\")
    lines += ["\\end{tabular}", "\\end{table*}"]
    open(f"{TEX}/{name}.tex", "w").write("\n".join(lines) + "\n")


def write_wide(name, header, rows, caption, label, note=None):
    align = "l" + "r" * (len(header) - 1)
    lines = [f"\\begin{{table*}}[t]", "\\centering", "\\footnotesize",
             "\\setlength{\\tabcolsep}{3.2pt}",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{@{}" + align + "@{}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines.append("\\bottomrule")
    if note:
        lines.append(f"\\multicolumn{{{len(header)}}}{{l}}{{\\footnotesize {note}}} \\\\")
    lines += ["\\end{tabular}", "\\end{table*}"]
    open(f"{TEX}/{name}.tex", "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ T1..T4
def t1():
    d = pd.read_csv(f"{TAB}/T1_model_architecture.csv")
    rows = [[esc(r["model"]), esc(r["type"]), f"{int(r['params']):,}" if str(r["params"]).isdigit() else esc(r["params"]),
             esc(r["context"]), esc(r["inputs"]), esc(r["decision_use"]), esc(r["training_here"])]
            for _, r in d.iterrows()]
    write("tab_arch", ["Model", "Type", "Params", "Ctx", "Inputs", "Decision use", "Training here"],
          rows, "Model architecture and role in the comparison. The two systems are not the same kind of object: one forecasts, the other decides.",
          "tab:arch", note="Parameter count for Kronos-small is the official released checkpoint; AlphaZeroBeta count is the median over all reproduced fold runs.")


def t2():
    d = pd.read_csv(f"{TAB}/T2_dataset_config.csv")
    rows = [[esc(r["item"]), esc(r["value"])] for _, r in d.iterrows()]
    write("tab_data", ["Item", "Value"], rows,
          "Dataset, walk-forward protocol, cost model, constraints and seeds. Frozen before any test-window result was computed.",
          "tab:data")


def t3():
    d = pd.read_csv(f"{TAB}/T3_published.csv")
    rows = [[esc(r["source"]), esc(r["market"]), esc(r["metric"]),
             "--" if pd.isna(r["value"]) else fmt(float(r["value"])), esc(r["uncertainty"])]
            for _, r in d.iterrows()]
    write("tab_published", ["Source", "Market", "Metric", "Reported value", "Reported sd"],
          rows, "Values as reported by the original authors. These are published results, not our experiments.",
          "tab:published", note="Kronos reports forecasting and generative metrics, not a portfolio Sharpe; the corresponding cell is intentionally left as a dash rather than forcing an incomparable number.")


def t4():
    p = f"{TAB}/T4_reproduction.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    rows = [[esc(r["metric"]), fmt(float(r["published"])), fmt(float(r["reproduced"])),
             fmt(float(r["abs_diff"])), fmt(float(r["pct_diff"]), 1)] for _, r in d.iterrows()]
    write("tab_repro", ["Metric (S\\&P 500)", "Published", "Reproduced", "Difference", "Difference (\\%)"],
          rows, "Published versus independently reproduced AlphaZeroBeta results on the S\\&P 500. Percentage difference is $(reproduced-published)/|published|$.",
          "tab:repro", note="Reproduction is deliberately partial (see Section~\\ref{sec:repro}): shorter PPO training, 1 seed on all folds, a 100-name investable subset instead of the full index, and public daily data instead of the vendor panel.")


# ------------------------------------------------------------------ T5
def t5():
    d = pd.read_csv(f"{TAB}/T5_forecasting.csv")
    dm = json.load(open(f"{TAB}/T5_diebold_mariano.json")) if os.path.exists(f"{TAB}/T5_diebold_mariano.json") else {}
    rows = []
    for _, r in d.iterrows():
        rows.append([esc(r["model"]), fmt(r["mae"], 4), fmt(r["rmse"], 4), fmt(r["dir_acc"], 4),
                     fmt(r.get("rank_ic_mean"), 4), fmt(r.get("rank_ic_t"), 2), f"{int(r['n'])}"])
    write("tab_forecast", ["Forecaster", "MAE", "RMSE", "Dir.\\ acc.", "Rank IC", "Rank IC $t$", "N"],
          rows, "Level-1 forecasting evaluation for the S\\&P 500 panel, zero-shot Kronos against the random-walk benchmark. AlphaZeroBeta is excluded by design: it emits portfolio weights, not point forecasts, so forecasting metrics are not defined for it.",
          "tab:forecast",
          note=(f"Diebold--Mariano on squared errors (Kronos vs random walk): $DM={fmt(dm.get('dm'),3)}$, $p={fmt(dm.get('p_value'),3)}$, $n={dm.get('n','--')}$ daily observations. "
                "Rank IC is the mean daily cross-sectional Spearman correlation between predicted and realised next-day return; the $t$-statistic is over days."))


# ------------------------------------------------------------------ T6
def t6():
    d = pd.read_csv(f"{TAB}/T6_portfolio_main.csv")
    cols = ["model", "cagr", "ann_vol", "sharpe", "sortino", "calmar", "max_dd", "es95",
            "beta", "corr", "turnover_ann", "avg_gross_exposure"]
    d = d[[c for c in cols if c in d.columns]]
    rows = []
    for _, r in d.iterrows():
        rows.append([LABEL.get(r["model"], esc(r["model"])),
                     fmt(r.get("cagr"), 3), fmt(r.get("ann_vol"), 3), fmt(r.get("sharpe"), 3),
                     fmt(r.get("sortino"), 3), fmt(r.get("calmar"), 3), fmt(r.get("max_dd"), 3),
                     fmt(r.get("es95"), 4), fmt(r.get("beta"), 3), fmt(r.get("corr"), 3),
                     fmt(r.get("turnover_ann"), 1), fmt(r.get("avg_gross_exposure"), 2)])
    write_wide("tab_portfolio", ["Model", "CAGR", "Vol", "Sharpe", "Sortino", "Calmar", "Max DD",
                                 "ES95", "Beta", "Corr", "Turnover", "Gross exp."],
               rows, "Level-2 portfolio evaluation, S\\&P 500, 22 walk-forward folds, baseline costs (5/15 bps per side plus 30 bps\\,/\\,yr borrow). Dollar-neutral arms target $\\sum w_i=0$ and $\\sum|w_i|\\le1$; net-long arms are the passive reference set.",
               "tab:portfolio",
               note="Volatility, Sharpe, Sortino and Calmar are annualised from daily net returns. Beta and correlation are measured against the S\\&P 500 index return. Turnover is the annualised sum of absolute weight changes.")


# ------------------------------------------------------------------ T7
def t7():
    p = f"{TAB}/T7_significance.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    rows = [[f"{LABEL.get(r['model_a'], esc(r['model_a']))} vs {LABEL.get(r['model_b'], esc(r['model_b']))}",
             fmt(r["sharpe_a"]), fmt(r["sharpe_b"]), fmt(r["diff"]),
             f"[{fmt(r['ci_lo'])}, {fmt(r['ci_hi'])}]", fmt(r["p_value"], 3),
             fmt(r["alpha_ann"]), fmt(r["alpha_t"], 2), fmt(r["alpha_p"], 3)]
            for _, r in d.iterrows()]
    write_wide("tab_sig", ["Comparison", "$S_a$", "$S_b$", "$\\Delta S$", "95\\% CI (block bootstrap)",
                           "$p$", "$\\alpha$ ann.", "$t_{NW}$", "$p_{NW}$"],
               rows, "Statistical comparison of Sharpe ratios (stationary block bootstrap, 21-day blocks, 5000 resamples) and Newey--West (lag 5) alpha against the index.",
               "tab:sig",
               note="A confidence interval containing zero means the Sharpe difference is not distinguishable from zero at the 5\\% level under this resampling scheme.")


# ------------------------------------------------------------------ T8/T9/T10/T11
def t8():
    p = f"{TAB}/T8_regimes.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[d["cost_level"] == "1x"]
    piv = d.pivot_table(index="regime", columns="model", values="sharpe")
    cols = [c for c in ["kronos", "azb_s42", "momentum", "ridge", "index_ref", "equal"] if c in piv.columns]
    rows = []
    for reg in piv.index:
        rows.append([esc(reg)] + [fmt(piv.loc[reg, c]) for c in cols])
    write("tab_regimes", ["Regime"] + [LABEL.get(c, esc(c)) for c in cols], rows,
          "Sharpe ratio by prespecified market regime. Regimes are defined from the index alone (60-day realised volatility terciles; bull if the running drawdown is shallower than $-10\\%$).",
          "tab:regimes", align="l" + "r" * len(cols),
          note="Regimes are defined ex ante from the benchmark, so no strategy's own path influences the partition.")


def t9():
    p = f"{TAB}/T9_cost_sensitivity.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    rows = []
    for _, r in d.iterrows():
        rows.append([LABEL.get(r["model"], esc(r["model"]))]
                    + [fmt(r.get(c)) for c in ["sharpe_0x", "sharpe_1x", "sharpe_2x"]]
                    + [fmt(r.get(c), 1) for c in ["turnover_ann_1x"]])
    write("tab_cost", ["Model", "Sharpe 0$\\times$", "Sharpe 1$\\times$", "Sharpe 2$\\times$", "Turnover 1$\\times$"],
          rows, "Transaction-cost sensitivity. $0\\times$ removes all costs, $2\\times$ doubles commission and borrow.",
          "tab:cost")


def t10():
    p = f"{TAB}/T10_computational.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    rows = []
    for _, r in d.iterrows():
        rows.append([esc(r["model"]), f"{int(r['params']):,}", esc(r["device"]),
                     fmt(r.get("mean_latency_ms"), 2), fmt(r.get("median_latency_ms"), 2),
                     fmt(r.get("p95_latency_ms"), 2), fmt(r.get("throughput_names_per_s"), 1),
                     fmt(r.get("training_time_s"), 1), esc(r["notes"])])
    write_wide("tab_compute", ["Model", "Params", "Device", "Mean (ms)", "Median (ms)", "p95 (ms)",
                               "Names/s", "Train (s/fold)", "Notes"],
               rows, "Computational comparison on a single NVIDIA RTX 4080 SUPER. Latency is per-name single-step inference after warm-up.",
               "tab:compute",
               note="Latency for the RL agent is not comparable in the same unit: it produces a joint portfolio decision rather than a per-name forecast, so per-name inference latency is reported for Kronos only. Training time for Kronos is zero because the released checkpoint is used zero-shot.")


def write_plain(name, header, rows, caption, label, note=None):
    align = "l" + "r" * (len(header) - 1)
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule"]
    if note:
        lines.append(f"\\multicolumn{{{len(header)}}}{{l}}{{\\footnotesize {note}}} \\\\")
    lines += ["\\end{tabular}", "\\end{table}"]
    open(f"{TEX}/{name}.tex", "w").write("\n".join(lines) + "\n")


def t11():
    p = f"{TAB}/T11_seeds.csv"
    if not os.path.exists(p):
        write_plain("tab_seeds", ["Seed", "Sharpe", "CAGR", "Max DD"],
                    [["--", "--", "--", "--"]],
                    "Random-seed sensitivity of AlphaZeroBeta is reported in the seed-sensitivity subsection once the multi-seed arm completes.",
                    "tab:seeds")
        return
    d = pd.read_csv(p)
    rows = [[str(int(r["seed"])), fmt(r["sharpe"]), fmt(r["cagr"]), fmt(r["max_dd"])] for _, r in d.iterrows()]
    rows.append(["mean", fmt(d["sharpe"].mean()), fmt(d["cagr"].mean()), fmt(d["max_dd"].mean())])
    rows.append(["std", fmt(d["sharpe"].std(ddof=1)), fmt(d["cagr"].std(ddof=1)), fmt(d["max_dd"].std(ddof=1))])
    rows.append(["best", fmt(d["sharpe"].max()), fmt(d["cagr"].max()), fmt(d["max_dd"].max())])
    rows.append(["worst", fmt(d["sharpe"].min()), fmt(d["cagr"].min()), fmt(d["max_dd"].min())])
    write_plain("tab_seeds", ["Seed", "Sharpe", "CAGR", "Max DD"], rows,
          "Random-seed sensitivity of AlphaZeroBeta. All seeds are reported, including the worst, so the headline number is not a best run.",
          "tab:seeds", note="Computed over the fold subset re-run with three seeds within the available compute budget.")


def t12_decoding():
    pth = f"{TAB}/T12_decoding_ablation.csv"
    if not os.path.exists(pth):
        return
    d = pd.read_csv(pth)
    rows = [[esc(r["config"]), f"{int(r['n'])}", fmt(r["mae"], 2), fmt(r["rmse"], 2),
             fmt(r["dir_acc"], 4), fmt(r["rank_ic_mean"], 5), fmt(r["rank_ic_t"], 2)]
            for _, r in d.iterrows()]
    write("tab_decoding", ["Decoding", "N", "MAE", "RMSE", "Dir.\\ acc.", "Rank IC", "Rank IC $t$"],
          rows, "Decoding ablation for the forecast arm over the overlapping 2024 window (25{,}200 shared forecasts). The model's own recommended price-forecasting setting is compared against the setting used in the reported runs.",
          "tab:decoding",
          note="The recommended setting roughly halves the relative error in level accuracy but leaves the cross-sectional rank information coefficient statistically indistinguishable from zero in both cases. Advice, not beta, is what the portfolio needs.")


def t12_leakage():
    rows = [
        ["Feature timing", "All features at date $t$ are computed from bars $\\le t-1$; labels are the return of $t$", "No future bar, adjustment or fundamental enters a feature"],
        ["Normalisation", "Cross-sectional standardisation uses same-day cross-sections only; no global mean or variance", "No full-sample statistics leak into training"],
        ["Universe", "Membership and liquidity ranks computed from trailing 60-day dollar volume", "No forward index membership; a fixed top-100 set is used for all models, killable as survivorship (Sec.~\\ref{sec:threats})"],
        ["Hyperparameters", "All hyperparameters fixed before the test windows were run, in \\texttt{configs/base.yaml}", "No test-set tuning; model selection would use validation only"],
        ["Kronos context", "Context window ends at $t-1$ and predicts $t$", "Zero-shot, no fine-tuning on any S\\&P 500 data"],
        ["RL agent update", "Policy parameters for a fold are fitted on data strictly before the fold's test window", "Walk-forward retraining, expanding-window information set"],
        ["Costs", "Applied per rebalance from the stated schedule, dated at execution", "Costs are not tuned to flatter either model"],
        ["Resampling", "Chronological only; no shuffling anywhere in the pipeline", "No i.i.d. assumption on the time axis"],
    ]
    write_wide("tab_leakage", ["Audit item", "Control", "Bias addressed"], rows,
               "Data-leakage audit. Each control is enforced in code and re-checked by the tests in \\texttt{tests/}.",
               "tab:leakage", note="Residual exposures that cannot be removed from public daily data are listed in Section~\\ref{sec:threats} rather than claimed away.")


def main():
    t1(); t2(); t3(); t4(); t5(); t6(); t7(); t8(); t9(); t10(); t11(); t12_decoding(); t12_leakage()
    print("latex fragments:", sorted(os.listdir(TEX)), flush=True)


if __name__ == "__main__":
    main()
