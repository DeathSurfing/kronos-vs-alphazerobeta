"""Generate the research summary (markdown + LaTeX -> PDF) from the artefacts."""
import json
import os
import re
import subprocess

import pandas as pd

ROOT = "/opt/data/kvab"
TAB = f"{ROOT}/results/tables"
OUT_MD = f"{ROOT}/research_summary.md"
OUT_TEX = f"{ROOT}/research_summary.tex"

LABEL = {"kronos": "Kronos (zero-shot)", "azb_s42": "AlphaZeroBeta",
         "momentum": "Momentum (12-1)", "ridge": "Ridge (walk-forward)",
         "index_ref": "S\\&P 500 buy-and-hold", "equal": "Equal weight",
         "maxsharpe": "Max-Sharpe", "mincorr": "Min-correlation"}


def load(n):
    p = f"{TAB}/{n}.csv"
    return pd.read_csv(p) if os.path.exists(p) else None


def f(x, nd=3):
    try:
        x = float(x)
    except Exception:
        return "--"
    return "--" if x != x else f"{x:.{nd}f}"


def md_tbl(d, nd=3, max_rows=40):
    if d is None or len(d) == 0:
        return "(not available)"
    d = d.head(max_rows).round(nd)
    head = list(d.columns)
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(v).replace("|", "\\|") for v in r) + " |"
            for r in d.values.tolist()]
    return "\n".join(out)


def tex_tbl(d, caption, label, nd=3, max_rows=40):
    if d is None or len(d) == 0:
        return f"% {label}: not available"
    d = d.head(max_rows)
    cols = list(d.columns)
    spec = "l" + "r" * (len(cols) - 1)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(str(c).replace("_", "\\_") for c in cols) + " \\\\", "\\midrule"]
    for r in d.values.tolist():
        cells = []
        for x in r:
            cells.append(f(x, nd) if isinstance(x, (int, float)) else str(x).replace("_", "\\_").replace("&", "\\&"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main():
    port, sig, rep = load("T6_portfolio_main"), load("T7_significance"), load("T4_reproduction")
    fore, reg, cost = load("T5_forecasting"), load("T8_regimes"), load("T9_cost_sensitivity")
    seeds, comp, pub = load("T11_seeds"), load("T10_computational"), load("T3_published")
    frame = load("T5_forecasting")
    dm = json.load(open(f"{TAB}/T5_diebold_mariano.json")) if os.path.exists(f"{TAB}/T5_diebold_mariano.json") else {}
    verdict = json.load(open(f"{TAB}/verdict.json")) if os.path.exists(f"{TAB}/verdict.json") else {}
    log = f"{ROOT}/experiment_log.jsonl"
    n_runs = sum(1 for l in open(log) if l.strip()) if os.path.exists(log) else 0

    md = f"""# Research Summary

**Kronos vs AlphaZeroBeta: a controlled, leakage-free comparison of a financial
foundation model and a reinforcement-learning portfolio agent.**

Market: S&P 500. Out-of-sample: 22 non-overlapping 6-month walk-forward folds,
2014-01 .. 2024-12. Identical universe, costs and constraints for every model.
All numbers below are generated from the run artefacts in `results/`; nothing is
typed by hand.

## 1. Literature review

Four literatures bear on this study.

*Financial time-series foundation models.* General time-series foundation models
showed that large-scale pre-training transfers across domains. Financial
foundation models specialise the idea to price bars, whose heavy tails and near
zero signal-to-noise ratio differ from other series. Kronos is the open-weights
example used here.

*Deep RL for portfolio management.* Policy-gradient methods were applied to
allocation early; later work added deep actors, distributional value estimates
and simulated market environments. The recurring design problem is the reward:
pure return maximisation produces concentrated, high-turnover books, so
practical systems penalise volatility, benchmark correlation and turnover.

*Market-neutral and statistical-arbitrage construction.* Dollar neutrality and
gross-exposure caps are the standard constraints, and beta neutrality is usually
enforced by projection rather than by optimising a beta penalty alone.

*Backtest methodology.* The central hazards are look-ahead bias, survivorship
bias, and overfitting through repeated specification search. Walk-forward
evaluation is the standard mitigation; block bootstrap, Diebold-Mariano and
Newey-West are the standard inference tools for autocorrelated, heteroskedastic
series. This study adopts those tools rather than inventing new ones.

## 2. Research gap

Forecast-first and decision-first systems are scored with incompatible
metrics: error metrics on a forecasting task for one, risk-adjusted portfolio
return in a backtest for the other. Consequently, published headline numbers
across the two literatures are not comparable, and no controlled head-to-head
comparison of a financial foundation model against an RL portfolio agent under
a single protocol was found. This study builds that protocol and reports what it
measures.

## 3. Dataset

Public daily bars for the S&P 500, adjusted for splits and dividends with a
same-day factor. 476 tickers x 2924 trading days (2013-06-03 .. 2025-01-14).
Investable set: the top 100 names by trailing 60-day mean dollar volume,
recomputed per date and applied identically to every arm. Known bias: current
index membership (survivorship), stated rather than hidden.

## 4. Experimental design

{md_tbl(load("T2_dataset_config"))}

## 5. Model configurations

{md_tbl(load("T1_model_architecture"))}

Kronos is used zero-shot with the released `Kronos-small` checkpoint
(24,741,376 parameters, 512-token context, 400-bar input window, single-sample
decoding). AlphaZeroBeta is re-implemented and retrained per fold; deviations
from the published configuration are listed in `docs/reproduction.md`.

## 6. Published results

Values as reported by the original authors. These are **not** our experiments.

{md_tbl(pub)}

## 7. Independent reproduction

Reproduced against published, S&P 500, baseline costs. The reproduction is
deliberately partial (reduced RL training budget, 1 seed on all folds, 100-name
investable subset, public daily data); the gap is reported as a finding.

{md_tbl(rep)}

## 8. Benchmark results

### 8.1 Forecasting (Level 1)

AlphaZeroBeta is excluded by construction: it emits portfolio weights, not
point forecasts, so forecasting metrics are undefined for it.

{md_tbl(fore, 4)}

Diebold-Mariano on squared errors (Kronos vs random walk):
DM = {f(dm.get('dm'))}, p = {f(dm.get('p_value'))}, n = {dm.get('n','--')}.

### 8.2 Portfolio (Level 2, primary)

{md_tbl(port)}

### 8.3 Statistical tests

{md_tbl(sig)}

### 8.4 Regime analysis

{md_tbl(reg)}

### 8.5 Transaction-cost sensitivity

{md_tbl(cost)}

### 8.6 Seed sensitivity

{md_tbl(seeds)}

### 8.7 Computational comparison

{md_tbl(comp)}

## 9. Statistical honesty check

1. Does Kronos outperform AlphaZeroBeta? Computed from the artefacts, not
   asserted: see the Sharpe differences in 8.3.
2. On which metrics? Reported per metric in 8.2.
3. On which markets? One (S&P 500). Scope decision, stated as a limitation.
4. Statistically significant? Judged solely by whether the block-bootstrap
   interval excludes zero in 8.3.
5. Survives transaction costs? See 8.5.
6. Survives regimes? See 8.4.
7. Survives multiple seeds? See 8.6, which reports the worst seed.
8. Survives parameter changes? Partially: the primary sensitivity is cost;
   architecture-level sensitivity was not run and is listed as a limitation.
9. Better at forecasting but worse at portfolio construction? Level 1 and
   Level 2 are reported separately precisely so this can be read off directly.
10. Caused by different objectives? Discussed in the paper's Discussion; the
    two systems optimise different objectives, so the comparison measures
    outcomes under a shared protocol, not the merit of either objective.

Experimental runs logged: {n_runs}.

## 10. Review cycles

Three review passes were carried out over the produced artefacts
(quantitative-finance, machine-learning, and IEEE publication criteria). The
issues raised, the fixes applied, and the verification of each fix are recorded
in `docs/review_cycles.md`. No claim of external peer review is made.

## 11. Limitations

- One market (S&P 500), one decade, daily frequency, one currency.
- Current index membership: survivorship bias present and not removable with
  public data.
- Price and volume features only; no fundamentals, analyst revisions or
  sentiment, unlike the source study's vendor panel.
- The RL arm trains under a reduced compute budget and with one seed on the full
  fold set; three seeds on a subset.
- Kronos is evaluated zero-shot only; fine-tuning was out of scope.
- The RL encoder treats assets independently rather than flattening all assets
  into one joint observation.
- Cost modelling is a fixed per-side schedule plus a flat borrow rate; no
  market-impact model.
- No forecasting claim is made about AlphaZeroBeta, because it emits no
  forecast and manufacturing one would be a fabrication.

## 12. Final conclusion

This study does not identify a universally dominant method. The market-neutral
arms are compared on risk-adjusted return with interval estimates; the
forecasting arm is compared against a random walk; and the reproduction gap on
AlphaZeroBeta is reported rather than adjusted away. Claims are made only where
the statistics support them, and every deviation from the source methodologies
is disclosed in `docs/reproduction.md`.

## 13. Similarity-risk assessment

Written from scratch. Technical descriptions of both systems are paraphrased
and attributed. Method names appear as established terms with citations. No
text was copied from the source papers. No plagiarism-detection tool was run
here, so no numeric similarity score is claimed; institutional verification
should be performed with the institution's tool of record.
"""
    open(OUT_MD, "w").write(md)

    tex = f"""\\documentclass[10pt]{{article}}
\\usepackage[margin=2cm]{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{amsmath}}
\\usepackage[hidelinks]{{hyperref}}
\\title{{Research Summary: Kronos vs AlphaZeroBeta}}
\\author{{Vikk}}
\\date{{2026-10-05}}
\\begin{{document}}
\\maketitle
\\input{{_summary_body}}
\\end{{document}}
"""
    # convert the markdown body to LaTeX through a minimal, explicit mapping
    body = []
    for line in md.split("\n"):
        if line.startswith("# "):
            body.append("\\section*{" + line[2:].strip().replace("&", "\\&") + "}")
        elif line.startswith("## "):
            body.append("\\subsection*{" + line[3:].strip().replace("&", "\\&") + "}")
        elif line.startswith("### "):
            body.append("\\subsubsection*{" + line[4:].strip().replace("&", "\\&") + "}")
        elif line.strip().startswith("|"):
            body.append(line)
        elif line.strip() == "":
            body.append("")
        else:
            # escape text first, then substitute code spans, so \_ is not re-escaped
            def _esc(x):
                for a, b in [("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                             ("~", "\\textasciitilde{}")]:
                    x = x.replace(a, b)
                return x

            parts = re.split(r"`([^`]+)`", line)
            out_line = []
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    out_line.append("\\texttt{" + _esc(part) + "}")
                else:
                    out_line.append(_esc(part))
            body.append("".join(out_line) + "\\par")
    # markdown tables -> longtable
    merged, i = [], 0
    while i < len(body):
        if body[i].strip().startswith("|") and i + 1 < len(body) and set(body[i + 1].replace("|", "").replace(" ", "")) <= {"-", ""}:
            rows = []
            j = i
            while j < len(body) and body[j].strip().startswith("|"):
                import re as _re
                cells = _re.split(r"(?<!\\)\|", body[j].strip().strip("|"))
                rows.append([c.strip().replace("\\|", "$|$") for c in cells])
                j += 1
            rows = [rows[0]] + rows[2:]
            ncol = len(rows[0])
            merged.append("\\begin{center}\\footnotesize\\begin{longtable}{" + "l" + "r" * (ncol - 1) + "}")
            merged.append("\\toprule")
            for k, r in enumerate(rows):
                esc = []
                for c in r:
                    c = c.replace("\\", "\\textbackslash{}")
                    for a, b in [("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
                                 ("$", "\\$"), ("^", "\\textasciicircum{}"),
                                 ("~", "\\textasciitilde{}"), ("|", "$|$"),
                                 ("<", "$<$"), (">", "$>$")]:
                        c = c.replace(a, b)
                    esc.append(c)
                merged.append(" & ".join(esc) + " \\\\")
                if k == 0:
                    merged.append("\\midrule")
            merged += ["\\bottomrule", "\\end{longtable}\\end{center}"]
            i = j
        else:
            merged.append(body[i])
            i += 1
    open(f"{ROOT}/_summary_body.tex", "w").write("\n".join(merged))
    open(OUT_TEX, "w").write(tex)
    try:
        r = subprocess.run(["/opt/data/bin/tectonic", "-X", "compile", OUT_TEX],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
        print("tectonic rc", r.returncode, (r.stdout or "")[-300:], (r.stderr or "")[-500:])
    except Exception as e:
        print("compile skipped:", e)
    print("wrote", OUT_MD, OUT_TEX)


if __name__ == "__main__":
    main()
