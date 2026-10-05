"""One-command reproduction of the full study.

    python experiments/run_all.py             # everything, using cached artefacts
    python experiments/run_all.py --remote    # also (re)run the GPU arms first
    python experiments/run_all.py --skip-remote

Stages
  1 data      : raw panel -> processed panel
  2 signals   : model outputs -> target weights
  3 backtest  : weights -> net returns per arm and per cost level
  4 stats     : tables (incl. bootstrap/DM/Newey-West) and figures
  5 paper     : LaTeX fragments -> PDF (needs `tectonic` or pdflatex)

The GPU arms (Kronos inference, AlphaZeroBeta training) live on the remote host
and are driven by experiments/kronos_infer.py and code/alphazerobeta/azb_train.py;
their outputs are pulled into results/raw before stage 2.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = "/opt/data/kvab"
PY = sys.executable


def run(desc, cmd, cwd=ROOT):
    print(f"\n=== {desc} ===\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd)
    print(f"--- {desc}: exit={r.returncode} ({time.time()-t0:.1f}s)", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true", help="run the GPU arms on the remote host")
    ap.add_argument("--skip-remote", action="store_true")
    ap.add_argument("--skip-paper", action="store_true")
    a = ap.parse_args()

    stages = [
        ("data", [PY, "code/data/build_panel.py"]),
    ]
    if a.remote:
        stages = [("kronos inference (remote)", [PY, "experiments/run_kronos.py"]),
                  ("alphazerobeta training (remote)", [PY, "experiments/run_alphazerobeta.py"])] + stages
    stages += [
        ("baselines + signals", [PY, "experiments/run_baselines.py"]),
        ("backtest", [PY, "code/backtest/run_backtest.py"]),
        ("tables", [PY, "code/statistics/generate_tables.py"]),
        ("seed sensitivity", [PY, "code/statistics/seed_sensitivity.py"]),
        ("latex tables", [PY, "code/statistics/to_latex.py"]),
        ("figures", [PY, "code/statistics/generate_figures.py"]),
    ]
    if not a.skip_paper:
        stages.append(("paper", ["make", "-C", "paper", "paper"]))
    bad = [d for d, c in stages if run(d, c) != 0]
    print("\nDONE" if not bad else f"\nFAILED stages: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
