# Reproduction notes

Two systems are reproduced here: Kronos is used exactly as released
(zero-shot), and AlphaZeroBeta is re-implemented and retrained. This document
records, for each, what was followed, what was changed, and why. Anything not
listed as a deviation follows the source description.

Sources of record
- AlphaZeroBeta: Belyakov, B., *AlphaZeroBeta: Deep Reinforcement Learning for
  Market-Neutral Portfolios*, arXiv:2607.18001, Financial Innovation,
  DOI 10.1186/s40854-026-00955-4.
- Kronos: Shi, Y. et al., *Kronos: A Foundation Model for the Language of
  Financial Markets*, arXiv:2508.02739 (AAAI 2026).

---

## Kronos (zero-shot)

| Item | Value |
|---|---|
| Checkpoint | `NeoQuasar/Kronos-small` (official release) |
| Tokenizer | `NeoQuasar/Kronos-Tokenizer-base` |
| Parameters | 24,741,376 (measured) |
| Model context | 512 tokens |
| Context used | 400 daily bars |
| Forecast horizon | 1 bar |
| Input variables | open, high, low, close, volume, amount |
| Sampling | temperature 1.0, top-p 0.9, 1 sample (model defaults) |
| Mode | zero-shot; no fine-tuning on any S&P 500 data |
| Fine-tuning | not attempted (out of scope for this budget) |

Pipeline: for each test date `t` and each universe name, build a window of
daily bars ending at `t-1`, predict the next bar, take
`pred_close/prev_close - 1` as the expected-return signal. Because the window
ends strictly before `t`, no day at or after `t` influences the forecast.

`KronosPredictor` is loaded from the released repository code
(`third_party/kronos/model/`) without modification. The official weight files
are fetched from the public model hub on the GPU host; they are not committed
to this repository.

Sources of the released artefacts, for verification:
- code: announced repository of the paper (model/, examples/)
- weights: official hub IDs above

---

## AlphaZeroBeta (re-implementation)

Followed as specified:

| Component | Source spec | Reproduced |
|---|---|---|
| CNN kernels / strides / filters | 8/4/3, 4/2/1, 32/64/64 | same |
| GRU hidden | 512 | same |
| Shared FC + head width | 512 | same |
| Policy output | tanh, bounded | same |
| Agent observation window | 100 steps | same |
| Multi-resolution input | daily/weekly/monthly stacked on channels | same |
| Reward | Eq. 8: `(rp-rm)/sigma_p - l1*corr - l2*sum|dw|` | same |
| lambda1 | 0.5 | same |
| lambda2 | 0.001 | same |
| sigma_p floor | 1e-8 | same |
| Rolling window | 60 business days | same |
| gamma | 0.99 | same |
| learning rate | 3e-4 Adam | same |
| PPO clip | 0.20 | same |
| GAE lambda | 0.95 | same |
| entropy coefficient | 0.01 | same |
| value coefficient | 0.5 | same |
| gradient clip | 0.5 | same |
| Optimiser epochs per update | 10 | **3** (deviation D5) |
| Trajectories per update | 256 | **1** (deviation D2/D5) |
| Dollar-neutral projection | centre + L1 ball radius 1 | same |
| Gross exposure cap | 1 | same |
| Per-name box | [-1, 1] | same |
| Walk-forward | 756 / 126 / 126, step 126, 22 folds | same |
| Costs | 5/15 bps per side, 30 bps/yr borrow | same |
| Seeds | 9 | **1 on all folds, 3 on a fold subset** (D4) |

### Deviations, with reasons

**D1 — encoder treats assets independently.**
The source architecture flattens the observation over all assets and feeds a
single GRU. That makes the parameter count and the per-step cost scale with the
number of assets, and is infeasible for a 100-name universe on a single
consumer GPU. Our implementation applies the same CNN-GRU trunk per asset
(asset dimension folded into the batch) and lets the shared trunk produce each
name's weight; the policy is then projected jointly. Practical consequence: the
agent cannot learn cross-asset interactions inside the trunk, only through the
shared parameters and the joint projection. This is the single largest
architectural deviation.

**D2/D5 — reduced RL budget.**
The source study reports 57–106 GPU-hours per index for a single-seed sweep on
a V100 and roughly 12 days wall-clock for the full seven-index nine-seed sweep
on eight A100s. Our budget on one RTX 4080 SUPER is a small fraction of that.
We therefore reduce the PPO epochs per update (10 -> 3) and the rollout length
(24 -> 16), and cap training steps per fold. The reproduced Sharpe should be
read as a lower bound attributable to this budget, not as the source
configuration's ceiling.

**D3 — training steps per fold.** Set in `configs/base.yaml`
(`alphazerobeta.steps`). Chosen to fit all 22 folds inside the session budget.

**D4 — seeds.** Nine in the source; one on all folds plus three on a fixed fold
subset here. All seeds are reported, including the worst, in `T11_seeds.csv`.

**D6 — investable universe.** The source uses the full index panel. We use the
top 100 names by trailing 60-day dollar volume, identical for every arm, to
equalise compute and remove a confound between arms.

**D7 — feature set.** Eight price- and volume-derived features (returns at
several horizons, reversal, realised volatility, skew, volume z-score, RSI),
cross-sectionally standardised per day. The source panel also includes
fundamentals, analyst earnings surprises, insider activity and news sentiment
from a commercial terminal, none of which are available here. This is a
material information disadvantage for our reproduction.

**D8 — data.** Public daily bars instead of a commercial terminal panel, and
current index membership instead of point-in-time membership (survivorship
bias; see `docs/methodology.md` and the threats-to-validity section).

**D9 — validation slice.** The folds are constructed with a six-month
validation window, matching the source, but early stopping is not used; the
policy is trained for a fixed number of steps on the training window.

---

## What was NOT reproduced

- The other six indices of the source study (Nasdaq-100, DJIA, FTSE 100, DAX,
  Hang Seng, SSE Composite). The full reproduction is restricted to the S&P 500
  by explicit scope decision.
- The source study's own table of published results is reported verbatim in
  `T3_published.csv` and never mixed with reproduced numbers.
- Kronos fine-tuning.

## Environment

GPU host: NVIDIA RTX 4080 SUPER, 16 GB, driver 616.64, Windows 11 (Build 26100).
Python 3.13.5, PyTorch 2.6.0+cu124, pandas 3.0.6, numpy, pyarrow 25, einops
0.8.2, huggingface_hub 2.1.1.

CPU pipeline: 4 vCPU container, 4 GB cgroup memory limit, Python 3.13.5 in a
`uv` virtual environment with numpy, pandas, scipy, matplotlib, pyarrow,
statsmodels, yfinance, paramiko.

Data and weights are downloaded from their official public sources on first
run; neither is redistributed in this repository.
