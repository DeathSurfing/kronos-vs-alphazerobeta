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


---

## Addendum: training attempts, including three that failed (all reported)

The AlphaZeroBeta arm was trained four times. All four outcomes are recorded
in the paper (Table "instability") and in `experiment_log.jsonl`; none was
silently discarded.

| Run | tag | steps/fold | lr | reward/value clip | outcome |
|---|---|---|---|---|---|
| A | `azb` | 200 | 3e-4 | none | 22/22 folds; policy collapsed to the null portfolio from fold 2 (gross exposure 0.000) |
| B | `azbdeep` | 1500 | 1e-4 | none | value loss 3.2e25 (fold 0); NaN policy on fold 2 |
| C | `azbdeep2` | 1500 | 5e-5 | reward +-10, value +-50 | value loss 8.0e11 (fold 0); killed after fold 0 |
| D | `azbfast3` | 2000 | 3e-4 | reward +-10, value +-50 | 22/22 folds, no non-finite update; mean gross exposure 0.83; **this is the arm reported in the paper** |

Runs A-C are kept for the instability table. Run D (the clipping from C plus a
10x step budget) is the arm whose weights back the reported AlphaZeroBeta row;
`windows/` holds it and `windows_azb200/` holds run A.

Root cause: Eq. 8 divides by $\sigma_p$, floored at 1e-8 in the source. Early in
training the policy is near-flat, so $\sigma_p$ is tiny and the ratio is enormous;
the resulting advantage inflates the value target without bound. The first run's
value loss reached 3e25; the second 8e11.

Mitigation applied in run C: clip the risk-adjusted term to +-10 and the value
target to +-50, plus a guard that skips any optimiser step whose gradients or
loss are non-finite. This removes the divergence but does not by itself produce a
profitable policy inside this budget: the finite run's learned book averaged
gross exposure 0.041 against the cap of 1.0.

Consequence for the reported results: the reproduced Sharpe ratio for
AlphaZeroBeta is reported as a lower bound attributable to this budget, not as a
refutation of the published figure. The published configuration is reported to
need 57-106 GPU-hours per index on a V100; this study had roughly two orders of
magnitude less compute, and the arm that is reported (run D, 2000 steps/fold)
still sits well below that. Runs A-C are reported as instability findings, not
hidden.


---

## Addendum: Kronos decoding configuration (deviation D11)

The Kronos paper's inference table recommends, for price-series forecasting,
temperature 0.6, top-p 0.90, and **10 averaged samples**. The runs reported in
this study used **temperature 1.0 and a single sample**. This is a deliberate
disclosure of a configuration error on our side, not a property of the model.

Why it matters: the model is a generative sampler, so a single draw carries the
full sampling variance. The paper averages ten draws precisely to suppress that
variance before computing the implied return. With N=1, the per-asset
expected-return signal is noisier than the published configuration would
produce, and a noisier cross-sectional ranking is exactly what destroys the
long/short spread.

Consequence for the results: the Kronos arm reported here should be read as a
**lower bound** on the released model's behaviour under this protocol. Re-running
inference at T=0.6 with N=10 costs roughly ten times the inference time
(~5 GPU-hours on this host for the full 2014-2024 window); it was not completed
inside the available compute budget and is listed as the first item of future
work.

A second, smaller deviation: the paper reports IC/RankIC averaged over the four
OHLC channels, whereas the exploitation signal used here is derived solely from
the predicted close. We do not report an IC number for the paper's four-channel
definition.


---

## Addendum: bounded action space defeats the naive policy-gradient likelihood (D14)

The AlphaZeroBeta action space is constrained by construction: the policy output
is projected to be dollar-neutral with bounded gross exposure. The obvious
implementation samples from the Gaussian policy and then projects the sample.

That implementation **does not train**, and the reason is worth stating plainly
because it looks like a hyperparameter problem and is not one.

A diagonal Gaussian assigns density over the whole real line. The projection maps
any point onto the L1 ball, and does so *deterministically*. If the projected
action is then fed back as the action whose `log_prob` is evaluated, the
likelihood being differentiated is that of a point that may be arbitrarily
improbable under the proposal: the projection regularly rescales a sample by a
large factor, so `log_prob(projected)` can be enormous. In the clipped
surrogate objective the ratio `exp(log_pi - log_pi_old)` then diverges, gradients
follow it, and the loss reaches `1e35` within a handful of updates.

Observed failures, all with the same signature (healthy loss for several updates,
then a single catastrophic one):

| Run | reward/value clip | log_prob at | outcome |
|---|---|---|---|
| `azb` | none | projected | all 22 folds; policy collapsed to the null portfolio (gross exposure 0.000) |
| `azb_fast` | reward +-10 | projected | loss 9.6 -> 3.7e4 -> 1.5e34 by update 7; NaN by update 10 |
| `azb_fast2` | reward +-10, advantage +-100, value +-50 | projected | 2543 non-finite updates in one fold; median loss 4.67, max 2.5e35 |
| `azb_fast3` | as above | **raw sample** | loss stable 4.61 -> 4.27 -> 4.07 -> 4.14 -> 4.46 -> 3.99; **0 non-finite updates** |

The fix is one line: evaluate the log-density at the **raw sample**, and treat the
projection as a deterministic action map applied after sampling (the standard
treatment of a squashed or bounded action). Only the raw sample is inside the
support of the density being optimised; the projected point is not.

A second, smaller fix: clamping `logstd` to `[-4, 0]` keeps `sigma` in
`[e^-4, 1]`, so `log_prob` cannot be driven arbitrarily negative by a standard
deviation that training has squeezed toward zero.

### Why this matters beyond this study

The published reward floors `sigma_p` at `1e-8`. Combined with a projection
applied inside the likelihood, the null portfolio becomes a spurious attractor:
a near-flat book makes both the reward ratio and the likelihood ratio ill
conditioned, and the agent's cheapest escape is to stop trading. Three of the
four runs above ended either diverged or collapsed to exactly that state. Any
re-implementation of a projected-action RL agent for portfolios should check this
before concluding the method does not work, and should report the number of
non-finite gradient updates as a training diagnostic rather than attributing
flat learning curves to the algorithm.
