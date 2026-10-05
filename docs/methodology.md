# Methodology

Frozen protocol: `experiments/configs/base.yaml` (written before any test-window
result was computed). Generated tables: `results/tables/`. This document explains
the reasoning; the config is the machine-readable source of truth.

## Research question

Given one dataset, one investable universe, one walk-forward split, one cost
model and one set of portfolio constraints, how does a financial foundation
model (Kronos) compare with an RL portfolio agent (AlphaZeroBeta) in forecasting
accuracy (where defined), risk-adjusted portfolio performance, market
neutrality, robustness and computational cost?

## Why the comparison is asymmetric by design

Kronos emits a forecast. AlphaZeroBeta emits portfolio weights. There is no
shared task, so there is no single number that ranks them.

We therefore evaluate on three levels and keep them separate:

| Level | What is compared | Who participates |
|---|---|---|
| 1 forecasting | MAE, RMSE, directional accuracy, rank IC | Kronos vs random walk only |
| 2 portfolio (primary) | Sharpe, Sortino, Calmar, drawdown, ES95, beta, corr, turnover | all arms |
| 3 computational | params, latency, throughput, train time | both, in their own units |

AlphaZeroBeta is **excluded** from Level 1. It produces no point forecast, and
manufacturing one from its value head would be an invention, not a measurement.

## Data

- Market: S&P 500 (`^GSPC` benchmark), daily bars.
- Source: public market-data endpoint, retrieved 2026-10-05. Not redistributed.
- Adjustment: `Open/High/Low/Close * (AdjClose/Close)`. Same-day factor only.
- Panel: 476 tickers x 2924 trading days, 2013-06-03 .. 2025-01-14.
- Investable set: top 100 by trailing 60-day mean dollar volume, recomputed
  per date, applied identically to every arm.
- Returns: adjusted close-to-close simple returns; `|r| >= 75%` dropped as bad
  prints (`code/data/build_panel.py`).

## Walk-forward (matches AlphaZeroBeta Section 3)

- train 756 trading days (36 months)
- validation 126 days (6 months)
- test 126 days (6 months), non-overlapping
- step 126 days
- 22 folds, test windows 2014-01 .. 2024-12
- 2004-2010 equivalent warm-up: our panel starts 2013-06 to give 252 days of
  history before the first test window and to initialise rolling estimators.

## Execution model

`weights[t]` is formed from information available through `t-1` and earns the
return of `t`. Implemented in `code/backtest/harness.py::simulate`, which also
applies costs at the same step. This ordering is asserted in
`tests/test_core.py::test_momentum_is_past_only`.

## Portfolio constraints

Market-neutral arms (Kronos, AlphaZeroBeta, momentum, ridge):

```
sum_i w_i = 0        dollar neutral
sum_i |w_i| <= 1     gross exposure cap (leverage)
-1 <= w_i <= 1       per-name box
```

Projection: centre, then scale onto the L1 ball of radius 1, then clip
(`code/backtest/harness.py::dollar_neutral`). AlphaZeroBeta's own projection is
identical in form; both arms pass through the same operator so the constraint
geometry cannot favour either.

Net-long reference arms (index, equal weight, max-Sharpe, min-correlation):
`sum_i w_i = 1`, `w_i >= 0`, reported separately and never ranked against the
neutral arms.

Top/bottom 20 names per side, equal notional, fixed a priori.

## Transaction costs

From the source study's schedule:

| Bucket | Per side |
|---|---|
| top liquidity decile | 5 bps |
| remaining names | 15 bps |
| borrow (shorts) | 30 bps / year, accrued per holding day |

Sensitivity: 0x, 1x, 2x. Turnover is `sum_i |w_{i,t} - w_{i,t-1}|`.

## Metric definitions

- CAGR from the compounded net return series, annualised on 252 days.
- Sharpe: annualised mean / annualised standard deviation, `ddof=1`.
- Sortino: annualised mean / annualised downside deviation.
- Calmar: CAGR / |max drawdown|.
- Max drawdown from the compounded equity curve.
- ES95: mean return in the worst 5% of days.
- Beta, correlation: OLS on the index return over the same days.
- Turnover: annualised sum of absolute weight changes.
- Gross/net/long/short exposure: per-day, from the weight matrix.

## Statistical tests, and why these

| Test | Applied to | Reason |
|---|---|---|
| Stationary block bootstrap (21-day blocks, 5000 resamples) | Sharpe **difference** | returns are autocorrelated and heteroskedastic; resampling i.i.d. understates the variance of a Sharpe difference |
| Diebold-Mariano (HAC long-run variance, HLN small-sample correction) | Kronos vs random-walk squared errors | the two error series are paired and dependent |
| Newey-West (lag 5) HAC regression on the index | every arm | tests whether alpha survives autocorrelation/heteroskedasticity |
| Cross-sectional rank IC (Spearman, averaged over days, t-stat over days) | Kronos | measures the actual decision-relevant signal, not level error |

Deliberately **not** run: any forecasting test on AlphaZeroBeta (no forecast),
any test whose assumptions the design violates. Each reported interval either
contains zero or it does not; no test is described as significant without one.

## Regimes (defined before results)

- Realised volatility terciles from the 60-day trailing standard deviation of
  index returns: low, mid, high.
- Drawdown state: bull if the running drawdown is shallower than -10%, bear
  otherwise.

Both partitions come from the benchmark alone, so no strategy's own path
influences the split.

## Reproducing

```
python experiments/run_all.py            # cached artefacts
python experiments/run_all.py --remote   # re-run the GPU arms first
make -C paper paper                      # rebuild the PDF
```
