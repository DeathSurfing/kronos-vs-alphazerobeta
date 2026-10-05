# Frozen evaluation protocol (S&P 500)

Frozen at 2026-10-05T08:05Z, BEFORE any test-period result was inspected.
Every deviation from the source papers is listed in docs/reproduction.md.

## Scope (user-directed)
S&P 500 only (`^GSPC`). AlphaZeroBeta's published Indian-market results do not
exist, so India is out of scope. Cross-market analysis from the original brief is
dropped and stated as a limitation.

## Data
Source: Yahoo Finance via `yfinance` (adjusted OHLCV), retrieved 2026-10-05.
Window 2013-06-01..2025-01-14 (needed for 252d warm-up before the 2014 test start).
Universe pages: Wikipedia current index membership (survivorship caveat, see below).

## Walk-forward (matches AZB §3 exactly)
- train  : 36 months (756 trading days)
- validate: 6 months (126 days)
- test   : 6 months (126 days), non-overlapping
- step   : 6 months
- folds  : 22, test windows 2014-01..2024-12
- execution: weights formed from data <= t, earn return of t+1.

## Investable universe
Top 100 names by 60d median dollar volume among eligible S&P 500 members
(>=252d history). **Applied identically to every model.** Deviation from AZB
(which uses the full index) -> disclosed; it equalises compute and makes the
comparison apples-to-apples.

## Portfolio constraints (AZB §3.5)
- dollar neutral: sum(w) = 0
- gross <= 1: sum|w| <= 1
- box: -1 <= w_i <= 1
- shorting allowed; leverage bounded by the L1 ball

## Transaction costs (AZB Table D4)
- 5 bps/side top decile, 15 bps/side other US names
- borrow 30 bps/yr accrued over holding days
Sensitivity: 0x / 1x / 2x.

## Seeds
3 seeds (42, 123, 456) for every stochastic model. Mean AND std AND best AND
worst reported. AZB published 9 -> disclosed shortfall.

## Statistical tests
- Paired bootstrap (10k resamples, block=21d) on Sharpe difference
- Diebold-Mariano on squared forecast errors (Kronos vs random walk)
- Newey-West (lag=5) HAC t-stat on alpha vs ^GSPC
- Sharpe difference CI via stationary bootstrap

## Regimes (defined now)
Benchmark 60d realized vol terciles (low/mid/high) and ^GSPC drawdown state
(bull: dd > -10%; bear: dd <= -10%). Fixed before results.
