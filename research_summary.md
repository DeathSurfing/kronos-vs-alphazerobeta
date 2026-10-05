# Research Summary

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

| item | value |
|---|---|
| market | S&P 500 (^GSPC) |
| data source | Yahoo Finance daily bars via yfinance, retrieved 2026-10-05 |
| price adjustment | OHLC scaled by AdjClose/Close (splits+dividends), same-day factor |
| panel | 476 tickers x 2924 trading days |
| sample | 2014-01-02 .. 2024-12-31 |
| investable set | top 100 by 60d mean dollar volume, identical for all models |
| train window | 756 trading days (~36 months) |
| validation window | 126 trading days (~6 months) |
| test window | 126 trading days (~6 months) |
| step | 126 trading days (6 months) |
| out-of-sample folds | 22 non-overlapping, 2014-01-02 .. 2025-01-07 |
| rebalance | daily |
| execution | weights formed from information <= t-1, applied to the return of t |
| transaction costs | 5 bps/side top liquidity decile, 15 bps/side otherwise (AZB Table D4) |
| borrow cost | 30 bps/yr on short notional, accrued per holding day |
| sensitivity | 0x / 1x / 2x the baseline schedule |
| constraints (neutral arms) | sum w = 0, sum\|w\| <= 1, w_i in [-1,1], 20 names per side |
| constraints (long arms) | sum w = 1, w_i >= 0 |
| seeds | AZB: 1 seed all 22 folds; 3 seeds on folds {0,7,14} for seed sensitivity |
| hardware | NVIDIA RTX 4080 SUPER 16 GB, Windows 11; CPU pipeline on 4-core container |

## 5. Model configurations

| model | type | params | context | inputs | objective | decision_use | training_here |
|---|---|---|---|---|---|---|---|
| Kronos-small | financial foundation model (decoder-only transformer) | 24741376 | 512 | OHLCV + amount | next-bar K-line generation (self-supervised pre-training) | forecast -> cross-sectional ranking -> weights | zero-shot (no fine-tuning) |
| AlphaZeroBeta | CNN-GRU actor-critic (recurrent PPO) | 1703587 | 100 | 8 past-only price/volume features x 3 resolutions | reward Eq.8: risk-adjusted excess return - corr penalty - turnover | policy emits dollar-neutral weights directly | trained per walk-forward fold |

Kronos is used zero-shot with the released `Kronos-small` checkpoint
(24,741,376 parameters, 512-token context, 400-bar input window, single-sample
decoding). AlphaZeroBeta is re-implemented and retrained per fold; deviations
from the published configuration are listed in `docs/reproduction.md`.

## 6. Published results

Values as reported by the original authors. These are **not** our experiments.

| source | market | metric | value | uncertainty |
|---|---|---|---|---|
| AlphaZeroBeta (Belyakov 2026) | ^GSPC | Sharpe | 1.61 | 0.48 |
| AlphaZeroBeta (Belyakov 2026) | ^NDX | Sharpe | 1.48 | 0.41 |
| AlphaZeroBeta (Belyakov 2026) | ^DJI | Sharpe | 1.2 | 0.28 |
| AlphaZeroBeta (Belyakov 2026) | ^FTSE | Sharpe | 0.94 | 0.19 |
| AlphaZeroBeta (Belyakov 2026) | ^GDAXI | Sharpe | 0.86 | 0.23 |
| AlphaZeroBeta (Belyakov 2026) | ^HSI | Sharpe | 1.04 | 0.33 |
| AlphaZeroBeta (Belyakov 2026) | 000001.SS | Sharpe | 1.63 | 0.38 |
| AlphaZeroBeta (Belyakov 2026) | ^GSPC | max drawdown | -0.26 | 0.15 |
| AlphaZeroBeta (Belyakov 2026) | ^GSPC | corr to index | 0.15 | 0.09 |
| AlphaZeroBeta (Belyakov 2026) | all 7 | average Sharpe | 1.251 | 0.30 |
| Kronos (Shi et al. 2025) | various | zero-shot forecasting | nan | - |

## 7. Independent reproduction

Reproduced against published, S&P 500, baseline costs. The reproduction is
deliberately partial (reduced RL training budget, 1 seed on all folds, 100-name
investable subset, public daily data); the gap is reported as a finding.

| metric | published | reproduced | abs_diff | pct_diff |
|---|---|---|---|---|
| sharpe | 1.61 | -0.227 | -1.837 | -114.1 |
| max_dd | -0.26 | -0.001 | 0.259 | 99.6 |
| corr | 0.15 | 0.0 | -0.15 | -99.9 |
| index buy-and-hold sharpe | 0.72 | 0.725 | 0.005 | 0.7 |

## 8. Benchmark results

### 8.1 Forecasting (Level 1)

AlphaZeroBeta is excluded by construction: it emits portfolio weights, not
point forecasts, so forecasting metrics are undefined for it.

| mae | rmse | dir_acc | n | rw_mae | rw_rmse | mae_skill | model | rank_ic_mean | rank_ic_std | rank_ic_t |
|---|---|---|---|---|---|---|---|---|---|---|
| 4.7306 | 13.1554 | 0.4983 | 251003 | 1.6306 | 3.3577 | -1.9011 | Kronos-small (zero-shot) | 0.0011 | 0.175 | 0.3204 |
| 1.6306 | 3.3577 | 0.9959 | 251003 | nan | nan | 0.0 | Random walk (pred = prev close) | nan | nan | nan |

Diebold-Mariano on squared errors (Kronos vs random walk):
DM = 30.190, p = 0.000, n = 2515.

### 8.2 Portfolio (Level 2, primary)

| model | cagr | ann_vol | sharpe | sortino | calmar | max_dd | var95 | es95 | beta | corr | skew | n_days | turnover_ann | n_rebalances | avg_gross_exposure | avg_net_exposure | avg_long | avg_short | cost_drag_ann | cost_level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| momentum | -0.028 | 0.126 | -0.16 | -0.197 | -0.077 | -0.363 | -0.012 | -0.02 | -0.06 | -0.083 | -1.112 | 2524 | nan | nan | nan | nan | nan | nan | nan | nan |
| ridge | -0.155 | 0.11 | -1.478 | -2.104 | -0.19 | -0.819 | -0.01 | -0.017 | 0.004 | 0.006 | 0.552 | 2461 | nan | nan | nan | nan | nan | nan | nan | nan |
| equal | 0.131 | 0.185 | 0.756 | 0.913 | 0.371 | -0.352 | -0.017 | -0.028 | -0.127 | -0.117 | -0.474 | 2772 | nan | nan | nan | nan | nan | nan | nan | nan |
| maxsharpe | 0.194 | 0.238 | 0.864 | 1.122 | 0.598 | -0.325 | -0.022 | -0.035 | -0.129 | -0.095 | -0.008 | 2523 | nan | nan | nan | nan | nan | nan | nan | nan |
| mincorr | 0.184 | 0.217 | 0.887 | 1.151 | 0.484 | -0.38 | -0.02 | -0.031 | -0.109 | -0.088 | -0.218 | 2523 | nan | nan | nan | nan | nan | nan | nan | nan |
| kronos | -0.236 | 0.088 | -3.01 | -4.874 | -0.256 | -0.921 | -0.009 | -0.012 | 0.01 | 0.021 | 1.407 | 2367 | nan | nan | nan | nan | nan | nan | nan | nan |
| index_ref | 0.115 | 0.171 | 0.725 | 0.859 | 0.34 | -0.339 | -0.016 | -0.026 | 1.0 | 1.0 | -0.582 | 2772 | nan | nan | nan | nan | nan | nan | nan | nan |
| azb_s42 | -0.0 | 0.0 | -0.227 | -0.117 | -0.057 | -0.001 | -0.0 | -0.0 | 0.0 | 0.0 | 2.402 | 2625 | nan | nan | nan | nan | nan | nan | nan | nan |

### 8.3 Statistical tests

| model_a | model_b | sharpe_a | sharpe_b | diff | ci_lo | ci_hi | p_value | alpha_ann | alpha_t | alpha_p |
|---|---|---|---|---|---|---|---|---|---|---|
| kronos | azb_s42 | -3.01 | -0.342 | -2.668 | -3.614 | -1.76 | 0.0 | -0.271 | -9.583 | 0.0 |
| kronos | index_ref | -3.01 | 0.728 | -3.738 | -4.888 | -2.89 | 0.0 | -0.271 | -9.583 | 0.0 |
| azb_s42 | index_ref | -0.227 | 0.734 | -0.961 | -1.807 | -0.177 | 0.018 | -0.0 | -0.653 | 0.514 |
| kronos | momentum | -3.01 | -0.163 | -2.847 | -4.041 | -1.731 | 0.0 | -0.271 | -9.583 | 0.0 |
| azb_s42 | momentum | -0.457 | -0.156 | -0.301 | -0.923 | 0.542 | 0.245 | -0.0 | -0.653 | 0.514 |
| kronos | ridge | -3.01 | -1.413 | -1.597 | -2.495 | -0.681 | 0.0 | -0.271 | -9.583 | 0.0 |
| azb_s42 | ridge | -0.335 | -1.479 | 1.144 | 0.405 | 1.89 | 0.003 | -0.0 | -0.653 | 0.514 |
| kronos | equal | -3.01 | 0.776 | -3.785 | -4.862 | -2.986 | 0.0 | -0.271 | -9.583 | 0.0 |

### 8.4 Regime analysis

| regime | model | cost_level | n_days | ann_return | sharpe |
|---|---|---|---|---|---|
| low vol | momentum | 0x | 800 | 0.087 | 0.948 |
| low vol | ridge | 0x | 800 | -0.03 | -0.382 |
| low vol | equal | 0x | 912 | 0.039 | 0.337 |
| low vol | maxsharpe | 0x | 800 | 0.179 | 1.05 |
| low vol | mincorr | 0x | 800 | 0.05 | 0.379 |
| low vol | kronos | 0x | 745 | -0.076 | -1.14 |
| low vol | index_ref | 0x | 912 | 0.114 | 1.244 |
| low vol | azb_s42 | 0x | 847 | 0.0 | 0.309 |
| low vol | momentum | 1x | 800 | 0.052 | 0.569 |
| low vol | ridge | 1x | 800 | -0.213 | -2.679 |
| low vol | equal | 1x | 912 | 0.031 | 0.271 |
| low vol | maxsharpe | 1x | 800 | 0.167 | 0.976 |
| low vol | mincorr | 1x | 800 | 0.04 | 0.302 |
| low vol | kronos | 1x | 745 | -0.355 | -5.337 |
| low vol | index_ref | 1x | 912 | 0.114 | 1.244 |
| low vol | azb_s42 | 1x | 847 | -0.0 | -0.082 |
| low vol | momentum | 2x | 800 | 0.017 | 0.19 |
| low vol | ridge | 2x | 800 | -0.395 | -4.954 |
| low vol | equal | 2x | 912 | 0.024 | 0.206 |
| low vol | maxsharpe | 2x | 800 | 0.154 | 0.902 |
| low vol | mincorr | 2x | 800 | 0.03 | 0.225 |
| low vol | kronos | 2x | 745 | -0.635 | -9.515 |
| low vol | index_ref | 2x | 912 | 0.114 | 1.244 |
| low vol | azb_s42 | 2x | 847 | -0.0 | -0.493 |
| mid vol | momentum | 0x | 812 | -0.044 | -0.439 |
| mid vol | ridge | 0x | 749 | 0.038 | 0.417 |
| mid vol | equal | 0x | 907 | 0.139 | 0.94 |
| mid vol | maxsharpe | 0x | 811 | 0.089 | 0.462 |
| mid vol | mincorr | 0x | 811 | 0.109 | 0.639 |
| mid vol | kronos | 0x | 710 | 0.057 | 0.834 |
| mid vol | index_ref | 0x | 907 | 0.11 | 0.78 |
| mid vol | azb_s42 | 0x | 866 | 0.0 | 0.082 |
| mid vol | momentum | 1x | 812 | -0.08 | -0.801 |
| mid vol | ridge | 1x | 749 | -0.146 | -1.592 |
| mid vol | equal | 1x | 907 | 0.132 | 0.894 |
| mid vol | maxsharpe | 1x | 811 | 0.078 | 0.402 |
| mid vol | mincorr | 1x | 811 | 0.099 | 0.582 |
| mid vol | kronos | 1x | 710 | -0.235 | -3.456 |
| mid vol | index_ref | 1x | 907 | 0.11 | 0.78 |
| mid vol | azb_s42 | 1x | 866 | -0.0 | -0.408 |

### 8.5 Transaction-cost sensitivity

| model | cagr_0x | cagr_1x | cagr_2x | sharpe_0x | sharpe_1x | sharpe_2x |
|---|---|---|---|---|---|---|
| azb_s42 | 0.0 | -0.0 | -0.0 | 0.134 | -0.227 | -0.548 |
| equal | 0.138 | 0.131 | 0.123 | 0.793 | 0.756 | 0.719 |
| index_ref | 0.115 | 0.115 | 0.115 | 0.725 | 0.725 | 0.725 |
| kronos | 0.02 | -0.236 | -0.427 | 0.27 | -3.01 | -6.276 |
| maxsharpe | 0.209 | 0.194 | 0.179 | 0.916 | 0.864 | 0.812 |
| mincorr | 0.196 | 0.184 | 0.172 | 0.934 | 0.887 | 0.84 |
| momentum | 0.006 | -0.028 | -0.061 | 0.115 | -0.16 | -0.435 |
| ridge | 0.017 | -0.155 | -0.298 | 0.207 | -1.478 | -3.157 |

### 8.6 Seed sensitivity

| seed | n_folds | sharpe | n_sharpe_folds | cagr | max_dd | ann_vol |
|---|---|---|---|---|---|---|
| 42.0 | 3.0 | -4.081 | 1.0 | -0.059 | -0.03 | 0.016 |
| 123.0 | 3.0 | -2.119 | 2.0 | -0.049 | -0.029 | 0.014 |
| 456.0 | 3.0 | -1.655 | 2.0 | 0.002 | -0.034 | 0.029 |

### 8.7 Computational comparison

| model | params | device | mean_latency_ms | median_latency_ms | p95_latency_ms | throughput_names_per_s | training_time_s | notes |
|---|---|---|---|---|---|---|---|---|
| Kronos-small | 24741376 | RTX 4080 SUPER | nan | nan | nan | nan | 0.0 | zero-shot; no training |
| AlphaZeroBeta | 1703587 | RTX 4080 SUPER | nan | nan | nan | nan | 54.2 | mean per fold over 31 fold runs |

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

Experimental runs logged: 31.

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
