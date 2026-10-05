# Review cycles

Three review passes were carried out on the artefacts of this study before
release. They are recorded here because the request that produced the study
asked for them explicitly. **No external peer review took place**: these are
self-reviews, performed against the criteria a hostile reviewer in each area
would apply. Each issue lists the fix and how the fix was verified. "Verified"
means the named test, file or artefact was actually checked, not that the issue
was acknowledged.

---

## Cycle 1 — Quantitative finance

Adversarial stance: assume the backtest is optimistic until proven otherwise.

| Issue | Severity | Reviewer comment | Fix | Verification |
|---|---|---|---|---|
| Look-ahead in features | Critical | Do features at day `t` use any information from `t` or later? | All features are built from bars `<= t-1`; labels are the return of `t`; execution maps weights at `t-1` to returns at `t` | `tests/test_core.py::test_momentum_is_past_only` perturbs the last price and asserts the prior day's weights are unchanged |
| Look-ahead in the Kronos context | Critical | A forecast formed from a window that includes `t` would be leaked | Context window ends strictly at `t-1`; the predicted bar is `t`. An earlier build ended the window at `t-2`, wasting a day of signal, and was corrected | Code re-read after edit; window slice `df.iloc[pos-CTX_DAYS:pos]` verified against the index position of `t` |
| Look-ahead in normalisation | High | Full-sample standardisation is a classic leak | Features are standardised cross-sectionally within each day only; no global mean or variance is computed | `code/alphazerobeta/azb_train.py::build_features` computes mean/std along the asset axis; no time-axis reduction |
| Universe selection bias | High | Selecting names by a trailing statistic can leak if the window is misindexed | Liquidity ranks use a 60-day trailing mean of dollar volume and are recomputed per date | `code/data/build_panel.py` rolling window with `min_periods`; the same rank matrix feeds every arm |
| Survivorship bias | High | Current index membership is not achievable history | Not removable with public data. Disclosed in the protocol, in the data section, and in threats to validity | `docs/methodology.md`, paper Section *Threats to validity* |
| Transaction costs | Critical | A cost-free backtest is meaningless, and unequal costs would bias the comparison | One cost schedule applied to every arm: 5 bps/side top liquidity decile, 15 bps/side otherwise, 30 bps/yr borrow accrued per holding day, from the source study | `code/backtest/harness.py::simulate`; `tests/test_core.py::test_simulate_costs_reduce_returns` asserts costs never increase returns |
| Cost sensitivity | Medium | A strategy that dies at 2x costs is not tradable | Full re-run at 0x, 1x, 2x | `results/tables/T9_cost_sensitivity.csv`, paper Table *cost* |
| Unequal constraints | Critical | A market-neutral arm against a long-only arm is not a comparison | Neutral arms share `sum w = 0`, `sum|w| <= 1`, box `[-1,1]`; long-only arms are separated and never ranked against them | `harness.dollar_neutral` is applied to every neutral arm; `tests/test_core.py::test_dollar_neutral_and_gross_cap` |
| Sharpe comparison by eye | Critical | Point estimates without intervals are not evidence | Block-bootstrap confidence intervals on the Sharpe *difference*; significance claimed only when the interval excludes zero | `code/statistics/significance.py`; `tests/test_core.py::test_bootstrap_ci_brackets_point`; `results/tables/T7_significance.csv` |
| i.i.d. resampling understates risk | Medium | Daily returns are autocorrelated and heteroskedastic | Stationary block bootstrap with 21-day blocks, 5000 resamples | `sharpe_diff_test(block=21, n_boot=5000)`; `tests/test_core.py::test_block_bootstrap_indices_in_range` |
| VaR/ES without context | Low | Tail metrics are unstable at short samples | ES95 reported alongside drawdown and volatility rather than alone | `results/tables/T6_portfolio_main.csv` |

---

## Cycle 2 — Machine learning

Adversarial stance: assume the model comparison is unfair or irreproducible.

| Issue | Severity | Reviewer comment | Fix | Verification |
|---|---|---|---|---|
| Forcing an RL agent into a forecasting metric | Critical | AlphaZeroBeta emits weights, not forecasts | Excluded from Level 1 by construction; the exclusion is stated in the abstract, the forecasting section and the tables | `results/tables/T5_forecasting.csv` contains only the forecast arm and the random walk; paper Section *Forecasting evaluation* |
| RL seed cherry-picking | Critical | RL results vary wildly across seeds; the best run is not a result | Three seeds on a fixed fold subset, reporting mean, standard deviation, best **and worst** | `code/statistics/seed_sensitivity.py`; `results/tables/T11_seeds.csv`; paper Table *seeds* |
| Architectural deviation hidden | Critical | A re-implementation that changes the encoder must say so | The shared-trunk per-asset encoder (D1) is stated as the largest deviation, with its consequence spelled out | `docs/reproduction.md` deviation D1; paper Section *Independent reproduction* |
| Reduced training budget passed off as a reproduction | Critical | Fewer PPO epochs and fewer steps will lower performance | Budget reductions (D2, D3, D5) disclosed, and the reproduced figure described as a lower bound, not a refutation | `docs/reproduction.md`; `experiments/configs/base.yaml` |
| Non-standard shape handling in the RL env | High | A transposed tensor silently trains a meaningless policy | Observed tensor shape is `(T, N, F)` and the environment's window builder transposes explicitly to `(N, L, F)`; the asset dimension is folded into the batch only at the PPO update | `code/alphazerobeta/azb_train.py` `Env.obs`, `rollout_weights`, and the folded `O = torch.stack(O).reshape(-1, 3*F_, AEON_WINDOW)` line |
| Hyperparameters chosen on the test set | Critical | Test-set tuning invalidates the comparison | All values frozen in `experiments/configs/base.yaml` before any test window was run | Config file committed; values referenced by table generator |
| Baselines too weak or too few | High | A comparison against trivial baselines proves little | Two neutral baselines (12-1 momentum, walk-forward ridge) and four long-only references (index, equal weight, max-Sharpe, min-correlation), all under identical costs and constraints | `code/kronos/signals.py`; `experiments/run_baselines.py` reports per-arm coverage so a missing arm fails loudly |
| Random walk ignored | High | In finance, models routinely fail to beat a naive forecast | Random walk is the reference forecaster and is tested formally with Diebold-Mariano | `code/statistics/significance.py::diebold_mariano`; `tests/test_core.py::test_dm_detects_better_forecast` |
| DM without small-sample correction | Medium | Naive DM is oversized in small samples | Harvey-Leybourne-Newbold small-sample correction applied | `significance.py::diebold_mariano` correction term; `results/tables/T5_diebold_mariano.json` |
| HAC inference for alpha | Medium | OLS standard errors are wrong under autocorrelation | Newey-West with lag 5 | `tests/test_core.py::test_newey_west_recovers_beta` recovers a known beta within 0.05 |
| Latency measured on different hardware | High | Latency figures across machines are meaningless | All latency and training-time figures measured on the same single GPU host | `results/raw/latency.json` records device and parameter count; paper Section *Computational analysis* |
| Missing measurement fabricated to fill a table | Critical | A per-name inference latency for a joint portfolio decision would be invented | Reported as not available, with the reason stated in the table note | Paper Table *compute*, note |

---

## Cycle 3 — IEEE publication criteria

Adversarial stance: assume every unsupported claim, formatting fault and
citation error will be caught.

| Issue | Severity | Reviewer comment | Fix | Verification |
|---|---|---|---|---|
| Novelty overclaimed | Critical | "First comparison" claims are almost always wrong and easy to refute | Reduced to a verifiable statement: no controlled head-to-head comparison of a financial foundation model and an RL portfolio agent under one protocol was found in the literature searched | Paper Section *Contributions*, final paragraph |
| Placeholder values in tables | Critical | Placeholders in a submitted paper are grounds for rejection | All tables are emitted as LaTeX fragments from the result files by a script; the source contains no experimental literals | `code/statistics/to_latex.py`; `paper/tables/*.tex` are generated |
| Figures not backed by data | High | Decorative figures are penalised | Every figure is generated from `results/processed` or `results/tables`; figures whose inputs are missing are skipped rather than faked | `code/statistics/generate_figures.py` |
| Citations | Critical | Every reference must be real and correctly attributed | Bibliography entries were resolved against arXiv, DOI, publisher pages or Semantic Scholar and logged with the verification URL | `docs/references_verification.md` |
| Scope inflation | High | Seven markets in the source study versus one here | Scope restriction to the S&P 500 stated in the abstract, the cross-market section and the limitations | Paper Section *Cross-market analysis* |
| Unsupported causal claim | High | A performance difference is not proof that an objective is better | Discussion frames differences as outcomes under a shared protocol, and states that the two systems optimise different objectives | Paper Section *Discussion* |
| Significance language | High | "Significant" must match the statistic | Significance asserted only where the bootstrap interval excludes zero; elsewhere the paper says the difference is not distinguishable from zero | Paper Section *Statistical analysis*; `results/tables/verdict.json` |
| Formatting | Medium | Two-column IEEE style, complete metadata | `IEEEtran` conference class, keywords, generated tables, vector figures | `make -C paper paper` |
| Similarity risk | Medium | Paraphrasing must be genuine | Systems described in original wording with attribution; no numeric similarity score claimed because no detector was run | Paper Section *Similarity-risk assessment* |
| Reproducibility statement | High | "Available on request" is not reproducible | One-command pipeline, config files, per-fold logs, generated tables and figures, and documented remote commands | `experiments/run_all.py`; `docs/reproduction.md`; `docs/experiment_log.md` |

---

## Residual issues carried into the release

These are recorded rather than resolved, because resolving them requires
resources outside this study's budget:

1. **Survivorship bias** in the universe (current index membership).
2. **Vendor feature gap**: no fundamentals, analyst revisions or sentiment.
3. **Reduced RL budget**, which makes the reproduced figure a lower bound.
4. **One seed on the full fold set** for the RL arm.
5. **One market.**
6. **No architecture-level sensitivity sweep** for either system.
