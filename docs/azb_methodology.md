# AlphaZeroBeta — Complete Experimental Methodology Extraction

Source: `/opt/data/kvab/papers/azb.txt` (text layer of `azb.pdf`).
Paper: **AlphaZeroBeta: Deep Reinforcement Learning for Market-Neutral Portfolios**, Boris Belyakov (HSE University). arXiv:2607.18001v1 [q-fin.PM] 20 Jul 2026. DOI 10.1186/s40854-026-00955-4 (Financial Innovation).

All quotes are verbatim from the extracted text. Section labels come from the paper's own headings (page numbers appear inline in the text as bare integers, e.g. `10`, `16`, `45`).

---

## 1. Data source(s) actually used

> "We conduct empirical evaluation on a dataset retrieved from Bloomberg Terminal and Financial Modeling Prep, covering calendar years 2004–2024 of historical equity and macro-financial data [12, 25]." — §4.1 Data

> "The dataset comprises daily closing prices, trading volumes, and corporate actions (e.g. dividends, stock splits), alongside detailed company-level fundamentals such as income statements, balance sheets, financial ratios, and sector classifications. Additionally, we include analyst-estimated earnings surprise signals, which capture deviations between expected and reported earnings and serve as a proxy for information shocks." — §4.1

Reference entries give the retrieval date:
> "[12] Bloomberg L.P. (2025). Bloomberg Terminal financial data [Data set]. Retrieved February 1, 2025, from https://www.bloomberg.com/professional."
> "[25] Financial Modeling Prep. (2025). Financial Modeling Prep API: aggregated financial data [Data set]. Retrieved February 1, 2025, from https://financialmodelingprep.com."

- Vendor: **Bloomberg Terminal** (primary, incl. membership feeds, fundamentals, index metadata) + **Financial Modeling Prep** (secondary fundamentals).
- Ticker format: Bloomberg index codes `000001.SS`, `ˆDJI`, `ˆFTSE`, `ˆGDAXI`, `ˆGSPC`, `ˆHSI`, `ˆNDX` (the caret renders as `ˆ` in the text; conceptually `^DJI`, `^GSPC`, etc.).
- Download method: "Bloomberg constituent history **queried by trading date**"; pseudocode uses `load_price_data(index)` / `pd.read_csv(path, parse_dates=["date"])`.
- Date downloaded: **February 1, 2025** (both vendors). SSE constituent snapshot also "as of 2025-02-01".
- Data statement / availability: no standalone "Data Availability" section. The nearest statement is Appendix D.3.3:
  > "The complete code base cannot be made publicly available, as certain components depend on licensed Bloomberg functionality and proprietary third-party libraries that are not eligible for redistribution."
- Fundamental lag rule: "we assume that each statement becomes tradable information only after a fixed lag of **60 calendar days** following the period end".

---

## 2. Universe construction

**Seven indices** (Table 1, "Overview of equity indices used in the empirical study"):

| Code | Name | Region | Size | Curr. | Notes |
|---|---|---|---|---|---|
| `000001.SS` | SSE Composite | China | **>2,200** | CNY | all A- and B-shares on Shanghai Stock Exchange; proxy for Chinese market breadth |
| `ˆDJI` | Dow Jones Ind. Avg | U.S. | **30** | USD | price-weighted; most concentrated; infrequent constituent changes |
| `ˆFTSE` | FTSE 100 | U.K. | **100** | GBP | 100 largest LSE companies by mkt cap; rebalanced quarterly; energy/financials heavy |
| `ˆGDAXI` | DAX | Germany | **40** | EUR | 40 major German cos on Xetra; free-float adjusted |
| `ˆGSPC` | S&P 500 | U.S. | **500** | USD | free-float weighted, rebalanced quarterly; ~80% of total U.S. market cap |
| `ˆHSI` | Hang Seng | Hong Kong SAR, China | **83** | HKD | largest/most liquid HKEX companies; rebalanced quarterly; includes dual-listed mainland firms |
| `ˆNDX` | NASDAQ-100 | U.S. | **100** | USD | tech-focused; modified capitalization weighting |

> "Two universes (the SSE Composite with more than 2,200 stocks and the S&P 500 with 500 constituents) meet or exceed the 500-name threshold, so the agent operates over a large cross-section on those markets." — §4.1

**Index membership / survivorship** (§4.2.4, Table 3):
> "Index membership, corporate actions, and free-float shares are sourced from Bloomberg. … Historical time-varying membership sets are used for ˆGSPC, ˆNDX, ˆFTSE, ˆGDAXI, ˆHSI, and ˆDJI. For 000001.SS, where full historical constituent history is not available in our licensed extract, we use the published constituent snapshot as of 2025-02-01 as a fixed proxy universe; **this is the main survivorship-bias caveat in our setup**."

Table 3: `000001.SS` = "Static proxy — Bloomberg constituent snapshot as of 2025-02-01." All six others = "Time-varying — Bloomberg constituent history queried by trading date."

Factor attribution inherits the caveat: "For 000001.SS, factor legs are computed on the same 2025-02-01 proxy universe described in Table 3; the resulting loadings therefore inherit the same survivorship caveat." (§6.4.1)

**Liquidity / top-decile bucketing** (§4.2.6):
> "Top decile is defined by **trailing 60-trading-day average dollar volume (ADV60 = price × volume)** within each index universe, computed with information available up to the rebalance date and **refreshed monthly (first rebalance day of each month)**."

Example tickers listed in the paper: index codes as above; sector-ETF example `XLK US EQUITY` (Tech ETF, Table C2); macro `NAPMPMI INDEX` (PMI), `FEDL01 INDEX`, `VIX INDEX`, `BCOM INDEX`, `USYC2Y10 INDEX`. No individual constituent stock tickers are named anywhere.

---

## 3. Evaluation period, windows, folds, rebalancing, execution

- **Evaluation period:** 2014–2024 out-of-sample. Full data span **2004–2024**.
- **Training window:** "rolling **three-year** windows (approximately **756 trading days**)" = 36 months.
- **Validation window:** "a **six-month** validation segment (about **126 days**) used for early stopping and hyperparameter checks".
- **Test window:** "a **six-month** out-of-sample test segment".
- **Step size:** "The block advances in **six-month steps** (sliding window)".
- **Number of folds:** "yielding **K = 22 non-overlapping test segments** through December 2024."
- **Overlap:** "training windows overlap across folds (by **30 months** between consecutive folds); the six-month validation and test slices advance disjointly".
- **First fold explicit:** "for the first fold, the model trains on **July 2010–June 2013** (36 months), validates on **July–December 2013**, and tests on **January–June 2014**; the next fold shifts forward six months (train **January 2011–December 2013**, validate **January–June 2014**, test **July–December 2014**), and so on."
- **Warm-up:** "The **2004–2010 segment is reserved for initialization of rolling estimators** (volatility, correlation, and winsorization thresholds) and feature standardization statistics, so the first training window starts in July 2010".
- **Retraining:** "At the beginning of each fold, the agent's network weights are **reset and retrained from scratch** on that fold's training window; portfolio weights start from zero (flat position)".
- **Rebalancing frequency:** daily. "Our main experiments assume a **daily rebalancing scheme**". Figure 1 caption: "daily rebalancing, retrained per fold".
- **Execution timing:** "information up to and including day **t** is used to set portfolio weights for the next trading day **t+1**, implying **execution at (or near) the close of t+1**." (§4.2.6, "Close-to-Close Trading Rule")
- **Warm-up note:** "portfolio weights converge toward their stationary distribution after roughly **ten trading days**; this short warm-up period is **included in all reported metrics**".
- Metrics computed on 22 out-of-sample windows; dispersion = mean ± std across "the Cartesian product of the K = 22 folds and **nine independent RL initializations per fold** … (**198 samples**)."

---

## 4. Feature set = the state vector

State definition:
> "At time t, the environment is characterized by a state s_t ∈ S, which includes the current portfolio weights w_t, global market signals m_t (macroeconomic indicators, commodity prices, and broad market indices), and asset-specific features x_t (price history, fundamentals, and sentiment)."
> s_t = (w_t, m_t, x_t).  **(Eq. 1)**

**Four alpha-signal families** (Fig. 1):
1. Momentum (pct change)
2. Mean Reversion (-1*mean)
3. Volatility (std)
4. Skew / Jump (skew, kurtosis)
"normalize[d] across sectors / index" — "normalized per sector and index into the input tensor".

**Multi-resolution input:** "daily, weekly, and monthly intervals". "Weekly and monthly tensors are constructed by **resampling the daily panel using end-of-period values** … ensuring synchronous alignment when the streams are concatenated."

**Table C2 — Categorization of Bloomberg-derived features** (Example Field/Ticker → Model Role):
- Price — `PX LAST` (close price)
- Volume — `PX VOLUME`
- Rolling Stats — rolling mean/std of `PX LAST`
- Technical — `EMAVG` (EMA), `MACD`, `RSI`
- Corporate Actions — `EQY DVD YLD`
- Fundamentals — `RETURN ON EQUITY`
- Earnings — EPS surprise vs. estimate
- Insiders — `NUM INSIDER SHARES SOLD`
- Sentiment — `NEWS SENTIMENT DAILY MIN` (news headline polarity)
- Governance — `ISS QUALITYSCORE` (ESG proxy)
- Macro — `NAPMPMI INDEX` (PMI)
- Rates — `FEDL01 INDEX` (Effective Fed Funds)
- Volatility — `VIX INDEX`
- Options Flow — `12MO CALL IMP VOL`
- Cross-Asset — `BCOM INDEX` (Commodities)
- Term Structure — `USYC2Y10 INDEX` (2s10s)
- Sectors — `XLK US EQUITY` (Tech ETF)
- Metadata — GICS sector / index tags

**Table C3 — Feature groups, sampling frequency, approximate signal counts (per asset):**

| Feature category | Examples | Frequency | Approx. count |
|---|---|---|---|
| Price-based | Close, open, high, low prices; log returns; rolling means and std (5/20/60-day) | Daily | ~12 |
| Volume/liquidity | Volume, turnover, bid-ask spread, rolling avg volume | Daily | ~6 |
| Technical indicators | EMA, MACD, RSI, Bollinger bands, momentum | Daily/weekly | ~18 |
| Corporate actions | Dividend yield, split factors, shares outstanding | Daily/monthly | ~4 |
| Fundamentals | EPS, ROE, book-to-market, debt-to-equity | Quarterly | ~24 |
| Earnings surprises | Actual vs expected earnings; PEAD signals | Quarterly | ~8 |
| Insider trading | Net insider purchases/sales; insider sentiment indices | Daily/weekly | ~4 |
| Sentiment | News polarity, social sentiment, CDS-implied risk | Daily | ~6 |
| Governance | ESG/gov. scores, quality ratings | Quarterly | ~5 |
| Macroeconomic | Inflation, unemployment, GDP, policy rates | Monthly/qtrly | ~12 |
| Rates & curves | 3M/10Y yields, yield-curve slope, term premia | Daily | ~6 |
| Volatility & options | VIX, implied vol surfaces, skew | Daily | ~7 |
| Cross-asset indices | Commodity, currency, credit indices | Daily | ~8 |
| Sector ETFs/indices | Sector ETF returns (e.g. XLK, XLF), rotation signals | Daily | ~12 |
| Metadata | Market cap, GICS sector, region tags | Static | ~6 |

Lookback windows used: `RET_WINDOWS = (5, 20, 60)` for returns/volume rolling mean/std; EMA-12/26, MACD, RSI-14, Bollinger-20 (2.0 std). Agent observation window = **100 daily timesteps per channel**.

Normalization: "All features are normalized to zero mean and unit variance within each training batch. Missing values are forward-filled or, where appropriate, replaced by industry-median values."
Winsorization: "Winsorize at the **[1%, 99%] PIT quantiles** fitted on the 2004-2010 warm-up segment; thresholds are frozen before evaluation opens."

---

## 5. Architecture

> "The multi-scale observations are then processed by a hierarchical convolutional pipeline, where **three 1D convolutional layers apply progressively narrower filters of sizes 8, 4, and 3, with corresponding strides of 4, 2, and 1**."
> "The resulting feature map is then flattened and passed through a **GRU with 512 hidden units**."
> "The recurrent embedding is then forwarded through **two separate fully connected branches**."

Figure 2 spec:
> "Three multi-resolution observation tensors (daily, weekly, and monthly), each resampled at the same close, are stacked along the channel axis and processed by a sequential CNN (**8/4/3 kernels, 4/2/1 strides, 32/64/64 filters**). The flattened feature map feeds a **GRU (512 hidden units)** and a **shared fully connected layer (512 units)**; the recurrent embedding then branches into a **value head** that outputs a scalar value estimate and a **policy head** that outputs an **N-dimensional weight vector** … Each head has a **512-unit ReLU hidden layer**; the policy output is bounded by a **Tanh**."

CNN layer list (from Listing D.4.3, `MultiScaleEncoder`):
```
in_ch = 3 * feature_dim
for out_ch, k, s in zip((32, 64, 64), (8, 4, 3), (4, 2, 1)):
    nn.Conv1d(prev, out_ch, kernel_size=k, stride=s); nn.ReLU(inplace=True)
self.conv = nn.Sequential(*layers, nn.Flatten())
self.gru  = nn.GRU(self.flat_dim, hidden_size=512, batch_first=True)
```
Heads:
```
policy_head = Linear(512, 512) -> ReLU -> Linear(512, num_assets) -> Tanh
value_head  = Linear(512, 512) -> ReLU -> Linear(512, 1)
```

- **No attention mechanism** anywhere (transformers explicitly rejected — §2.3: "we adopt a CNN-GRU encoder instead of a transformer stack").
- **Action space:** continuous, box `[-1, 1]^N`: `self.action_space = gym.spaces.Box(-1, 1, shape=(self.N,), dtype=np.float32)`.
- **Observation dimension:** `obs_dim = self.N * (self.F_d + weekly.shape[2] + monthly.shape[2]) + self.N` where `F_d` = daily feature count, plus `self.N` previous weights. Observation space `Box(-inf, inf, shape=(obs_dim,))`.
- **Action dimension** = universe size N: 30 (DJIA), 40 (DAX), 83 (HSI), 100 (FTSE, NDX), 500 (S&P 500), ~2,200 (SSE Composite).
- **Parameter count: NOT STATED IN PAPER** (no param count, no FLOPs).
- Action sampling: `torch.distributions.Normal(mean, scale=0.1)`, then `action = torch.tanh(sample)`.
- Recurrent hidden init: `torch.zeros(1, 1, self.hidden_size)` (1 layer).

---

## 6. PPO configuration (Table D5 + Listing D.1.4 / D.4.4)

| Component | Value |
|---|---|
| RL algorithm | Proximal Policy Optimization (Recurrent PPO) |
| Policy/value network | CNN-GRU encoder; each head 512-unit hidden ReLU layer |
| Discount factor γ | **0.99** |
| Learning rate | **3 × 10⁻⁴** (Adam optimizer) |
| PPO clip ratio | **0.20** |
| GAE parameter λ_GAE | **0.95** |
| Entropy coefficient | **0.01** |
| Value loss coefficient | **0.5** |
| Minibatch size | **256 trajectories per update** (across environments) |
| Training window length | 36 months (3 years) of daily data |
| Agent observation window (n_agent_window) | **100 daily timesteps** per channel of stacked daily/weekly/monthly CNN input |
| Rolling window for σ_p and Corr(r_p, r_m) | **60 business days** (≈3 months) |
| Number of PPO epochs per update | **10** |
| Correlation penalty λ₁ | **0.5** |
| Turnover penalty λ₂ | **0.001** |
| Number of walk-forward splits K | 22 non-overlapping 6-month test splits (Jan 2014–Dec 2024) |

Additional from pseudocode:
- Optimizer: `Adam(model.parameters(), lr=3e-4)`.
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)` — **grad clip norm = 0.5**.
- Rollout length (horizon): `collect_trajectory(self, horizon=200)` — **200 steps**.
- GAE bootstrap: "bootstrap V(s_{T+1}) = 0 at horizon".
- Advantage normalization: `advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)`.
- PPO loss: `-min(ratio*adv, clip(ratio,1±0.2)*adv).mean() + 0.5 * mse_loss(values, returns)`.
- Recurrent hidden state: 1 layer, zeros init.
- "PPO_CLIP, PPO_EPOCHS = ..., 0.20, 10"; "GAMMA, GAE_LAMBDA = 0.99, 0.95".
- Regularization: "Standard machine learning techniques, including **dropout, weight decay, and early stopping**, are applied during training." (§4.4) — **no numeric values given for dropout rate or weight-decay coefficient**.
- Batching note: "minibatches contain **32–64 parallel environments**, each representing a different index window".
- λ selection: "λ1 and λ2 are chosen through **manual optimization on pilot experiments** to balance market neutrality and turnover against Sharpe ratio".

---

## 7. Reward function

**Excess return (Eq. 4):**
> "∆r(t) = r_p(t) − r_m(t)"  where "r_p(t) and r_m(t) denote the portfolio and benchmark (index) returns at time t".

**Risk-adjusted term (Eq. 5):**
> "R⁽⁰⁾_t = ∆r(t) / σ_p(t)" — "we normalize by the portfolio's rolling volatility σ_p(t)".

**With correlation penalty (Eq. 6):**
> "R⁽¹⁾_t = ∆r(t)/σ_p(t) − λ₁ · Corr(r_p(t), r_m(t))" where "λ₁ ≥ 0 is a regularization coefficient controlling the strength of the neutrality penalty. Larger values of λ₁ push the policy toward lower beta and weaker co-movement with the index."

Correlation computed "over a rolling window (e.g. 60 business days)".

**Turnover (Eq. 7):**
> "C_t = Σ_i |∆w_i(t)|,  ∆w_i(t) = w_i(t) − w_i(t − 1)"

**Final reward (Eq. 8) — verbatim:**
> **R_t = (r_p(t) − r_m(t))/σ_p(t) − λ₁ · Corr(r_p(t), r_m(t)) − λ₂ Σ_i |∆w_i(t)|**
where "λ₂ ≥ 0 controls the strength of the turnover penalty and can be calibrated to reflect typical cost levels in the underlying markets."

Reference implementation (Listing D.1.5), default coefficients **λ₁ = 0.5, λ₂ = 0.001**, σ_p floored at **1e-8**:
```python
def alpha_zero_beta_reward(rp_t, rm_t, sigma_p_t, corr_pm_t, w, w_prev,
                           lambda1=0.5, lambda2=0.001):
    risk_adjusted = (rp_t - rm_t) / max(sigma_p_t, 1e-8)
    turnover      = float(np.abs(w - w_prev).sum())
    return float(risk_adjusted - lambda1 * corr_pm_t - lambda2 * turnover)
```

**Path dependence:** "Because the turnover penalty depends on the absolute change in weights, the environment becomes path-dependent … **Each fold therefore starts from a flat, self-financing portfolio**".

**Dollar-neutral projection / ℓ1-ball projection** (§3.5 + §4.3 + Listing D.4.2):
> "we restrict the weights to lie in an **ℓ1 ball, ‖w_{t+1}‖₁ ≤ τ, with τ = 1 in every experiment**, and we **subtract the cross-sectional mean of the action so that Σ_i w_i = 0 before the ℓ1 projection** … This combination enforces **dollar neutrality at each rebalance** while keeping **gross exposure at most one**."

Environment code:
```python
weights = action - action.mean()          # dollar-neutral
gross = np.abs(weights).sum()
if gross > 1.0:
    weights = weights / gross             # ||w||_1 <= 1
```
Per-asset bounds remain the same box as baselines: "−1 ≤ w_i ≤ 1".

σ_p / Corr estimation in the environment: "Rolling σ_p and Corr over weights HELD on [t−W, t), so r_p(s) is realised (not in-sample)", window `vol_window=60`.

---

## 8. Transaction cost schedule (Table D4, in full)

> "Transaction costs use a deterministic schedule: **5 bps per side for top-decile U.S. names** (ˆGSPC/ˆNDX/ˆDJI), **10 bps for top-decile names in the U.K., Germany, and Hong Kong SAR, China** (ˆFTSE/ˆGDAXI/ˆHSI), **15 bps for the remaining U.S./U.K./German names**, **20 bps for the remaining Hang Seng names**, and **30 bps for 000001.SS constituents**."

| Market bucket | Top-decile cost | Other names | Borrow fee | Applies to |
|---|---|---|---|---|
| U.S. large-cap | 5 bps/side | 15 bps/side | 30 bps/year | ˆGSPC, ˆNDX, ˆDJI |
| U.K./Germany | 10 bps/side | 15 bps/side | 45 bps/year | ˆFTSE, ˆGDAXI |
| Hong Kong SAR, China | 10 bps/side | 20 bps/side | 75 bps/year | ˆHSI |
| China (proxy universe) | 30 bps/side | 30 bps/side | 120 bps/year | 000001.SS |

> "Borrow fees are **30 bps/year (U.S.), 45 bps/year (U.K./Germany), 75 bps/year (Hong Kong SAR, China), and 120 bps/year (China), accrued linearly over holding days**."
> "These charges are **debited on every rebalance** so that the close-to-close convention reflects adverse drift associated with delayed fills."

**Slippage:** no separate numeric slippage parameter. Figure 1 shows "Market Orders + Slippage"; text refers to "The identical slippage/borrow model is applied to every strategy". Slippage is apparently subsumed in the bps schedule; **no explicit slippage constant is stated**.

**Realized cost magnitudes reported:**
> "the realized absolute turnover Σ_i|∆w_i| for AlphaZeroBeta averages **0.56 ± 0.11 per rebalance** across markets (**95th percentile 0.92**), translating into an all-in implementation cost of roughly **8–12 bps per day** in the most active windows."
> "the mean turnover is **≈0.50 for Decorr** and **≈0.47 for MxSharpe** across markets".
> "All reported Sharpe ratios and drawdowns are **net of these cost assumptions**."

---

## 9. Portfolio constraints

- **Gross exposure cap:** ℓ1 ball radius **τ = 1** → `Σ_i |w_i| ≤ 1`. "keeping gross exposure at most one".
- **Net exposure:** exactly zero — `Σ_i w_i = 0` enforced by subtracting cross-sectional mean before projection. "dollar-neutral by construction".
- **Per-name cap:** box `−1 ≤ w_i ≤ 1` (same as baselines).
- **Leverage:** bounded by τ = 1 (gross ≤ 1).
- **Shorting:** allowed (weights in [-1,1], dollar-neutral long/short).
- **Position limits:** only the box `[-1, 1]` and ℓ1 budget; **no explicit per-name turnover cap, no sector neutrality constraint, no hard beta constraint** (correlation penalty is soft only).
- Baselines use identical box but with `Σ_i w_i = 1` (full investment), so "their net exposure is always long one dollar".
- "This design keeps runtime near-linear in the number of assets and allows the same PPO implementation to handle 500-name U.S. universes and the 2,200-name Shanghai panel with identical code."
- Evaluation note: "All metrics are computed assuming an **initial capital base of 1 unit** in the benchmark index's currency (e.g. $1) … and **no external cash flows**."

---

## 10. Baselines

Three baselines (§4.3):

**4.3.1 Index Buy-and-Hold (Index B&H)** — Eq. 9:
> "w_i(t) = w_i(0), ∀t" — "The passive benchmark maintains the initial allocation unchanged over the entire evaluation period". "buy-and-hold only pays when index membership changes."

**4.3.2 Maximum Sharpe Ratio Portfolio (MxSharpe)** — Eq. 10:
> "max_w  E[R_p − R_f]/σ_p  s.t.  Σ_i w_i = 1,  −1 ≤ w_i ≤ 1"
> "where R_p = wᵀR is the portfolio return and σ_p = √(wᵀΣw) denotes portfolio volatility." Solved via **Sequential Least Squares Programming (SLSQP)**.

**4.3.3 Minimal Correlation Portfolio (Decorr)** — Eq. 11:
> "min_w |ρ_p| = |Cov(R_p, R_b)/(σ_p σ_b)|  s.t.  Σ_i w_i = 1,  −1 ≤ w_i ≤ 1"
> "Because |ρ_p| is non-smooth at zero, in practice we solve the equivalent smooth program **min_w ρ_p²** with SLSQP".

**Fourth baseline (ablation): RL (return-max)** — Appendix B:
> "an ablation agent that shares AlphaZeroBeta's architecture, training windows, and transaction-cost penalty but **sets λ₁ = 0**, so the reward reduces to risk-adjusted excess return minus the turnover penalty (with no correlation term)."

**Optimization lookbacks used:** **NOT STATED IN PAPER.** The text says only "MxSharpe and Decorr … are recomputed at each rebalance" and "sample means and covariances computed on **short windows** are unstable" (§7.1) — the actual estimation window length for Σ and means is never given numerically. The only named lookbacks are the 60-day ADV60 (liquidity bucketing) and the 60-day σ_p/corr window (agent reward).

---

## 11. Seeds / runs, sensitivity, ablation

- **Seeds:** "The full evaluation reported in this paper uses **nine independent seeds per fold**". "different random seeds for the policy and value networks under identical hyperparameters". Total samples = 22 folds × 9 seeds = **198**.
- **Single-seed compute:** "a single-seed sweep over all 22 folds of one index takes 57–106 h on a single V100 … or 3–6 h on the 8×A100 node". Full seven-index nine-seed sweep ≈ **12 days wall-clock** on 8×A100.
- **Training convergence:** "The variation in early training smooths out after **20k steps**"; "Most runs converge above **0.97** (cross-seed mean ≈**0.95**), with a minority settling near **0.85**" (explained variance of the value function). "the dispersion across seeds narrows after roughly **20,000 steps**."
- **Ablation (Table B1):** λ₁ = 0 return-maximizing RL. Removing the penalty "raises market correlations into the **0.4–0.6** range and deepens drawdowns while also lowering Sharpe in every market"; "maximum drawdowns deepen by approximately **7–20 percentage points**".
- **Sensitivity analysis:** authors state they did NOT run one: "These values represent **heuristic choices** that work well across the indices tested; **systematic sensitivity analysis and market-specific tuning may further improve performance and represent important directions for future research**." (Appendix D.2.2)
- **Feature attribution study** (§6.4.4): grouped permutation importance + integrated gradients, 4 feature blocks. "The marginal drops are **37% (price/momentum), 18% (volatility/regime), 12% (macro/cross-asset), and 9% (sentiment/flows); the remaining 24% reflects interaction effects and redundancy**." Integrated gradients "averaged over **256 random test-window evaluations per market**".

**Table B1 — AlphaZeroBeta vs RL (return-max):**

| Index | Method | Sharpe | Max DD | Corr |
|---|---|---|---|---|
| 000001.SS | AlphaZeroBeta | 1.63 ± 0.38 | −0.34 ± 0.11 | −0.02 ± 0.05 |
| 000001.SS | RL (return-max) | 0.95 ± 0.27 | −0.52 ± 0.18 | 0.42 ± 0.16 |
| ˆFTSE | AlphaZeroBeta | 0.94 ± 0.19 | −0.28 ± 0.09 | −0.02 ± 0.08 |
| ˆFTSE | RL (return-max) | 0.65 ± 0.21 | −0.35 ± 0.12 | 0.51 ± 0.19 |
| ˆGDAXI | AlphaZeroBeta | 0.86 ± 0.23 | −0.16 ± 0.11 | 0.05 ± 0.04 |
| ˆGDAXI | RL (return-max) | 0.75 ± 0.24 | −0.33 ± 0.14 | 0.48 ± 0.17 |
| ˆGSPC | AlphaZeroBeta | 1.61 ± 0.48 | −0.26 ± 0.15 | 0.15 ± 0.09 |
| ˆGSPC | RL (return-max) | 1.12 ± 0.31 | −0.38 ± 0.13 | 0.57 ± 0.15 |
| ˆHSI | AlphaZeroBeta | 1.04 ± 0.33 | −0.21 ± 0.22 | 0.01 ± 0.04 |
| ˆHSI | RL (return-max) | 0.58 ± 0.25 | −0.41 ± 0.16 | 0.49 ± 0.18 |
| ˆNDX | AlphaZeroBeta | 1.48 ± 0.41 | −0.32 ± 0.10 | 0.07 ± 0.06 |
| ˆNDX | RL (return-max) | 1.10 ± 0.28 | −0.41 ± 0.15 | 0.61 ± 0.14 |
| ˆDJI | AlphaZeroBeta | 1.20 ± 0.28 | −0.27 ± 0.14 | 0.03 ± 0.07 |
| ˆDJI | RL (return-max) | 0.88 ± 0.26 | −0.36 ± 0.12 | 0.52 ± 0.12 |

---

## 12. Full numeric results (Table 4)

**Table 4 — Performance comparison across methods** (mean ± std; AlphaZeroBeta dispersion over 198 samples; Index B&H zero dispersion single path; Decorr/MxSharpe dispersion = fold-to-fold over 22 windows).

| Index | Method | Sharpe ratio | Max drawdown | Correlation |
|---|---|---|---|---|
| **000001.SS** | AlphaZeroBeta | **1.63 ± 0.38** | −0.34 ± 0.11 | −0.02 ± 0.05 |
| | Index B&H | 0.30 ± 0.00 | −0.71 ± 0.00 | 1.00 ± 0.00 |
| | Decorr | −0.24 ± 0.20 | −0.24 ± 0.08 | −0.05 ± 0.21 |
| | MxSharpe | 0.80 ± 0.26 | −0.63 ± 0.36 | 0.89 ± 0.09 |
| **ˆFTSE** | AlphaZeroBeta | **0.94 ± 0.19** | −0.28 ± 0.09 | −0.02 ± 0.08 |
| | Index B&H | 0.23 ± 0.00 | −0.43 ± 0.00 | 1.00 ± 0.00 |
| | Decorr | −0.07 ± 0.18 | −0.44 ± 0.07 | −0.02 ± 0.17 |
| | MxSharpe | 0.37 ± 0.16 | −0.22 ± 0.20 | 0.73 ± 0.21 |
| **ˆGDAXI** | AlphaZeroBeta | **0.86 ± 0.23** | −0.16 ± 0.11 | 0.05 ± 0.04 |
| | Index B&H | 0.51 ± 0.00 | −0.57 ± 0.00 | 1.00 ± 0.00 |
| | Decorr | −0.16 ± 0.17 | −0.49 ± 0.12 | −0.10 ± 0.23 |
| | MxSharpe | 0.11 ± 0.25 | −0.28 ± 0.21 | 0.44 ± 0.35 |
| **ˆGSPC** | AlphaZeroBeta | **1.61 ± 0.48** | −0.26 ± 0.15 | 0.15 ± 0.09 |
| | Index B&H | 0.72 ± 0.00 | −0.67 ± 0.00 | 1.00 ± 0.00 |
| | Decorr | −0.27 ± 0.17 | −0.88 ± 0.14 | −0.11 ± 0.19 |
| | MxSharpe | 0.91 ± 0.44 | −0.09 ± 0.31 | 0.92 ± 0.07 |
| **ˆHSI** | AlphaZeroBeta | **1.04 ± 0.33** | −0.21 ± 0.22 | 0.01 ± 0.04 |
| | Index B&H | 0.10 ± 0.00 | −0.67 ± 0.00 | 1.00 ± 0.00 |
| | Decorr | −0.19 ± 0.18 | −0.63 ± 0.16 | −0.07 ± 0.18 |
| | MxSharpe | 0.62 ± 0.31 | −0.18 ± 0.27 | 0.81 ± 0.13 |
| **ˆNDX** | AlphaZeroBeta | **1.48 ± 0.41** | −0.32 ± 0.10 | 0.07 ± 0.06 |
| | Index B&H | 0.90 ± 0.00 | −0.75 ± 0.00 | 1.00 ± 0.00 |
| | Decorr | −0.12 ± 0.15 | −0.53 ± 0.13 | −0.08 ± 0.22 |
| | MxSharpe | 0.85 ± 0.37 | −0.19 ± 0.26 | 0.86 ± 0.11 |
| **ˆDJI** | AlphaZeroBeta | **1.20 ± 0.28** | −0.27 ± 0.14 | 0.03 ± 0.07 |
| | Index B&H | 0.58 ± 0.00 | −0.53 ± 0.00 | 1.00 ± 0.00 |
| | Decorr | −0.18 ± 0.19 | −0.46 ± 0.11 | −0.06 ± 0.15 |
| | MxSharpe | 0.76 ± 0.29 | −0.22 ± 0.24 | 0.79 ± 0.13 |

**Headline average:**
> "On average AlphaZeroBeta delivers a Sharpe ratio of **1.25 (cross-market standard deviation 0.30)**, computed as the simple mean and standard deviation of the per-market entries in Table 4; the best convex baseline in each market averages **0.70 (standard deviation 0.19)**. AlphaZeroBeta also maintains correlations within **±0.15 of zero** in every region."

> "Drawdown is competitive but not uniformly best: AlphaZeroBeta has the smallest (most positive) maximum drawdown only on **ˆGDAXI**, while MxSharpe or Decorr record less-severe drawdowns in the other markets."

Note: **no confidence intervals (CI) are reported anywhere** — dispersion is reported as standard deviation across folds/seeds. **Annualized return is not tabulated** (Table 4 has only Sharpe, MDD, correlation). §5.3 lists cumulative return / net exposure / gross exposure as *visual-only* diagnostics (Figure 4 / Appendix A), not tabulated numerically.

**Table 2 — Descriptive statistics of daily log returns, 2014–2024 (benchmark indices):**

| Index | N | Mean | Median | Std | Min | Max | Skew | Ex. Kurt. | JB p |
|---|---|---|---|---|---|---|---|---|---|
| 000001.SS | 2708 | 0.00017 | 0.00047 | 0.01298 | −0.08873 | 0.07755 | −1.008 | 8.319 | < 0.001 |
| ˆDJI | 2797 | 0.00036 | 0.00066 | 0.01074 | −0.13842 | 0.10764 | −0.945 | 23.322 | < 0.001 |
| ˆFTSE | 2814 | 0.00009 | 0.00054 | 0.00967 | −0.11512 | 0.08667 | −0.870 | 13.364 | < 0.001 |
| ˆGDAXI | 2820 | 0.00031 | 0.00080 | 0.01195 | −0.13055 | 0.10414 | −0.591 | 10.293 | < 0.001 |
| ˆGSPC | 2797 | 0.00043 | 0.00067 | 0.01093 | −0.12765 | 0.08968 | −0.812 | 16.111 | < 0.001 |
| ˆHSI | 2742 | −0.00001 | 0.00026 | 0.01330 | −0.09879 | 0.08693 | −0.019 | 3.776 | < 0.001 |
| ˆNDX | 2797 | 0.00065 | 0.00117 | 0.01352 | −0.13003 | 0.09597 | −0.537 | 7.361 | < 0.001 |

---

## 13. Statistical tests / regressions

**Table 5 — Extended factor regression, Eq. (18), Newey–West t-stats in parentheses:**

> "re_{p,t} = α_p + β_{m,p} r_{MKT,t} + β_{s,p} SMB_t + β_{v,p} HML_t + β_{q,p} RMW_t + β_{mom,p} MOM_t + β_{rev,p} REV_t + β_{qual,p} QUAL_t + ε_{p,t}"  **(Eq. 18)**

Factors: MKT, SMB, HML, RMW (Fama–French, CMA excluded due to collinearity with QUAL), MOM (Carhart), REV (Jegadeesh), QUAL (quality-minus-junk proxy built from ROE, gross profitability, leverage filters per Asness et al.).
> "**Newey–West heteroskedasticity- and autocorrelation-robust t-statistics accompany all coefficients.**"

Six markets used (ˆDJI excluded: "local SMB/HML/RMW/QUAL spread construction is economically thin and statistically unstable").

| Market | α | MKT | SMB | HML | RMW | MOM | REV | QUAL |
|---|---|---|---|---|---|---|---|---|
| 000001.SS | 0.00047*** (3.15) | −0.002 (−0.14) | 0.032 (1.56) | 0.009 (0.48) | 0.005 (0.44) | 0.114*** (3.55) | −0.029* (−1.76) | −0.013 (−0.99) |
| ˆFTSE | 0.00029** (2.12) | −0.004 (−0.22) | −0.017 (−0.88) | 0.011 (0.55) | 0.014 (1.02) | 0.091*** (3.74) | −0.027* (−1.76) | −0.009 (−0.77) |
| ˆGDAXI | 0.00033*** (2.76) | 0.006 (0.42) | −0.021 (−1.02) | 0.019 (1.05) | 0.004 (0.41) | 0.105*** (4.09) | −0.031** (−2.24) | −0.022 (−1.19) |
| ˆGSPC | 0.00042*** (3.01) | 0.011 (0.62) | 0.028 (1.44) | −0.014 (−0.88) | 0.006 (0.55) | 0.137*** (4.92) | −0.042** (−2.11) | −0.011 (−1.08) |
| ˆHSI | 0.00031** (2.09) | 0.009 (0.55) | 0.013 (0.72) | 0.018 (0.99) | 0.011 (0.88) | 0.128*** (3.88) | −0.036** (−2.01) | −0.016 (−1.06) |
| ˆNDX | 0.00055*** (3.44) | 0.019 (0.88) | 0.044* (1.79) | −0.022 (−1.31) | −0.031* (−1.96) | 0.162*** (5.12) | −0.038* (−1.82) | −0.067** (−2.32) |

`* p < 0.10, ** p < 0.05, *** p < 0.01`.

Other tests:
- **Jarque–Bera** normality test on index daily log returns (all p < 0.001).
- **Engle ARCH LM test** — mentioned, **NOT tabulated**: "A standard Engle ARCH LM test (not tabulated) further indicates pronounced conditional heteroscedasticity for all indices".
- **Newey–West** — used for factor regression t-stats only.
- **Bootstrap:** only inside GAE (value bootstrap V(s_{T+1})=0). No statistical bootstrap / block bootstrap for performance inference.
- **No Diebold–Mariano, no White reality-check, no deflated Sharpe / PBO test** despite citing Bailey et al. backtest-overfitting paper.
- Factor data: "U.S. factors (Ken French library data) are merged at New York close and used contemporaneously for ˆGSPC and ˆNDX. For European indices … and Asian indices …, where no equivalent public factor library is available, we construct MKT/SMB/HML/RMW directly from local index constituents using the standard Fama–French sort breakpoints".

---

## 14. Reproducibility

- **Official code released? NO.** "The complete code base **cannot be made publicly available**, as certain components depend on licensed Bloomberg functionality and proprietary third-party libraries that are not eligible for redistribution." (Appendix D.3.3)
- **No GitHub link, no Zenodo, no repository URL anywhere in the paper.** Searches for "github"/"reposit" returned nothing relevant.
- What IS provided: pseudocode Listings D.1.1–D.1.5 (data pre-processing, RL environment, agent training loop, reward) and D.4.1–D.4.4 (production data loader, market-neutral env, multi-scale policy net, recurrent PPO agent skeleton), plus Appendix D.2 config tables (Table D4 costs, Table D5 hyperparameters).
- Recommended stack: "pandas, numpy, torch, gymnasium, the ta technical-analysis package … and stable-baselines3. In practice, one can rely on the recurrent extensions in **sb3-contrib (RecurrentPPO)**".
- **Is data public? NO.** Data is licensed Bloomberg Terminal + Financial Modeling Prep; "All features used in our experiments are subject to the licensing constraints discussed in Appendix D." Reference [65] cites sb3-contrib: "Retrieved February 1, 2025, from https://pypi.org/project/sb3-contrib/".
- Replication requirement stated: "researchers require a high-quality historical dataset containing prices, volumes, corporate actions, and macro variables for the indices described in the main paper."
- **No explicit Data Availability Statement section** in the paper.

---

## 15. Limitations the authors themselves admit

1. **Survivorship bias (SSE Composite):** "For 000001.SS, where full historical constituent history is not available in our licensed extract, we use the published constituent snapshot as of 2025-02-01 as a fixed proxy universe; **this is the main survivorship-bias caveat in our setup**." (§4.2.4; repeated for factor attribution §6.4.1)
2. **Constraint asymmetry vs baselines:** "The comparison is **leverage-neutral in terms of gross exposure yet differs in net exposure**: baselines are net-long by design, whereas AlphaZeroBeta is dollar-neutral by construction. **This limitation should be considered when interpreting performance comparisons** in Section 6." (§4.3)
3. **Not uniformly best drawdown:** "Drawdown is competitive but not uniformly best: AlphaZeroBeta has the smallest (most positive) maximum drawdown only on ˆGDAXI." (§6)
4. **No systematic hyperparameter sensitivity:** "These values represent heuristic choices that work well across the indices tested; **systematic sensitivity analysis and market-specific tuning may further improve performance and represent important directions for future research**." (§D.2.2)
5. **λ₁/λ₂ chosen by manual pilot optimization:** "chosen through manual optimization on pilot experiments".
6. **No live-execution modeling:** "Integrating real-time deployment constraints, including **execution latency and slippage modeling**, would narrow the gap between backtest and live deployment." (§7.2) — slippage is not explicitly parameterized in the backtest.
7. **Only equity indices / single asset class:** future work = "extending the framework to multi-asset portfolios (fixed income, commodities, crypto)". (§7.2)
8. **Factor attribution caveats:** "Coefficient estimates are not reliable" for ˆDJI; CMA excluded due to collinearity with QUAL.
9. **Warm-up dip:** "portfolio weights converge toward their stationary distribution after roughly ten trading days; this short warm-up period is included in all reported metrics, but it also **explains the small dip in performance evident at the beginning of each out-of-sample window**."
10. **LLM use disclosed:** "the author used ChatGPT (OpenAI) and Claude Code (Anthropic) to assist with editing and LaTeX formatting."

---

## OPEN QUESTIONS / WHAT IS NOT STATED

Items needed to reproduce the paper that are absent or under-specified in the text:

1. **Exact ticker / constituent list** — no individual constituent tickers or ISINs are listed for any of the 7 universes (only index codes and counts). SSE proxy snapshot contents not enumerated.
2. **Data availability statement** — none; data is proprietary/licensed, not shareable, and no derived dataset is provided.
3. **Official code repository** — explicitly NOT released; no URL/GitHub/Zenodo.
4. **Parameter count** of the CNN-GRU network — never reported.
5. **Exact feature dimension F_d** (daily feature count) — only `~` approximate per-group counts (Table C3) and the formula `obs_dim = N*(F_d + F_weekly + F_monthly) + N`. Total per-asset feature count is not given as a single number.
6. **Weekly / monthly feature counts** (F_weekly, F_monthly) — not stated numerically.
7. **Precise set of rolling lookbacks per feature family** — only `RET_WINDOWS=(5,20,60)` and EMA-12/26, RSI-14, Bollinger-20 are given; the full mapping of which features use which window is not exhaustive.
8. **Dropout rate and weight-decay coefficient** — "dropout, weight decay, and early stopping are applied" but no numeric values.
9. **Early-stopping patience / criterion** — "6-month validation … used for early stopping and model selection" but patience/metric/threshold not stated.
10. **MxSharpe / Decorr estimation lookback window** — "short windows" is qualitative only; the covariance/mean estimation window length is never given. Same for the risk-free rate series (only "effective federal funds rate (lagged)").
11. **Slippage constant** — no explicit slippage bps; only the combined cost schedule and a qualitative "market orders + slippage" diagram.
12. **Risk-free rate source and exact series** — "effective federal funds rate (lagged to avoid look-ahead)" with no vendor/ticker.
13. **Number of PPO parallel environments** — "32–64 parallel environments" is a range, not a fixed value; total steps per fold not stated (only convergence "after 20k steps").
14. **Total training iterations / timesteps per fold** — not stated (pseudocode default `train(iterations=10)`, `horizon=200`).
15. **Whether λ₁/λ₂ were re-tuned per market** — states "kept fixed across markets in our main experiments" but pilot-tuning detail is absent.
16. **Annualized return / turnover / volatility per market** — Table 4 reports only Sharpe, MDD, correlation; annual return not tabulated.
17. **Confidence intervals** — none reported; only mean ± std across folds×seeds.
18. **Statistical significance tests for Sharpe differences** (Diebold–Mariano, deflated Sharpe, PBO) — not performed despite citing the backtest-overfitting literature.
19. **Exact bootstrap procedure** for performance uncertainty — none.
20. **Exact ADV60 → top-decile threshold values** per market — bucketing rule stated, thresholds not given.
21. **Handling of the 30-name DJIA and 40-name DAX** ℓ1-projection feasibility — not discussed.
22. **Reproducibility of the walk-forward first fold start** — training start July 2010 is given, but the exact first/last date of each of the 22 folds is only implied.
23. **Borrow-cost application to long positions / short-only** — "borrow fees … accrued linearly over holding days" but whether applied only to short legs is not explicit.
24. **How the benchmark return r_m is defined** (index total return vs price return) — not stated; Table 2 uses "local index currency" log returns.
25. **Random-seed values** — only "nine independent seeds" stated; actual seed integers not given.
26. **Hyperparameter search space / method** — "manual optimization on pilot experiments", no grid/range.
27. **Compute cost per baseline** stated only qualitatively ("≤1 day on a modern multi-core CPU"); no exact machine specs for baseline runs.
28. **Whether transaction costs are also charged inside the reward/environment beyond the λ₂ penalty** — the environment charges realized costs in Listing D.1.3 (toy env) but the production env (D.4.2) shows only the λ₂ turnover penalty term; the relationship between the reward penalty and the realized backtest cost debit is not fully reconciled.
