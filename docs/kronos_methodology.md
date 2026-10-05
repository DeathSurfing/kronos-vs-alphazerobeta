# Kronos — Complete Model + Evaluation Methodology Extraction

**Source:** `/opt/data/kvab/papers/kronos.txt` (text extracted with PyMuPDF from the PDF).
**Paper:** "Kronos: A Foundation Model for the Language of Financial Markets" — Yu Shi, Zongliang Fu, Shuo Chen, Bohan Zhao, Wei Xu, Changshui Zhang, Jian Li (Tsinghua University).
**Identifier:** `arXiv:2508.02739v1 [q-fin.ST] 2 Aug 2025`. Task brief says AAAI 2026; the text itself only shows the arXiv v1 stamp.
**Code:** `https://github.com/shiyu-coder/Kronos` — *"Our pre-trained model is publicly available at https://github.com/shiyu-coder/Kronos."*

All quotes verbatim from the source text. Where the text is silent: **NOT STATED IN PAPER**.

---

## 0. One-paragraph model summary (verbatim)

> "Kronos abstracts financial K-line sequences as a discrete language and implements this via a two-phase framework illustrated in Figure 2: (1) K-line Tokenization and (2) Autoregressive Pre-training."

> "In the first phase, we design a specialized Transformer-based tokenizer to quantize a continuous, multivariate K-line sequence into a corresponding sequence of discrete tokens, via a learnable codebook. Each K-line item (OHLCVA) is treated as an individual instance and quantized into a discrete token. Each token is composed of a coarse-grained subtoken and a fine-grained subtoken."

> "In the second phase, an autoregressive decoder-only Transformer is pre-trained on these tokenized sequences, using the standard next-token prediction objective to sequentially forecast both subtoken levels at each future time step conditioned on the given historical context."

Problem statement (verbatim):
> "Let D-dimensional vector x_t ∈ R^D denote the K-line observation at discrete time t... In this work, we fix the dimension D = 6 to represent OHLCVA attributes (Open, High, Low, Close prices, trading Volume, and Amount)." and "we fix the dimension D = 6 ... The rationale for this input choice is detailed in Appendix H (Q1)."

---

## 1. PRETRAINING DATA

### 1.1 Headline counts and provenance
- **Records:** *"over 12 billion K-line records"* (abstract); *"over 12 billion K-line records from over 45 global exchanges"* (abstract); *"over 12 billion K-line records drawn from over 45 global markets and 7 temporal granularities."*
- **Exchanges:** *"45 global exchanges"* / *"over 45 global markets"*. Appendix B softens/qualifies: *"The dataset is aggregated from over 40 exchanges across more than 30 countries, comprising a diverse range of asset classes at multiple temporal frequencies (1-minute to weekly)."*
- **Table 13 approximate totals (verbatim):** `Approximate Totals | 96569 | 12.11B | –` where columns are `Exchange / Country | Asset Types | Timeframes | # Assets | # Observations | Start Date`. So: **96,569 assets, 12.11B observations.**
- **Granularities:** Text claims **7 sampling frequencies**: *"Our dataset spans over 12 billion observations across 7 sampling frequencies, encompassing a broad spectrum of asset classes drawn from 45 global exchanges."* Table 4 and Table 13 use timeframes `T (1-min), 5T, 15T, 30T, H (1-hour), D (1-day), W (1-week)`. NOTE: Table 4 lists **12** frequency rows including 10min, 20min, 40min, 60min, 2H, 4H — i.e. the "7" and the Table 4 frequency list are inconsistent in the text. See OPEN QUESTIONS.
- **Date ranges:** Table 13 per-exchange "Start Date" values include **2000/1/1** (Nasdaq: *"2000/1/1"*; NYSE: *"2000/1/1"*), **1990/12/19** (Shenzhen: *"1990/12/19"*; Shanghai: *"1990/12/19"*), **2021/1/31** (Binance), **2010/1/1** (China Future), and **2020/1/31** or **2020/2/3** for the large majority of exchanges. **End date for pre-training:** *"The pre-training data for Kronos extends up to June 2024."*
- **Data sources (vendors):** **NOT STATED IN PAPER.** No data vendor / feed provider is named for the pre-training corpus. (Qlib is cited only as the *evaluation* backtest source — see §6.5.)
- **Asset classes (verbatim):** equities are emphasised as dominant: *"The raw pre-training corpus exhibits a natural imbalance across asset classes, with equities being more prevalent than cryptocurrencies, futures, and foreign exchange (forex) assets."* Table 13 includes exchange-listed Stock, ETF, Stock Index, Future (China futures), Crypto/Perpetual Swap (Binance), and Foreign Exchange (1,023 pairs, 462,434,562 obs, start 2020/1/31).

### 1.2 Table 13 — per-exchange descriptive statistics (verbatim rows, abbreviated)
| Exchange / Country | Asset Types | Timeframes | # Assets | # Observations | Start Date |
|---|---|---|---|---|---|
| Binance | Crypto, Perpetual Swap | T, 5T, 15T, 30T, H, D, W | 997 | 1,237,002,843 | 2021/1/31 |
| Athens Stock Exchange | Stock, ETF | D, W | 180 | 226,315 | 2023/4/11 |
| Beijing Stock Exchange | Stock | 5T, 15T, 30T, H, D, W | 272 | 10,197,628 | 2021/11/19 |
| Brazil Stock Exchange | Stock, ETF | D, W | 2,058 | 1,315,290 | 2020/1/31 |
| Moscow Exchange | Stock, ETF | D, W | 514 | 567,351 | 2020/1/31 |
| Euronext Amsterdam | Stock, ETF | D, W | 514 | 602,083 | 2020/1/31 |
| Australian Securities Exchange | Stock, ETF | 5T, 15T, 30T, H, D, W | 3,381 | 86,613,897 | 2020/1/31 |
| Stock Exchange of Thailand | Stock, ETF | 5T, 15T, 30T, H, D, W | 1,664 | 49,590,394 | 2020/1/31 |
| Bombay Stock Exchange | Stock, ETF | 5T, 15T, 30T, H, D, W | 5,491 | 284,428,211 | 2020/1/31 |
| Euronext Brussels | Stock, ETF | D, W | 166 | 195,491 | 2020/1/31 |
| Bucharest Stock Exchange | Stock, ETF | D, W | 247 | 176,080 | 2020/1/31 |
| Budapest Stock Exchange | Stock, ETF | D, W | 50 | 57,586 | 2022/1/14 |
| Buenos Aires Stock Exchange | Stock | D, W | 183 | 225,352 | 2020/1/31 |
| Colombo Stock Exchange | Stock | D, W | 292 | 372,627 | 2020/1/31 |
| Copenhagen Stock Exchange | Stock, ETF | D, W | 825 | 617,464 | 2020/1/31 |
| Frankfurt Stock Exchange | Stock, ETF | D, W | 17,054 | 21,547,744 | 2020/1/31 |
| Ghana Stock Exchange | Stock | D, W | 44 | 57,690 | 2020/1/31 |
| Hong Kong Stock Exchange | Stock, ETF | 5T, 15T, 30T, H, D, W | 3,500 | 359,434,220 | 2020/1/31 |
| Japan Exchange Group | Stock, ETF | 5T, 15T, 30T, H, D, W | 4,467 | 280,601,980 | 2020/1/31 |
| Indonesia Stock Exchange | Stock | 5T, 15T, 30T, H, D, W | 935 | 38,627,125 | 2020/1/31 |
| Borsa Istanbul | Stock | D, W | 627 | 784,147 | 2020/1/31 |
| Johannesburg Stock Exchange | Stock, ETF | D, W | 562 | 681,587 | 2020/1/31 |
| Pakistan Stock Exchange | Stock, ETF | D, W | 660 | 595,505 | 2020/1/31 |
| Kuala Lumpur Stock Exchange | Stock, ETF | 5T, 15T, 30T, H, D, W | 1,150 | 45,938,559 | 2020/1/31 |
| Korea Exchange | Stock, ETF | 5T, 15T, 30T, H, D, W | 2,928 | 205,061,301 | 2020/1/31 |
| Lima Stock Exchange | Stock, ETF | D, W | 166 | 63,503 | 2020/1/31 |
| Euronext Lisbon | Stock, ETF | D, W | 60 | 65,753 | 2020/1/31 |
| London Stock Exchange | Stock, ETF | 5T, 15T, 30T, H, D, W | 8,660 | 177,947,624 | 2020/1/31 |
| Luxembourg Stock Exchange | Stock | D, W | 5 | 7,598 | 2020/1/31 |
| Madrid Stock Exchange | Stock, ETF | D, W | 309 | 331,745 | 2020/1/31 |
| Mexican Stock Exchange | Stock, ETF | D, W | 775 | 937,637 | 2020/1/31 |
| Nasdaq Stock Exchange | Stock, ETF | T, 5T, 15T, 30T, H, D, W | 8,725 | 2,478,662,459 | 2000/1/1 |
| National Stock Exchange of India | Stock, ETF | 5T, 15T, 30T, H, D, W | 2,554 | 242,429,169 | 2020/1/31 |
| New York Stock Exchange | Stock, ETF | T, 5T, 15T, 30T, H, D, W | 7,073 | 2,133,143,549 | 2000/1/1 |
| Euronext Paris | Stock, ETF | D, W | 1,781 | 1,981,059 | 2020/1/31 |
| Philippine Stock Exchange | Stock, ETF | 5T, 15T, 30T, H, D, W | 351 | 4,388,378 | 2020/1/31 |
| Prague Stock Exchange | Stock | D, W | 50 | 62,666 | 2020/1/31 |
| Santiago Stock Exchange | Stock | D, W | 225 | 160,638 | 2020/1/31 |
| Shenzhen Stock Exchange | Stock, ETF | T, 5T, 15T, 30T, H, D, W | 3,519 | 1,754,519,331 | 1990/12/19 |
| Shenzhen Stock Exchange (B-shares) | Stock | 5T, 15T, 30T, H, D, W | 46 | 4,198,702 | 2020/2/3 |
| Shanghai Stock Exchange | Stock, ETF | T, 5T, 15T, 30T, H, D, W | 3,064 | 1,967,996,343 | 1990/12/19 |
| Shanghai Stock Exchange (B-shares) | Stock | 5T, 15T, 30T, H, D, W | 50 | 4,526,152 | 2020/2/3 |
| Stockholm Stock Exchange | Stock, ETF | D, W | 1,305 | 1,463,722 | 2020/1/31 |
| SIX Swiss Exchange | Stock, ETF | D, W | 1,981 | 2,451,675 | 2020/1/31 |
| Taiwan Stock Exchange | Stock, ETF | 5T, 15T, 30T, H, D, W | 1,252 | 71,619,260 | 2020/1/31 |
| Toronto Stock Exchange | Stock, ETF | D, W | 3,035 | 3,356,561 | 2020/1/31 |
| Vienna Stock Exchange | Stock | D, W | 98 | 123,643 | 2020/1/31 |
| China | Future | T, 5T, 15T, D | 75 | 63,318,960 | 2010/1/1 |
| \ | Foreign Exchange | 5T, 15T, 30T, H, D, W | 1,023 | 462,434,562 | 2020/1/31 |
| *(Stock Index rows continue)* Australia 40/183,158; Belgium 5/8,109; Brazil 3/4,766; Canada 18/27,622; China 597/55,884,065; Germany 18/28,622; Spain 2/3,257; France 38/55,945; Britain 51/5,355,869; Greece 1/1,589; Hong Kong, China 4/453,016; Hungary 1/1,602; Indonesia 2/47,816; India 113/3,189,450; Japan 9/125,024; Korea 5/274,292; Mexico 1/1,619; Malaysia 2/3,145; Netherlands 4/6,475; Pakistan 3/3,184; Philippines 2/3,187; Portugal 1/1,632; Romania 5/7,726; Russia 15/19,079; Sweden 11/16,389; Thailand 4/5,005; Taiwan, China 1/85,318; America 670/37,887,535; all Stock Index rows start 2020/1/31 except China 2020/2/3 | | | | | |

Table 13 caption: *"Descriptive statistics of the multi-exchange, multi-asset K-line dataset. The timeframe abbreviations are: T (1-min), H (1-hour), D (1-day), W (1-week)."*

### 1.3 Preprocessing / cleaning pipeline (Appendix B, verbatim)
Two-stage: (1) missing-value processing, (2) low-quality segment filtering.

> "Missing Value Processing ... We employ a field-specific strategy to handle missing values, which are typically represented as 'NaN' (Not a Number) or 'Inf' (Infinity)."
> "**Price Fields (Open, High, Low, Close):** For price-related fields, we treat missing values as hard boundaries. Inspired by TimeMOE (Xiaoming et al. 2025), we partition the time series into contiguous, valid sub-sequences at each occurrence of a missing price value. This approach ensures that each resulting segment maintains its internal temporal integrity without unwarranted imputation."
> "**Volume and Amount Fields:** In contrast, for volume and amount fields, which primarily serve as auxiliary covariates, we impute missing values with zero. To enhance model robustness to sparse or unavailable volumetric data, we introduce a regularization technique: during training, both volume and amount are randomly set to zero for 5% of the input samples. This encourages the model to learn to make effective predictions from price information alone."

Low-quality filtering (Algorithm 1) steps (verbatim headers):
> "**Structural Break Segmentation.** The initial filtering stage partitions the series based on significant price discontinuities. We identify these breaks by calculating the relative price jump between the previous bar's close and the current bar's open (|open_t/close_{t−1} − 1|). If this jump exceeds a frequency-specific threshold, the sequence is split."
> "**Filtering of Illiquid Periods.** ... A bar is deemed illiquid if its trading volume is zero or near-zero. If the number of consecutive illiquid bars exceeds a frequency-dependent threshold, the corresponding period is flagged as invalid."
> "**Filtering of Price Stagnation.** ... where the closing price remains constant over an extended duration."
> "**Final Segment Validation.** ... only the resulting sub-segments that meet the frequency-specific minimum length requirement (Θ_min len in Table 4) are retained for the final pre-training dataset."

**Table 4 — frequency-specific filtering parameters (verbatim):**
| Frequency | Min. Length (bars) | Price Jump Threshold | Illiquid | Stagnant |
|---|---|---|---|---|
| 1min | 2048 | 0.10 | 15 | 45 |
| 5min | 1024 | 0.15 | 3 | 10 |
| 10min | 512 | 0.15 | 3 | 6 |
| 15min | 512 | 0.15 | 2 | 5 |
| 20min | 512 | 0.15 | 2 | 5 |
| 30min | 512 | 0.20 | 2 | 3 |
| 40min | 256 | 0.20 | 1 | 3 |
| 60min | 256 | 0.20 | 1 | 3 |
| 2H | 128 | 0.25 | 1 | 3 |
| 4H | 128 | 0.25 | 1 | 3 |
| Daily | 128 | 0.30 | 1 | 3 |
| Weekly | 16 | 0.50 | 0 | 2 |

Caption: *"Frequency-specific parameters for the low-quality data filtering pipeline. Thresholds are adjusted to reflect the distinct dynamics of different time frequencies."*

### 1.4 Pre-training normalisation (Appendix C)
> "Each input K-line sequence x = (x_1, x_2, ..., x_T), where x_t ∈ R^D, is normalized in a two-step procedure before being passed to the tokenizer. First, we apply z-score normalization independently to each of the D feature dimensions (e.g., Open, High, Low, Close, Volume and Amount). Second, to mitigate the potential impact of extreme outliers on training stability, the normalized values are clipped to the range [−5, 5]."

### 1.5 Pre-training data rebalancing (Appendix C)
> "The raw pre-training corpus exhibits a natural imbalance across asset classes, with equities being more prevalent than cryptocurrencies, futures, and foreign exchange (forex) assets. To prevent potential underfitting on these less-represented classes, we apply strategic resampling to the training data. Specifically, we increase the sampling weights for data from crypto, futures, and forex markets."

### 1.6 Data volume caveat
Table 13 "Approximate Totals 96569 / 12.11B" vs. text "over 12 billion K-line records from 45 global exchanges" vs. Appendix B "over 40 exchanges across more than 30 countries". Handle these as approximate/slightly inconsistent.

---

## 2. TOKENIZER (K-line Tokenization / BSQ)

### 2.1 Structure
> "This is achieved using a Transformer-based autoencoder (Figure 3) composed of an encoder E_enc, a quantizer Q, and a decoder E_dec. Drawing inspiration from video quantization methods in generative modeling (Van Den Oord, Vinyals et al. 2017; Yu et al. 2023), we adapt Binary Spherical Quantization (BSQ) (Zhao, Xiong, and Kr¨ahenb¨uhl 2024), a variant of Look-up Free Quantization (LFQ) (Yu et al. 2023), for this task."

### 2.2 Quantization details
- **BSQ definition:** *"BSQ quantizes a continuous latent vector ξ_t into a k-bit binary code b_t ∈ {−1, 1}^k by projecting it onto a set of learnable hyperplanes."*
- **k:** *"While a large number of bits k (e.g., k = 20) is desirable for capturing rich financial patterns, it results in an exponentially large vocabulary of size 2^k..."* → **k = 20 bits, nominal vocabulary 2^20 = 1,048,576.**
- **Hierarchy / factorization:** *"we follow recent work in video quantization and generation (Yu et al. 2023; Wang et al. 2025) and factorize the k-bit code into n subspaces. Motivated by the trade-off between parameter savings and latency costs detailed in Appendix H (Q3), we set n = 2. We partition the code into a coarse subtoken b_t^c and a fine subtoken b_t^f of equal bit length, k_c = k_f = k/2, where k = k_c + k_f. The resulting code b_t is a concatenation of these two subtokens: b_t = [b_t^c, b_t^f], with b_t^c, b_t^f ∈ {−1, 1}^{k/2}."* → **two 10-bit subtokens, sub-vocabulary 2^10 = 1024 each.**
- **Why:** *"This decomposition transforms a single prediction over a large vocabulary of size 2^k into two sequential predictions over 2^{k/2} entries, substantially reducing both computational and parameter complexity."*

### 2.3 Tokenizer loss (BSQ loss)
> "L_tokenizer = L_coarse + L_fine + λ L_quant,   (2)"
> "• L_coarse = E[‖x − E_dec(b^c)‖²], which trains the coarse subtoken b^c to form a low-fidelity reconstruction."
> "• L_fine = E[‖x − E_dec(b)‖²], which evaluates the high-fidelity reconstruction using the complete token b."
> "• L_quant is the quantization loss from BSQ (Zhao, Xiong, and Kr¨ahenb¨uhl 2024) that regularizes the learning process. It penalizes the L2 distance between continuous latent vectors ξ and their binary codes b, aligning the encoder's outputs with the learned codebook to ensure stable training."
> "The balancing hyperparameter λ for the quantization loss in our objective is set to 1."

**BSQ hyperparameters (Appendix C, verbatim):**
> "Following the official open-source implementation of BSQ, we configure the key quantization hyperparameters as follows: a commitment weight β = 0.05, entropy penalty weights γ_0 = 1.0 and γ = 1.1, and an overall entropy scale ζ = 0.05. The balancing hyperparameter λ for the quantization loss in our objective is set to 1. The quantization group size is set to 5 for tractable entropy computation."
*Footnote source:* `https://github.com/zhaoyue-zephyrus/bsq-vit`

### 2.4 Tokenizer architecture (Appendix C)
> "The tokenizer's autoencoder is designed to be lightweight. The encoder and decoder each consist of 3 Transformer layers, with a model dimension of 256, a feed-forward network dimension of 512, and 4 attention heads."

### 2.5 Input / output variable names
- **Input:** OHLCVA, D = 6. *"we fix the dimension D = 6 to represent OHLCVA attributes (Open, High, Low, Close prices, trading Volume, and Amount)."*
- **Output/reconstruction:** *"The plots show that the reconstructed 'Close Price' and 'Volume' series closely track the ground truth..."* (Figure 10 caption). Full 6-dim OHLCVA reconstruction implied by L_coarse/L_fine over x.
- **Crypto/forex ablation on inputs:** *"For cryptocurrency and forex assets, we intentionally exclude volume and amount fields, providing only the OHLC price series."*

### 2.6 Normalization / clipping at tokenizer input
z-score per feature dimension + clip to [−5, 5] (same as §1.4; this is the tokenizer input preprocessing).

### 2.7 Codebook utilisation and geometry (Appendix H)
**Table 11:**
| Codebook Type | Size | Usage |
|---|---|---|
| Coarse-Level-Subtoken Codebook | 2^10 | 97.66% |
| Fine-Level-Subtoken Codebook | 2^10 | 85.25% |
> "the code usage of BSQ reaches 97.66% at the coarse level and 85.25% at the fine level."

> "BSQ's projection of embeddings onto a unit sphere prior to binarization guarantees that the expected distortion is strictly upper-bounded (Zhao, Xiong, and Kr¨ahenb¨uhl 2024): E_u‖u − b_u‖ < sqrt(2 − 2/sqrt(L)) < sqrt(2). This bound tightens as the codebook dimension L increases."

### 2.8 Subtoken factorization trade-off (Appendix H Q3, Table 12)
| Setup | Splits (n) | Sub-Vocab (2^{k/n}) | Core Params (M) | Vocab Params (M) | Fusion Params (M) | Total Params (M) | Inference Steps per Token |
|---|---|---|---|---|---|---|---|
| No Split | 1 | 1,048,576 | 97.5 | 1744.8 | 0.0 | 1842.3 | 1× |
| **Ours** | **2** | **1,024** | **97.5** | **3.4** | **1.4** | **102.3** | **2×** |
| More Splits | 4 | 32 | 97.5 | 0.2 | 2.8 | 100.5 | 4× |
| | 5 | 16 | 97.5 | 0.1 | 3.5 | 101.1 | 5× |
Caption: *"Trade-off analysis for factorizing a k = 20 bit token into n subtokens, based on the Kronos_base architecture. The model's core Transformer blocks have ≈97.5M parameters."*

---

## 3. TRANSFORMER ARCHITECTURE (autoregressive pre-training)

### 3.1 Autoregressive factorization
> "p(b) = ∏_{t=1}^{T} p(b_t | b_<t),   (3)"
> "p(b_t | b_<t) = p(b_t^c | b_<t) · p(b_t^f | b_<t, b_t^c).   (4)"
> "This formulation allows the model to first predict the coarse-grained subtoken, which serves as a scaffold for subsequently generating the fine-grained residual subtoken."

Fused input vector:
> "v_i = W_fuse([e^c(b_i^c); e^f(b_i^f)]),   (5)" — *"at time i, the subtokens b_i^c and b_i^f are independently projected into vector representations using two distinct embedding layers"*.

**Coarse prediction:** *"p(b_t^c | b_<t) = softmax(W_c h_t)   (6)"*
**Fine prediction:** *"h_t^update = CrossAttn(q = e^c(b̂_t^c), k = v = h_t); p(b_t^f | b_<t, b_t^c) = softmax(W_f h_t^update)   (7)"*
- **Sampling not teacher forcing for the coarse subtoken during training:** *"we use the model's own prediction from the previous step, b̂_t^c, which is sampled from the predicted distribution p(b_t^c | b_<t), rather than using the ground-truth subtoken (i.e., teacher-forcing). We find that this sampling strategy enhances model robustness by mitigating exposure bias."*

**Pre-training objective:**
> "L_ar = −E_{b∼D} ∑_{t=1}^{T} [ log p(b_t^c | b_<t) + log p(b_t^f | b_<t, b_t^c) ]   (8)"

### 3.2 Released model sizes — Table 1 (verbatim)
| Model | Layers | d_model | d_ff | Heads | Vocab. (2^k) | Params |
|---|---|---|---|---|---|---|
| Kronos_small | 8 | 512 | 1024 | 8 | 20 | 24.7M |
| Kronos_base | 12 | 832 | 2048 | 16 | 20 | 102.3M |
| Kronos_large | 18 | 1664 | 3072 | 32 | 20 | 499.2M |

Caption: *"Model configurations for the Kronos family. We detail the number of Transformer layers, model dimension (d_model), feed-forward dimension (d_ff), number of attention heads, vocabulary size, and the total number of parameters."* (The header "Vocab. (2^k)" prints value 20 = the bit count k, i.e. vocab size 2^20; Table 12 confirms 1,048,576.)

**IMPORTANT:** The paper releases **three** sizes — small / base / large. **There is NO "mini" model in the paper text.** (Upstream repo may add Kronos-mini; the paper does not describe one.)

### 3.3 Attention / positional scheme (Appendix C)
> "we employ causal self-attention with Rotary Position Embeddings (RoPE) (Su et al. 2024), which injects relative positional information."
> "Attention(Q, K, V) = CausalMask(Q'(K')^T / sqrt(d_k)) V   (9)" — *"where d_k is the dimension of the key vectors, and CausalMask prevents attending to future positions."*
> "we adopt the Pre-Layer Normalization (Pre-LN) (Xiong et al. 2020) to improve training stability, specifically utilizing Root Mean Square Layer Normalization (RMSNorm) (Zhang and Sennrich 2019)."

### 3.4 Temporal embeddings (Appendix C)
> "We extract five time-related features for each K-line entry: minute-of-day, hour-of-day, day-of-week, day-of-month, and month-of-year. Each feature is mapped to a dense vector via a dedicated embedding layer. These temporal embeddings are summed and then added to the input representation of each corresponding token."

### 3.5 Context length
> "Considering resource constraints and practical deployment scenarios, we limit the maximum context length to 512 tokens. Nevertheless, this design remains fully compatible with arbitrary forecasting horizons by leveraging K-line data at varying frequencies; for instance, using 1-minute data for short-term forecasting and daily data for weekly or monthly predictions."

### 3.6 Training hyperparameters — Table 5 (verbatim)
| Model | FFN Dropout | Residual Dropout | Attention Dropout | Token Dropout | Learning Rate | Weight Decay |
|---|---|---|---|---|---|---|
| Kronos_small | 0.25 | 0.25 | 0.1 | 0.1 | 1 × 10⁻³ | 0.01 |
| Kronos_base | 0.20 | 0.20 | 0.0 | 0.0 | 5 × 10⁻⁴ | 0.05 |
| Kronos_large | 0.00 | 0.00 | 0.0 | 0.0 | 2 × 10⁻⁴ | 0.10 |

Caption: *"Hyperparameter configurations for the Kronos model series. All models are trained with the AdamW optimizer."*
> "As model scale increases, we decrease the peak learning rate and dropout probability while increasing the weight decay. We employ the AdamW optimizer (Loshchilov and Hutter 2017) and a cosine learning rate schedule with a linear warm-up phase. The learning rate warms up from 10% of its peak value over the first 15,000 training steps."

### 3.7 Training compute (Appendix D)
> "All experiments are conducted within a Kubernetes (k8s) cluster. For all computational tasks, we utilize three identical pods. Each pod is provisioned with a dedicated set of resources comprising 96 CPU cores (Intel Xeon Gold 6330 @ 2.00 GHz), 200 GB of system memory (RAM), and eight NVIDIA GeForce RTX 4090D GPUs. This configuration provides a total of 24 GPUs, which are collectively employed for model training and all subsequent evaluations."

- **GPUs: 24 × NVIDIA RTX 4090D total (3 pods × 8).**
- **Training hours / wall-clock / total compute: NOT STATED IN PAPER.**
- **Pre-training batch size: NOT STATED IN PAPER.** (The only batch size of 256 given is for the *baseline* models, §below.)
- **Total pre-training steps / tokens seen: NOT STATED IN PAPER** (only warm-up = 15,000 steps).

### 3.8 Software environment (Appendix D)
> "Operating System: Ubuntu 24.04.1 LTS"
> "Software versions: Python 3.13.2, PyTorch 2.7.0, NumPy 1.26.2, Pandas 2.2.2, Matplotlib 3.9.3, Hugging Face Hub ('huggingface hub') 1.57.4"

---

## 4. INFERENCE

### 4.1 Sampling procedure
> "At inference time, we generate future token sequences autoregressively, analogous to text generation. The stochasticity of this process is controlled via standard techniques like temperature scaling and top-p (nucleus) sampling (Holtzman et al. 2019). The probability of sampling token i from logits z is given by p_i ∝ exp(z_i/T), where T is the temperature."
> "For tasks requiring high precision, prediction accuracy can be enhanced by generating multiple future trajectories (i.e., Monte Carlo rollouts) and averaging the decoded continuous values to produce a more stable forecast. As demonstrated in our experiments, this approach consistently improves forecast quality."

### 4.2 Inference sample ensembling (Test-Time Scaling)
> "By leveraging stochastic sampling, Kronos can generate multiple distinct future trajectories from the same context. We investigate the effect of ensembling these predictions by averaging the outcomes from an increasing number of sampled paths. Figure 7 presents the performance on forecasting tasks as a function of the number of samples... Averaging across multiple paths mitigates the stochasticity inherent in the generation process and reduces prediction variance."
Figure 7 x-axis: *"Number of Inference Samples (N, log scale)"* with ticks **1, 5, 10, 20**; *"The lines represent the mean performance over 5 runs with different random seeds, while the shaded areas indicate the standard deviation."* Annotated baseline floors in Fig 7: Price Series — *"Best Baseline (IC): 0.0317, Best Baseline (RankIC): 0.0138"*; Return — *"Best Baseline (IC): 0.0495, Best Baseline (RankIC): 0.0533"*.

### 4.3 Per-task inference hyperparameters — Table 6 (verbatim)
| Task | Temperature (T) | Top-p | Number of Inference Samples (N) |
|---|---|---|---|
| Price Series Forecasting | 0.6 | 0.90 | 10 |
| Return Forecasting | 0.6 | 0.90 | 10 |
| Realized Volatility Forecasting | 0.9 | 0.90 | 1 |
| Synthetic K-line Generation | 1.0 | 0.95 | 1 |
| Investment Simulation | 0.6 | 0.90 | 10 |

Caption: *"Inference hyperparameters for downstream tasks. T denotes the temperature for sampling, Top-p controls nucleus sampling, and N is the number of inference samples generated for each test instance."*
> "For forecasting tasks (price series and return), which demand precision, lower temperatures (e.g., T ≈ 0.6) are preferable... Conversely, realized volatility forecasting and synthetic K-line generation benefit from greater stochasticity, achieving optimal performance at temperatures closer to 1.0."
> "forecasting tasks favor smaller p values to restrict the sampling pool, whereas generative tasks perform better with a larger nucleus (p ≥ 0.9)... temperature scaling generally offers more effective and nuanced control, leading to slightly better peak performance across tasks."

### 4.4 Context vs prediction length (recommended)
Max context = 512 tokens (see §3.5). Look-back / horizon settings per frequency — **Table 8 (verbatim):**
| Frequency | Look-back Window | Forecast Horizon |
|---|---|---|
| 5min | 480 | 96 |
| 10min | 240 | 48 |
| 15min | 160 | 32 |
| 20min | 120 | 24 |
| 40min | 90 | 24 |
| 1-hour | 80 | 12 |
| 2-hour | 60 | 12 |
| 4-hour | 90 | 18 |
| Daily | 40 | 12 |

Caption: *"Look-back and forecast horizon settings for each K-line frequency in the forecasting tasks."*
Generation settings: *"For the 15-minute frequency, we use a look-back window of 120 and generate a future sequence of length 96. For the daily frequency, the look-back is 96 and the generation horizon is 35."*

---

## 5. FORECAST TARGET DEFINITION & METRICS

*(Note: the task brief's "KronosPredictor" name does not appear in the paper text; the paper describes the same procedure generically — autoregressive sampling with temperature/top-p and multi-sample averaging. "max_context" is not a literal string in the paper; max context = 512.)*

### 5.1 Prediction task
> "Given a historical sequence x_1:T = (x_1, x_2, ..., x_T), our objective is to predict the following H observations x̂_{T+1:T+H} = (x̂_{T+1}, x̂_{T+2}, ..., x̂_{T+H})."
> "The forecasting task then reduces to an autoregressive token-sequence modeling problem: p(b_{T+1:T+H} | b_{1:T}) = ∏_{h=1}^{H} p(b_{T+h} | b_{1:T+h−1}).   (1)"

### 5.2 IC / RankIC (Price Series Forecasting)
> "**Price Series Forecasting:** We assess the model's ability to predict future price series. Performance is measured by the Information Coefficient (IC) and Rank Information Coefficient (RankIC) between the predicted and actual values."
> "**Metric Calculation Details — Price Series Forecasting:** For each sample, the IC and RankIC are calculated between the predicted and true series for each of the four price channels (Open, High, Low, Close). The final reported metrics are the average across these four channels."

### 5.3 Return Forecasting
> "We define the predicted return r̂ based on the last value of the predicted close price sequence p̂_{t+H} and the last value of the historical close price sequence p_t: r̂ = p̂_{t+H}/p_t − 1.   (11)"
> "The IC and RankIC are then computed between the vector of predicted returns and the vector of actual returns for all samples within a given asset class and frequency."

### 5.4 Realized Volatility Forecasting
> "Using the model's predicted closing prices {p̂_i}_{i=1}^{H} over the forecast horizon, the realized volatility is calculated as the sum of squared log returns: σ̂² = ∑_{i=1}^{H−1} (log(p̂_{i+1}) − log(p̂_i))².   (12)"
> "We then compute the Mean Absolute Error (MAE) and Coefficient of Determination (R²) between the predicted and actual realized volatilities across all samples."

### 5.5 Investment metrics
> "**Investment Simulation:** ... we perform backtesting simulations. The performance is reported using Annualized Excess Return (AER) and Information Ratio (IR)."

### 5.6 Baseline loss function (used for full-shot baselines for fairness) — Eq. 10
> "L = (1/(M·H)) ∑_{i=1}^{M} ∑_{j=1}^{H} (y_{i,j} − ŷ_{i,j})² − λ · (1/M) ∑_{i=1}^{M} IC(y_i, ŷ_i)   (10)" with λ = 4 and *"All models are trained with a batch size of 256 and an Adam optimizer with a learning rate of 5 × 10⁻⁴... a maximum of 12 epochs, employing an early stopping mechanism with a patience of 3 epochs based on the validation loss."*

---

## 6. DOWNSTREAM EXPERIMENTS

Five representative tasks: price series forecasting, return forecasting, realized volatility forecasting, synthetic K-line generation, investment simulation.

### 6.1 Price Series Forecasting
- Datasets: 9 stock exchanges — in-distribution: Shanghai (XSHG), NASDAQ (XNAS), Japan (XJPX), India (XNSE), Korea (XKRX), Hong Kong (XHKG); out-of-distribution: Indonesia (XIDX), Malaysia (XKLS), Taiwan (XTAI). Plus Cryptocurrency (all Binance spot pairs) and Forex (>1,000 pairs).
- *"To test both in-distribution and out-of-distribution generalization, we use data from nine global stock exchanges."*
- Metrics: IC, RankIC (averaged over the four OHLC price channels).
- Headline: *"Kronos boosts price series forecasting RankIC by 93% over the leading TSFM and 87% over the best non-pre-trained baseline."*

### 6.2 Return Forecasting
- Same universes; metric IC/RankIC on per-asset, per-frequency return vectors.

### 6.3 Realized Volatility Forecasting
- Baselines add econometric **ARCH (Engle 1982)** and **GARCH (Bollerslev 1986)**. *"For each time series, we fit ARCH models with lag orders p ∈ {1, 2, 3}. The model with the lowest Bayesian Information Criterion (BIC) is selected..."*; *"GARCH: We perform a grid search over the lag orders for both the autoregressive term (p) and the moving average term (q), with p, q ∈ {1, 2, 3}. Similar to ARCH, the GARCH(p,q) model with the minimum BIC is chosen..."*
- Metric MAE, R².

### 6.4 Synthetic K-line Generation
- *"We use data from two stock exchanges (in-distribution XSHG and out-of-distribution XTAI), as well as the cryptocurrency and forex datasets. We evaluate generation on two frequencies: 15-minute and daily."*
- *"For each asset-frequency pair, we generate 6,000 synthetic sequences for evaluation."*
- **Discriminative Score:** *"we employ a post-hoc LSTM-based classifier... a single LSTM layer with a hidden dimension of 32. For training, we construct a balanced dataset of 6,000 samples (3,000 real, 3,000 synthetic) and a held-out test set of the same size and composition. The model is trained for 20 epochs with a batch size of 64, using the Adam optimizer (learning rate = 0.0005) and the binary cross-entropy (BCE) loss function. The Discriminative Score is defined as the classification error on the test set. A score approaching 0.5 indicates higher fidelity..."*
- **Usefulness (TSTR):** *"we adopt the Train-on-Synthetic, Test-on-Real (TSTR) methodology. We train a post-hoc LSTM prediction model... two LSTM layers with a hidden dimension of 64. It is trained exclusively on 6,000 generated synthetic sequences for 20 epochs using the Adam optimizer (learning rate = 0.001) and a batch size of 64, with the Mean Squared Error (MSE) loss... The look-back and horizon windows are set to (80, 16) for 15-minute data and (30, 5) for daily data... The final usefulness score is reported as the average Information Coefficient (IC) and Rank Information Coefficient (RankIC) of the predicted price series."*
- Diversity: t-SNE + KDE visual comparisons (Figures 5, 13, 14).
- Headline: *"a 22% improvement in generative fidelity for synthetic K-line sequences."*

### 6.5 Investment Simulation (BACKTEST) — full details
> "we conduct an investment simulation on the Chinese A-share market. For simplicity, regarding the Zero-shot Time Series Models, we only select the largest-sized model from each family for comparison."
- **Data / source:** *"Our empirical analysis utilizes daily market data for the Chinese A-share market, sourced from the Qlib platform (Yang et al. 2020), an open-source framework for quantitative finance. To promote transparency and reproducibility, we apply no additional filtering or preprocessing to the data, using it in its original, unprocessed state. Furthermore, we conduct all backtesting simulations within the Qlib framework."*
- **Universe:** *"on the constituents of the CSI 300 and CSI 800 indices. These indices are chosen as they represent two key segments of the Chinese A-share market: the CSI 300 comprises large-cap, highly liquid stocks, while the CSI 800 provides broader market coverage..."*
- **Strategy:** *"We employ the top-k/drop-n portfolio construction strategy. On each trading day, all stocks in the investment universe are ranked based on their predicted return signal. An equal-weight portfolio is formed by taking long positions in the top k stocks. To manage turnover and trading costs, a maximum of n stocks are bought or sold daily, and a minimum holding period of 5 days is enforced for all positions."*
- **Signal:** *"For any given stock on trading day t, a sequence of forecasted closing prices for the subsequent H days... is first generated by the respective model. The signal, which we term the H-day average expected return (R_{t→t+H}), is then calculated by comparing the arithmetic mean of these forecasted prices..."*
- **Parameters / costs:** *"For the CSI 300 index, we set k = 50 and n = 5. For the broader CSI 800 index, we set k = 200 and n = 10... a conservative transaction cost of 0.15% is applied to each trade."*
- **Horizon / look-back:** *"we set the forecast horizon to H = 10. All price forecasts are generated using daily K-line data with a 90-day look-back window. This methodology is designed to produce a robust signal by averaging the forecasted price path..."* (Note: H=10 but min holding = 5 days — slight tension, see OPEN QUESTIONS.)
- **Period:** backtest curves span **2024-07 to 2025-06** (Figure 9 x-axis ticks: 2024-07, 2024-08, 2024-10, 2024-12, 2025-02, 2025-04, 2025-06).
- **Reported results (Table 10, verbatim):**

| Model | CSI300 AER | CSI300 IR | CSI800 AER | CSI800 IR | Average AER | Average IR |
|---|---|---|---|---|---|---|
| TimeXer | 0.1035 | 0.7988 | 0.1509 | 1.5471 | 0.1272 | 1.1730 |
| TimeMixer | −0.0600 | −0.5721 | 0.0705 | 0.8113 | 0.0053 | 0.1196 |
| iTransformer | −0.1202 | −1.4441 | −0.0525 | −0.8558 | −0.0864 | −1.1500 |
| PatchTST | 0.1289 | 0.9895 | 0.1620 | 1.5033 | 0.1455 | 1.2464 |
| TimesNet | 0.1441 | 0.6558 | 0.0634 | 0.7225 | 0.1038 | 0.6892 |
| DLinear | −0.0066 | −0.0605 | 0.1112 | 1.2003 | 0.0523 | 0.5699 |
| FEDformer | 0.0362 | 0.2943 | 0.0539 | 0.5602 | 0.0451 | 0.4273 |
| NSTransformer | −0.0343 | −0.2889 | 0.0664 | 0.6979 | 0.0161 | 0.2045 |
| Time-MOE_base | 0.0985 | 0.8230 | 0.1315 | 1.3726 | 0.1150 | 1.0978 |
| Moirai_large | 0.1470 | 0.9747 | 0.1683 | 1.5215 | 0.1577 | 1.2481 |
| TimesFM | 0.0788 | 0.7357 | 0.1355 | 1.6427 | 0.1072 | 1.1892 |
| Moment_large | 0.1655 | 1.1993 | 0.1707 | 1.5361 | 0.1681 | 1.3677 |
| Chronos_large | −0.0659 | −0.7670 | 0.0056 | 0.0902 | −0.0302 | −0.3384 |
| **Kronos_small** | **0.1805** | **1.2394** | **0.1772** | **1.6050** | **0.1789** | **1.4222** |
| **Kronos_base** | **0.1911** | **1.3782** | **0.1867** | **1.6652** | **0.1889** | **1.5217** |
| **Kronos_large** | **0.2193** | **1.4177** | **0.1974** | **1.8805** | **0.2084** | **1.6491** |

→ **Kronos_large best: CSI300 AER 0.2193 / IR 1.4177; CSI800 AER 0.1974 / IR 1.8805; average AER 0.2084 / IR 1.6491.** No Sharpe ratio reported anywhere (see §12 / OPEN QUESTIONS).

### 6.6 Figure 1 headline radar values (verbatim labels + numbers)
```
Our Models:            Kronos_large | Kronos_base | Kronos_small
Price Forecasting (IC)         0.044 | 0.021 | 0.0267  (vs baselines 0.009, 0.301? — table jumbled)
Return Forecasting (RankIC)    0.0675 | 0.033 | 0.037
Volatility Forecasting (MAE)   0.066 | 0.262 | 0.120
Kline Generation (RankIC)      0.040 | 0.0301 | 0.001
Kline Generation (IC)          0.0702 | 0.0282 | 0.040
Investment Simulation (AER)/(IR)  0.208 | 0.066 | 1.65 | 0.964
```
*The Figure 1 label/value stream is jumbled by PDF text extraction — treat as indicative only.* These are figure annotations, not a clean table.

### 6.7 Baseline suite (25 models, 4 groups, verbatim)
> "• Full-shot Time Series Models: ... TimeXer (Wang et al. 2024c), TimesNet (Wu et al. 2022), TimeMixer (Wang et al. 2024a), PatchTST (Nie et al. 2022), Non-stationary Transformer (NSTransformer) (Liu et al. 2022), DLinear (Zeng et al. 2023), FEDformer (Zhou et al. 2022), and iTransformer (Liu et al. 2023)."
> "• Zero-shot Time Series Models: ... TimeMOE (Xiaoming et al. 2025), Moirai (Woo et al. 2024), TimesFM (Das et al. 2024), Moment (Goswami et al. 2024), and Chronos (Ansari et al. 2024), which we evaluate in a zero-shot setting."
> "• Econometric Volatility Models: ... ARCH (Engle 1982) and GARCH (Bollerslev 1986)."
> "• Generative Time Series Models: ... DiffusionTS (diffusion-based) (Yuan and Qiao 2024), TimeVAE (VAE-based) (Desai et al. 2021), and TimeGAN (GAN-based) (Yoon, Jarrett, and Van der Schaar 2019)."

Baseline hyperparameters — **Table 7 (verbatim):**
| Model | Layers | d_model | d_ff | Heads |
|---|---|---|---|---|
| TimeXer | 3 / 5 | 128 / 256 | 256 / 512 | 4 / 8 |
| TimesNet | 3 / 5 | 128 / 256 | 256 / 512 | — |
| TimeMixer | 3 / 5 | 128 / 256 | 256 / 512 | 4 / 8 |
| PatchTST | 3 / 5 | 128 / 256 | 256 / 512 | 4 / 8 |
| NSTransformer | 2 / 3 | 128 / 256 | 256 / 512 | 4 / 8 |
| FEDformer | 2 / 3 | 128 / 256 | 256 / 512 | 4 / 8 |
| iTransformer | 3 / 5 | 128 / 256 | 256 / 512 | 4 / 8 |
Caption: *"Hyperparameter configurations for the baseline models. Values for the two evaluated sets are separated by a slash (/)."*
> "For DLinear, instead of varying model dimensions, we evaluate two configurations based on its 'individual' parameter: one where a single linear layer is shared across all variates ('individual=False') and another where a separate linear layer is trained for each variate ('individual=True')."
> "All models are trained with a batch size of 256 and an Adam optimizer with a learning rate of 5 × 10⁻⁴... a maximum of 12 epochs, employing an early stopping mechanism with a patience of 3 epochs based on the validation loss."

### 6.8 Table 3 — financial-data share of TSFM pre-training corpora (verbatim)
| Model | Architecture | Tokenization | Probabilistic | Financial Data Ratio (Est.) | Primary Domain |
|---|---|---|---|---|---|
| Kronos (Ours) | Decoder-only | Discrete (BSQ) | Yes | 100% | Financial K-lines |
| Sundial | Decoder-only | Continuous | Yes | 1.02% | General |
| Time-MoE | Decoder-only | Continuous | No | <0.01% | General |
| Moirai | Encoder-only | Continuous | Yes | 0.10% | General |
| MOMENT | Encoder-only | Continuous | No | 1.60% | General |
| Chronos | Encoder-Decoder | Discrete (Quantization) | Yes | 0.45% | General |
| Timer | Decoder-only | Continuous | No | 0.03% | General |
| TimesFM | Decoder-only | Continuous | No | <0.01% | General |
| UniTS | Encoder-only | Continuous | No | Unknown | General |
| Lag-Llama | Decoder-only | Continuous | Yes | 0.01% | General |

---

## 7. DATASETS, SPLITS, BOUNDARIES

- **Pre-training end:** *"The pre-training data for Kronos extends up to June 2024."*
- **Evaluation test period:** *"Consequently, our test period for all tasks begins in July 2024 to ensure a strict temporal separation between training and evaluation."*
- **Explicit train/val/test boundaries beyond "test begins July 2024": NOT STATED IN PAPER.** No validation split construction for pre-training, no per-task train/val windows, and no look-back/cut-off for the backtest other than the visible 2024-07 → 2025-06 curve range.
- **Forecasting datasets (Table 8 frequencies):** 5min, 10min, 15min, 20min, 40min, 1-hour, 2-hour, 4-hour, Daily.
- **Backtest datasets:** CSI 300, CSI 800 (via Qlib).
- **Generation datasets:** XSHG, XTAI, Crypto (Binance), Forex — frequencies 15min and daily.
- Exchange codes: in-dist XSHG/XNAS/XJPX/XNSE/XKRX/XHKG; OOD XIDX/XKLS/XTAI.

---

## 8. FINE-TUNING SCRIPT DETAILS

**NOT STATED IN PAPER.** The paper contains **no fine-tuning section, no fine-tuning script, and no fine-tuning hyperparameters** (no finetuned lookback / prediction length / context / LR / batch size). The string "fine-tun" does not occur in the text at all. All Kronos results are reported **zero-shot** (*"Kronos excels in a zero-shot setting"*; zero-shot TSFM baselines are explicitly grouped as "Zero-shot Time Series Models").

The nearest thing to "training on downstream data" is the **TSTR** protocol, which trains a *post-hoc LSTM* (look-back/horizon (80,16) for 15-min, (30,5) for daily; 6,000 synthetic sequences; 20 epochs; Adam LR 0.001; batch 64; MSE) — this is not fine-tuning Kronos.

---

## 9. HEADLINE NUMERIC RESULTS TABLES

### 9.1 Table 14 — Price Series Forecasting (Part 1): Kronos vs full-shot models (IC / RankIC)
Metrics columns: KronosS (Kronos_small), KronosB (Kronos_base), KronosL (Kronos_large), TimeXer, TimeMixer, iTransformer, PatchTST, TimesNet, DLinear, FEDformer, NSTransformer.

Rows (verbatim where readable):
- **XSHG** IC: 0.0549 / 0.0564 / 0.0546 | 0.0280 0.0291 0.0350 0.0450 0.0424 0.0405 0.0233 0.0433
- **XSHG** RankIC: 0.0375 / 0.0390 / 0.0381 | 0.0053 0.0079 0.0128 0.0088 0.0175 0.0181 0.0107 0.0155
- (XNAS onward partially garbled by table extraction.)
- **Average IC**: KronosS/base/large lead. **Average RankIC: 0.0254 / 0.0258 / 0.0267** | −0.0007 0.0055 0.0030 0.0087 0.0143 0.0094 0.0008 0.0082
- **1st Count: 4 / 7 / 10** (Kronos_small 4, base 7, large 10) | 0 0 0 1 0 0 0 0

### 9.2 Table 15 — Price Series Forecasting (Part 2): zero-shot models
Average IC: Time-MOES 0.0168, Time-MOEB 0.0174, MoiraiS −0.0004, MoiraiB −0.0006, MoiraiL −0.0003, TimesFM 0.0052, MomentS 0.0025, MomentB −0.0050, MomentL −0.0029, ChronosS 0.0037, ChronosB 0.0021, ChronosL 0.0060.
Average RankIC: 0.0133 / 0.0138 / 0.0000 / 0.0009 / 0.0005 / 0.0021 / −0.0001 / −0.0033 / −0.0014 / 0.0057 / 0.0047 / 0.0078. **1st Count: 0 for all.**

### 9.3 Table 16 — Return Forecasting (Part 1): Kronos vs full-shot
- **XSHG** IC: KronosS 0.0677, KronosB 0.0652, KronosL 0.0662 | TimeXer 0.0456, TimeMixer 0.0114, iTransformer 0.0371, PatchTST 0.0467, TimesNet 0.0563, DLinear 0.0626, FEDformer 0.0589, **NSTransformer 0.0777**
- **XSHG** RankIC: 0.0617 / 0.0653 / 0.0642 | 0.0306 −0.0072 0.0266 0.0437 0.0421 0.0461 0.0568 0.0595
- **XNAS** IC: 0.0563 / 0.0626 / 0.0639 | ...
- **Average RankIC: 0.0622 / 0.0634 / 0.0675** | 0.0183 0.0063 0.0225 0.0392 0.0389 0.0423 0.0292 0.0421
- **1st Count: 2 / 3 / 10** | 1 0 0 0 2 1 0 1

### 9.4 Table 17 — Return Forecasting (Part 2): zero-shot
- XSHG IC: 0.0507 0.0501 0.0507 0.0579 0.0534 0.0322 0.0575 0.0579 0.0575 −0.0152 −0.0055 −0.0019
- XSHG RankIC: 0.0612 0.0621 0.0657 0.0647 0.0661 0.0445 0.0527 0.0530 0.0525 −0.0277 −0.0116 −0.0048
- Crypto IC: 0.0291 0.0293 −0.0051 −0.0081 −0.0046 −0.0042 −0.0042 −0.0039 −0.0043 0.0041 0.0067 0.0107
- Forex IC: 0.0334 0.0336 0.0355 0.0357 0.0347 0.0353 0.0155 0.0157 0.0157 0.0289 0.0255 0.0274
- Forex RankIC: 0.0217 0.0215 0.0262 0.0264 0.0264 0.0276 0.0162 0.0164 0.0164 0.0194 0.0218 0.0184
- Average IC: 0.0495 0.0487 0.0407 0.0404 0.0405 0.0325 0.0410 0.0410 0.0409 0.0337 0.0343 0.0357

### 9.5 Table 18 — Realized Volatility Forecasting (Part 1): Kronos vs full-shot + ARCH/GARCH
- **XSHG** MAE: 0.0199 / 0.0205 / 0.0203 | TimeXer 0.0510 TimeMixer 0.0349 iTransformer 0.0593 PatchTST 0.0356 TimesNet 0.0348 DLinear 0.0398 FEDformer 0.0231 NSTransformer 0.0348 **ARCH 0.0247 GARCH 0.0219**
- **XSHG** R²: 0.2597 / 0.2630 / 0.2809 | 0.1500 0.1585 0.2191 0.2401 0.1429 0.2400 0.2301 0.1232 0.1969 0.1986
- **XNAS** MAE: 0.1540 / 0.1407 / 0.1503 | 0.3323 0.3473 0.3223 0.2926 0.2492 0.2416 0.2223 0.2168 **ARCH 0.1472 GARCH 0.1259**
- **XJPX** MAE: 0.0198 / 0.0198 / 0.0196 | 0.1309 0.1324 0.0425 0.0842 0.0365 0.1527 0.0316 0.0353 ARCH 0.0320 GARCH 0.0271
- **Average R²: 0.2490 / 0.2470 / 0.2624** | 0.1295 0.0694 0.1366 0.1066 0.1199 0.0913 0.0731 0.1047 0.2281(ARCH) 0.2323(GARCH)
- **1st Count: 4 / 2 / 11** | 0 0 0 0 0 0 0 0 0 1 3

### 9.6 Table 19 — Realized Volatility Forecasting (Part 2): zero-shot
- XSHG MAE: 0.0462 0.0471 0.1158 0.0994 0.1048 0.0408 0.0357 0.0343 0.0366 0.0386 0.0384 0.0382
- XSHG R²: 0.2423 0.2417 0.2118 0.2233 0.2191 0.0995 0.2479 0.2461 0.2336 0.1946 0.1922 0.1663
- XNAS MAE: 0.2713 0.2498 0.3537 0.1927 0.2502 0.1902 0.1034 0.1020 0.1168 0.1896 0.1863 0.1881
- XJPX MAE: 0.0372 0.0367 0.1065 0.0829 0.0878 0.0345 0.0291 0.0278 0.0306 0.0331 0.0331 0.0329
- **Average R²: 0.1487 0.1375 0.1285 0.1427 0.1492 0.0887 0.1380 0.1468 0.1339 0.1490 0.1475 0.1458** — **1st Count: 0 except MomentB = 1**

### 9.7 Table 20 — Discriminative Score (synthetic K-line generation)
Columns: Kronos_small / Kronos_base / Kronos_large / DiffusionTS / TimeVAE / TimeGAN.
| Set | KronosS | KronosB | KronosL | DiffusionTS | TimeVAE | TimeGAN |
|---|---|---|---|---|---|---|
| XSHG 15min | 0.2313 | 0.2317 | 0.2393 | 0.0885 | 0.0015 | 0.2241 |
| XSHG daily | 0.1865 | 0.2227 | 0.2105 | 0.2532 | 0.0142 | 0.1193 |
| XTAI 15min | 0.1733 | 0.1478 | 0.1788 | 0.1420 | 0.0387 | 0.2689 |
| XTAI daily | 0.2088 | 0.2023 | 0.2235 | 0.1712 | 0.0097 | 0.0622 |
| Crypto 15min | 0.4100 | 0.4185 | 0.4187 | 0.3005 | 0.0637 | 0.0680 |
| Crypto daily | 0.2792 | 0.2575 | 0.2835 | 0.3188 | 0.0402 | 0.2114 |
| Forex 15min | 0.4783 | 0.4903 | 0.4688 | 0.4112 | 0.0492 | 0.4015 |
| Forex daily | 0.3337 | 0.4363 | 0.4152 | 0.3177 | 0.0295 | 0.2387 |
| **Average** | **0.2876** | **0.3009** | **0.3048** | **0.2504** | **0.0308** | **0.1993** |
| **1st Count** | 0 | 2 | 4 | 2 | 0 | 1 |
Caption: *"A higher score indicates a better generation quality."*

### 9.8 Table 21 — Predictive Usefulness (IC / RankIC, TSTR) for synthetic generation
| Set | KronosS | KronosB | KronosL | DiffusionTS | TimeVAE | TimeGAN |
|---|---|---|---|---|---|---|
| XSHG 15min IC | 0.0223 | 0.0231 | 0.0236 | 0.0103 | 0.0098 | 0.0102 |
| XSHG 15min RankIC | 0.0144 | 0.0147 | 0.0151 | 0.0087 | 0.0134 | 0.0081 |
| XSHG daily IC | 0.0918 | 0.0902 | 0.0845 | 0.0760 | −0.0789 | 0.0108 |
| XSHG daily RankIC | 0.0854 | 0.0839 | 0.0796 | 0.0684 | −0.0720 | 0.0150 |
| XTAI 15min IC | 0.0230 | 0.0274 | 0.0281 | 0.0074 | −0.0118 | 0.0045 |
| XTAI 15min RankIC | 0.0226 | 0.0276 | 0.0299 | 0.0037 | −0.0092 | −0.0003 |
| XTAI daily IC | 0.0460 | 0.0437 | 0.0560 | 0.0013 | −0.0213 | 0.0118 |
| XTAI daily RankIC | 0.0445 | 0.0431 | 0.0551 | −0.0001 | −0.0193 | 0.0118 |
| Crypto 15min IC | 0.0237 | 0.0243 | 0.0237 | −0.0016 | −0.0012 | 0.0096 |
| Crypto 15min RankIC | 0.0222 | 0.0231 | 0.0231 | −0.0026 | −0.0016 | 0.0079 |
| Crypto daily IC | 0. (truncated in extraction — see OPEN QUESTIONS) | | | | | |
Caption: *"Higher IC and RankIC scores suggest the generated data is more useful for building predictive financial models."*

### 9.9 Summary of headline claims (verbatim)
- *"Kronos boosts price series forecasting RankIC by 93% over the leading TSFM and 87% over the best non-pre-trained baseline."*
- *"It also achieves a 9% lower MAE in volatility forecasting and a 22% improvement in generative fidelity for synthetic K-line sequences."*
- *"as the model size scales up, performance on these tasks consistently improves, empirically validating the scaling laws for time series foundation models (Yao et al. 2024)."*

---

## 10. ABLATIONS + ZERO-SHOT vs FINE-TUNED

### 10.1 Table 2 — Modeling-paradigm ablation (verbatim)
Columns: Prediction Space / Training Objective / Price IC / Price RankIC / Return IC / Return RankIC / Vol MAE / Vol R²
| Variant | Space | Objective | Price IC | Price RankIC | Return IC | Return RankIC | Vol MAE | Vol R² |
|---|---|---|---|---|---|---|---|---|
| Direct-AR | Continuous | MSE | 0.0212 | 0.0149 | 0.0416 | 0.0399 | 0.0565 | 0.1608 |
| Prob-AR | Continuous | NLL | 0.0179 | 0.0102 | 0.0356 | 0.0329 | 0.0464 | 0.1383 |
| Kronos-Parallel | Discrete | Cross-Entropy | 0.0345 | 0.0226 | 0.0529 | 0.0505 | 0.0461 | 0.1784 |
| **Kronos_small** | Discrete | Cross-Entropy | **0.0431** | **0.0254** | **0.0665** | **0.0622** | **0.0384** | **0.2490** |
Caption: *"Ablation study dissecting the architectural choices of Kronos... Direct-AR serves as a standard regression baseline. Prob-AR evaluates the benefit of probabilistic modeling in the continuous space. Kronos-Parallel ablates our sequential subtoken design by predicting subtokens concurrently. Best results are in bold."*
> "our discrete-space models markedly outperform these continuous alternatives. We also find that Kronos-Parallel... performs worse than our sequential approach, demonstrating the importance of modeling subtoken dependencies."

- Direct-AR: *"trained to directly predict the value of the next time step... minimize the Mean Squared Error (MSE)."*
- Prob-AR: *"mixture of four Student-t distributions... trained by minimizing the Negative Log-Likelihood (NLL)."*
- Kronos-Parallel: *"removes the intra-block module... a single prediction head is used to concurrently predict the logits for both subtokens of the next time step."*

### 10.2 Table 9 — Tokenizer architecture ablation (verbatim)
| Tokenizer Architecture | MAE (↓) | MSE (↓) |
|---|---|---|
| Transformer w/ Hierarchical Loss (Ours) | 0.0785 | 0.0203 |
| Transformer w/ Standard Loss | 0.0781 | 0.0202 |
| CNN-based | 0.0916 | 0.0251 |
> "All models are trained with a vocabulary size of 2^18." → *"our model with hierarchical loss achieves reconstruction quality nearly identical to that of the standard loss variant."*

### 10.3 Figure 6 — Vocabulary-size ablation (14–20 bits)
X-axis *"Vocabulary Size (2^k)"* 14→20. Plots: Reconstruction MAE & MSE; Price Series Forecasting IC & RankIC; Return Forecasting IC & RankIC; Realized Volatility Forecasting MAE & R².
> "increasing the vocabulary size improves both reconstruction quality and forecasting accuracy. A larger vocabulary provides a finer-grained representation, reducing quantization error."

### 10.4 Figure 7 — Test-time scaling (number of inference samples)
> "a consistent improvement in both IC and RankIC as more samples are included in the ensemble." N ∈ {1, 5, 10, 20}; mean over 5 random seeds.

### 10.5 Zero-shot vs fine-tuned
**There is no fine-tuned Kronos in this paper.** All Kronos results are zero-shot. The paper does compare **zero-shot TSFM baselines** (TimeMOE, Moirai, TimesFM, Moment, Chronos) against **full-shot** (trained-from-scratch) models, and Kronos beats both:
> "general-purpose TSFMs often underperform specialized, non-pre-trained models (e.g., iTransformer) on financial tasks" — yet Kronos (zero-shot) outperforms them.
Numbers: Price Series average RankIC Kronos_large 0.0267 vs best zero-shot TSFM (TimeMOE_base) 0.0138 and best full-shot (NSTransformer) 0.0082. Vol average R² Kronos_large 0.2624 vs best zero-shot 0.1492. No "Kronos zero-shot vs Kronos fine-tuned" table exists.

---

## 11. REPRODUCIBILITY

- **Official code URL:** `https://github.com/shiyu-coder/Kronos` (verbatim: *"Our pre-trained model is publicly available at https://github.com/shiyu-coder/Kronos."*)
- **BSQ reference implementation:** `https://github.com/zhaoyue-zephyrus/bsq-vit` (footnote for the BSQ hyperparameters).
- **HuggingFace model IDs:** **NOT STATED IN PAPER.** (Only the library version is given: Hugging Face Hub 1.57.4.)
- **License:** **NOT STATED IN PAPER.**
- **Checkpoints / weights open?** Claimed public via the GitHub URL; no HF repo ID, no checkpoint file names, no size/format listed. The word "checkpoint" does not appear.
- **Versions:** OS Ubuntu 24.04.1 LTS; Python 3.13.2; PyTorch 2.7.0; NumPy 1.26.2; Pandas 2.2.2; Matplotlib 3.9.3; Hugging Face Hub 1.57.4.
- **Hardware:** 3 × K8s pods; per pod 96 CPU cores (Intel Xeon Gold 6330 @ 2.00 GHz), 200 GB RAM, 8 × NVIDIA GeForce RTX 4090D; total 24 GPUs.
- **Paper identifier:** `arXiv:2508.02739v1 [q-fin.ST] 2 Aug 2025`. The task brief states AAAI 2026 — no camera-ready/venue string is in the extracted text.
- **Data release:** **NOT STATED** (no statement that the 12B pre-training corpus or eval data is public). Backtest data via Qlib; Qlib paper = Yang et al. 2020.

---

## 12. LIMITATIONS, LOOK-AHEAD BIAS, LEAKAGE

- **No dedicated Limitations / Broader Impacts / Ethics section exists.** The words "Limitation", "look-ahead", "lookahead", "leakage", "Sharpe" do not appear as such. ("limitations" appears only rhetorically about TSFMs, e.g., *"the very generality that drives their success on broad benchmarks becomes a limitation in specialized domains."*)
- **What the authors DO admit:**
  - Generality is a liability for TSFMs: *"general-purpose TSFMs often underperform specialized, non-pre-trained models... on financial tasks and fail to generalize across the broader landscape of quantitative finance."*
  - Resource-driven context cap: *"Considering resource constraints and practical deployment scenarios, we limit the maximum context length to 512 tokens."*
  - Class imbalance forced resampling (a data-quality admission): *"The raw pre-training corpus exhibits a natural imbalance across asset classes, with equities being more prevalent than cryptocurrencies, futures, and foreign exchange (forex) assets."*
  - Tokenizer hierarchy has ~no fidelity cost (framed positively): *"achieves reconstruction quality nearly identical to that of the standard loss variant."*
  - Table 12 shows n=2 has a latency cost (2 sequential steps/token) — admitted trade-off, not framed as a limitation.
  - Daily XSHG generative discriminative score for TimeVAE is 0.0142 while DiffusionTS hits 0.2532 on XSHG daily — they report it plainly.
- **Look-ahead bias / leakage:** **NO EXPLICIT STATEMENT IN PAPER.** The only leakage-adjacent control stated is the temporal separation: *"our test period for all tasks begins in July 2024 to ensure a strict temporal separation between training and evaluation"* and *"The pre-training data for Kronos extends up to June 2024."* No discussion of survivorship bias, point-in-time fundamentals, or whether the Qlib data was adjusted for splits/dividends other than the structural-break segmentation in cleaning.
- **Sharpe ratio:** never reported; backtest reports AER and IR instead.

---

## OPEN QUESTIONS / WHAT IS NOT STATED (reproduction gaps)

1. **Data vendor(s)** for the 12.11B-record pre-training corpus — no source/feed provider named.
2. **Pre-training batch size** — not given (batch 256 is baselines only).
3. **Pre-training compute in GPU-hours / wall-clock** — only "24 × RTX 4090D" given; total training duration unknown.
4. **Total pre-training steps / tokens processed** — only warm-up = 15,000 steps.
5. **Number of pre-training epochs** — not stated.
6. **Validation split for pre-training** — not described; no early-stopping criteria for Kronos.
7. **Exact train/val/test boundaries per task** — only "test begins July 2024"; no train window or val window sizes.
8. **How the 4 price-channel IC/RankIC average is weighted** (per sample vs per channel) and whether IC is Pearson or Spearman — "IC/RankIC" definitions never formalised.
9. **Definition of "Annualized Excess Return (AER)" and "Information Ratio (IR)"** in the backtest — formulas not given.
10. **Backtest period boundaries** — curve spans 2024-07→2025-06; exact start/end dates and warm-up not stated. Also, with H=10 forecast horizon and a "minimum holding period of 5 days", the horizon/holding reconciliation is unexplained.
11. **Backtest signal exact formula** — "H-day average expected return (R_{t→t+H})" is described but the precise arithmetic (mean of predicted closes vs mean of the H monthly returns) is not spelled out; it also does not say whether prediction uses only past data at each rebalance (no look-ahead guarantee stated explicitly).
12. **Frequency list inconsistency** — text says "7 sampling frequencies" while Table 4 lists 12 (1min, 5, 10, 15, 20, 30, 40min, 60min, 2H, 4H, D, W) and Table 13 uses T/5T/15T/30T/H/D/W.
13. **Exchange-count inconsistency** — "45 global exchanges" (abstract/body) vs "over 40 exchanges across more than 30 countries" (Appendix B) vs Table 13 listing ~50 rows + foreign-exchange + index rows.
14. **Record-count inconsistency** — "over 12 billion" vs Table 13 total "12.11B" vs "96569" assets.
15. **HuggingFace model IDs / checkpoint filenames** — not given.
16. **License / usage terms / dataset licensing** — not given.
17. **Whether the pre-training dataset is released** — not stated; only the model repo URL.
18. **Species of model scoring in Figure 1** — the radar chart's per-axis values are jumbled; no clean table.
19. **Token-embedding of volume/amount when absent (crypto/forex OHLC-only)** — the paper says volume/amount are excluded but does not say how D=6 is handled (padding? separate D=4 model?). Ambiguous.
20. **"Kronos-mini"** — task asks for it; the paper releases only small/base/large. Any mini variant is external to the paper.
21. **RoPE base/theta, attention dropout schedule beyond Table 5, SwiGLU vs GELU, tokenizer latent dimension** — not specified.
22. **Eval sampling: whether N=10 samples are averaged for every baseline equally**, and reservoirs/seeds for stochastic baselines — partially stated (5 seeds for Fig 7) but not for the main tables.
23. **Table 21 Crypto daily IC/RankIC** and several Table 14/16/18 middle rows — lost to PDF table extraction; re-extract from the PDF directly to recover exact values.
24. **Any statement on statistical significance** (confidence intervals, p-values) — none reported.
25. **AAAI 2026 acceptance / camera-ready** — not verifiable from text (arXiv v1 only).
