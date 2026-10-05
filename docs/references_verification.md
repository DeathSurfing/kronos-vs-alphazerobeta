# Bibliography Verification — Kronos vs AlphaZeroBeta

Target bib: `/opt/data/kvab/paper/kronos_vs_alphazerobeta.bib`
Validation: `bibtexparser` 2.1.0 → 30 entries, 0 failed blocks, 30 unique keys, all have title/author/year.
Every entry below was confirmed by fetching the listed URL (arXiv atom/abs API, Crossref REST API, or publisher DOI). Semantic Scholar was rate-limited (HTTP 429) during collection, so arXiv API + Crossref were used instead.

| key | verification URL | evidence (title as shown on that URL) |
|---|---|---|
| ansari2024chronos | https://arxiv.org/abs/2403.07815 | "Chronos: Learning the Language of Time Series" |
| das2024timesfm | https://arxiv.org/abs/2310.10688 | "A decoder-only foundation model for time-series forecasting" |
| woo2024moirai | https://arxiv.org/abs/2402.02592 | "Unified Training of Universal Time Series Forecasting Transformers" (Moirai) |
| nie2023patchtst | https://arxiv.org/abs/2211.14730 | "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (comments: Accepted by ICLR 2023) |
| liu2024itransformer | https://arxiv.org/abs/2310.06625 | "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" |
| fischer2018lstm | https://api.crossref.org/works/10.1016/j.ejor.2017.11.054 | "Deep learning with long short-term memory networks for financial market predictions", European Journal of Operational Research, 270(2):654-669, 2018 |
| chung2014gru | https://arxiv.org/abs/1412.3555 | "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" |
| feng2019rsr | https://arxiv.org/abs/1809.09441 | "Temporal Relational Ranking for Stock Prediction" (journal ref: ACM Trans. Inf. Syst. 37, 2, Art. 27, 2019; DOI 10.1145/3309547) |
| corsi2009har | https://api.crossref.org/works/10.1093/jjfinec/nbp001 | "A Simple Approximate Long-Memory Model of Realized Volatility", Journal of Financial Econometrics, 7(2):174-196 |
| meese1983randomwalk | https://api.crossref.org/works/10.1016/0022-1996(83)90017-X | "Empirical exchange rate models of the seventies: Do they fit out of sample?", Journal of International Economics, 14(1-2):3-24 |
| jiang2017portfolio | https://arxiv.org/abs/1706.10059 | "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" |
| zhang2020drltrading | https://arxiv.org/abs/1911.10107 | "Deep Reinforcement Learning for Trading" (Zhang, Zohren, Roberts) |
| liu2021finrl | https://arxiv.org/abs/2011.09607 | "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance" |
| moody2001direct | https://api.crossref.org/works/10.1109/72.935097 | "Learning to trade via direct reinforcement", IEEE Transactions on Neural Networks, 12(4):875-889 |
| schulman2017ppo | https://arxiv.org/abs/1707.06347 | "Proximal Policy Optimization Algorithms" |
| schulman2016gae | https://arxiv.org/abs/1506.02438 | "High-Dimensional Continuous Control Using Generalized Advantage Estimation" |
| fama2015fivefactor | https://api.crossref.org/works/10.1016/j.jfineco.2014.10.010 | "A five-factor asset pricing model", Journal of Financial Economics, 116(1):1-22 |
| avellaneda2010statarb | https://api.crossref.org/works/10.1080/14697680903124632 | "Statistical arbitrage in the US equities market", Quantitative Finance, 10(7):761-782 |
| frazzini2014bab | https://api.crossref.org/works/10.1016/j.jfineco.2013.10.005 | "Betting against beta", Journal of Financial Economics, 111(1):1-25 |
| kelly1956criterion | https://api.crossref.org/works/10.1002/j.1538-7305.1956.tb03809.x | "A New Interpretation of Information Rate", Bell System Technical Journal, 35(4):917-926 |
| white2000snooping | https://api.crossref.org/works/10.1111/1468-0262.00152 | "A Reality Check for Data Snooping", Econometrica, 68(5):1097-1126 |
| bailey2014dsr | https://api.crossref.org/works/10.3905/jpm.2014.40.5.094 | "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality", Journal of Portfolio Management, 40(5):94-107 |
| bailey2016pbo | https://api.crossref.org/works/10.21314/JCF.2016.322 | "The probability of backtest overfitting", Journal of Computational Finance |
| diebold1995dm | https://api.crossref.org/works/10.1080/07350015.1995.10524599 | "Comparing Predictive Accuracy", Journal of Business & Economic Statistics, 13(3):253-263 |
| newey1987hac | https://api.crossref.org/works/10.2307/1913610 | "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix", Econometrica, 55(3):703-708 |
| frazzini2018tradingcosts | https://api.crossref.org/works/10.2139/ssrn.3229719 | "Trading Costs", SSRN (Frazzini, Israel, Moskowitz) |
| sezer2020survey | https://api.crossref.org/works/10.1016/j.asoc.2020.106181 | "Financial time series forecasting with deep learning: A systematic literature review: 2005-2019", Applied Soft Computing, 90:106181 |
| felizardo2022survey | https://arxiv.org/abs/2212.06064 | "Reinforcement Learning Applied to Trading Systems: A Survey" |
| shi2025kronos | https://ojs.aaai.org/index.php/AAAI/article/view/39730 | "Kronos: A Foundation Model for the Language of Financial Markets" — AAAI proceedings page; citation_doi 10.1609/aaai.v40i30.39730, vol 40, no 30, pp. 25366-25373, authors Yu Shi, Zongliang Fu, Shuo Chen, Bohan Zhao, Wei Xu, Changshui Zhang (arXiv 2508.02739) |
| belyakov2026alphazerobeta | https://api.crossref.org/works/10.1186/s40854-026-00955-4 | "AlphaZeroBeta: deep reinforcement learning for market-neutral portfolios", Financial Innovation, vol 12, issue 1, article 156, published 2026-09-20, DOI 10.1186/s40854-026-00955-4 (arXiv 2607.18001) |

## Notes
- Subject papers appear exactly once each: `shi2025kronos` (Kronos), `belyakov2026alphazerobeta` (AlphaZeroBeta).
- arXiv-only entries (no peer-reviewed venue confirmed on the fetched page) are typed `@article` with `journal = {arXiv preprint ...}` so no venue is invented. Chronos, Moirai, TimesFM, PatchTST, iTransformer, FinRL, Zhang-Zohren-Roberts, Jiang-Xu-Liang, GAE, PPO, and the two surveys fall here.
- Peer-reviewed entries carry verified DOI + venue (Crossref).
- `frazzini2018tradingcosts` is a working paper (SSRN DOI); Crossref has no container title, so it is typed `@techreport`.
