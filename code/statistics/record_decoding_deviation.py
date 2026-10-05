#!/usr/bin/env python3
"""Record the Kronos decoding-configuration deviation discovered in the paper
extraction, in both the reproduction notes and the paper itself.

The Kronos paper's Table 6 recommends, for price-series forecasting:
temperature 0.6, top-p 0.90, and 10 averaged samples. The runs reported here
used temperature 1.0 and a single sample. That is a material deviation and it
disadvantages the forecast arm, so it must be stated next to the results rather
than buried: with N=1 there is no variance reduction over the sampling noise,
which is exactly the mechanism the paper's N=10 averaging exists to suppress.
"""

SRC = "/opt/data/kvab"
REPRO = f"{SRC}/docs/reproduction.md"
TEX = f"{SRC}/paper/kronos_vs_alphazerobeta.tex"

REPRO_ADD = r"""

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
"""

TEX_ADD = r"""
\subsection{Disclosed decoding deviation for the forecast arm}
\label{sec:kronos-decoding}

The source model's own inference guidance recommends, for price-series
forecasting, a temperature of $0.6$ with ten averaged samples. The runs reported
here used a temperature of $1.0$ and a single sample. We state this next to the
results rather than in a footnote, because it works against the forecast arm:
the model is a generative sampler, so a single draw retains the full sampling
variance, and averaging ten draws is the mechanism that suppresses it. A noisier
per-asset expected-return estimate produces a noisier cross-sectional ranking,
which is precisely what erodes a long/short spread.

The forecast results in Table~\ref{tab:forecast} and the Kronos row of
Table~\ref{tab:portfolio} should therefore be read as a lower bound on what the
released checkpoint can do under this protocol. Re-running inference at the
recommended setting costs about ten times the inference time and was outside the
compute budget available here; it is the first item of future work.

"""


def main():
    r = open(REPRO).read()
    if "deviation D11" not in r:
        open(REPRO, "a").write(REPRO_ADD)
        print("reproduction addendum written")
    else:
        print("addendum already present")

    s = open(TEX).read()
    if "Disclosed decoding deviation" not in s:
        anchor = "\\subsection{AlphaZeroBeta arm}"
        if anchor in s:
            s = s.replace(anchor, TEX_ADD + anchor)
        else:
            anchor2 = "\\subsection{Baselines}"
            s = s.replace(anchor2, TEX_ADD + anchor2)
        open(TEX, "w").write(s)
        print("paper subsection inserted")
    else:
        print("paper subsection already present")


if __name__ == "__main__":
    main()
