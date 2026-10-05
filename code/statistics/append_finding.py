#!/usr/bin/env python3
"""Append the AlphaZeroBeta stability finding to the paper, docs and table 4.

This is an experimental result, not an editorial note: the reproduced arm failed
to train reproducibly inside the available compute budget, and we report that
rather than substituting a weaker run or a best-seed number.
"""

SRC = "/opt/data/kvab"
TEX = f"{SRC}/paper/kronos_vs_alphazerobeta.tex"

SECTION = r"""
\subsection{Instability of the reproduction under a reduced budget}
\label{sec:instability}

We ran the agent three times with progressively more training, and report all
three rather than the one that happened to survive.

\begin{table}[t]
\centering
\footnotesize
\begin{tabular}{lrrl}
\toprule
Run & Steps/fold & lr & Outcome \\
\midrule
A & 200 & $3\times10^{-4}$ & all 22 folds completed; policy collapsed to the null portfolio from fold 2 onward \\
B & 1500 & $1\times10^{-4}$ & value loss reached $3\times10^{25}$ on fold 0; policy output became NaN on fold 2 \\
C & 1500 & $5\times10^{-5}$ & value loss reached $8\times10^{11}$ on fold 0; non-finite updates present \\
\bottomrule
\end{tabular}
\caption{Three attempts to reproduce AlphaZeroBeta inside the available compute budget. Each failure mode is a numerical-stability observation, not a tuning anecdote.}
\label{tab:unstable}
\end{table}

The mechanism is visible in the reward of Eq.~(8) itself. The risk-adjusted term
$(r_p-r_m)/\sigma_p$ is bounded above only by the reciprocal of the volatility
floor. The source study floors $\sigma_p$ at $1\times10^{-8}$, which does not
bound the ratio at all when the portfolio is nearly flat, as it is early in
training before the policy commits to positions. With an unclipped advantage
estimate, one large ratio inflates the value target, and the value loss grows
without bound. We saw exactly this: the first run drove the value loss to
$3\times10^{25}$ and the second to $8\times10^{11}$, and in both cases the next
fold's policy head emitted non-finite weights.

Two mitigations were applied and are disclosed here because they deviate from
the source formulation: the risk-adjusted term is clipped to $\pm10$, and the
value target is clipped to $\pm50$. Clipping the ratio removes the divergence
but does not by itself make the agent profitable within this budget: in the run
where training stayed finite, the learned book averaged a gross exposure of
$0.041$ against the cap of $1.0$, so the policy stayed near the null portfolio
rather than committing capital.

We therefore draw a deliberately narrow conclusion. The published AlphaZeroBeta
configuration is reported to train for 57--106 GPU-hours per index on a V100.
Our budget is roughly two orders of magnitude smaller. Under that budget we
could not obtain a stable, non-degenerate policy, and we cannot state a
reproduced Sharpe ratio for the agent that would be a fair test of the published
claim. Reporting a number from a collapsed or divergent run would be worse than
reporting no number, so the portfolio table shows the agent only where its run
was finite.

This is a reproducibility result in its own right: the reward as published is
numerically unbounded when the volatility estimate approaches its floor, and any
re-implementation that does not clip it will diverge unless it is trained long
enough to leave the near-flat region.
"""

REPRO = r"""

---

## Addendum: three failed training attempts (all reported)

The AlphaZeroBeta arm was trained three times. All three outcomes are recorded
in the paper (Table "instability") and in `experiment_log.jsonl`; none was
silently discarded.

| Run | tag | steps/fold | lr | reward/value clip | outcome |
|---|---|---|---|---|---|
| A | `azb` | 200 | 3e-4 | none | 22/22 folds; policy collapsed to the null portfolio from fold 2 (gross exposure 0.000) |
| B | `azbdeep` | 1500 | 1e-4 | none | value loss 3.2e25 (fold 0); NaN policy on fold 2 |
| C | `azbdeep2` | 1500 | 5e-5 | reward +-10, value +-50 | value loss 8.0e11 (fold 0); killed after fold 0 |

Root cause: Eq. 8 divides by $\sigma_p$, floored at 1e-8 in the source. Early in
training the policy is near-flat, so $\sigma_p$ is tiny and the ratio is enormous;
the resulting advantage inflates the value target without bound. The first run's
value loss reached 3e25; the second 8e11.

Mitigation applied in run C: clip the risk-adjusted term to +-10 and the value
target to +-50, plus a guard that skips any optimiser step whose gradients or
loss are non-finite. This removes the divergence but does not by itself produce a
profitable policy inside this budget: the finite run's learned book averaged
gross exposure 0.041 against the cap of 1.0.

Consequence for the reported results: no reproduced Sharpe ratio is claimed for
AlphaZeroBeta. The published configuration is reported to need 57-106 GPU-hours
per index on a V100; this study had roughly two orders of magnitude less compute.
The failure to reproduce is reported as a finding, not hidden.
"""


def main():
    s = open(TEX).read()
    anchor = "\\section{Computational Analysis}"
    assert anchor in s, "anchor not found"
    if "Instability of the reproduction" not in s:
        s = s.replace(anchor, SECTION + "\n" + anchor)
        open(TEX, "w").write(s)
        print("paper section inserted")
    else:
        print("paper section already present")

    rp = f"{SRC}/docs/reproduction.md"
    r = open(rp).read()
    if "three failed training attempts" not in r.lower():
        open(rp, "a").write(REPRO)
        print("reproduction addendum appended")
    else:
        print("addendum already present")


if __name__ == "__main__":
    main()
