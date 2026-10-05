#!/usr/bin/env python3
"""Document the bounded-action-space bug found while building the fast trainer.

This is a real implementation pitfall, not a tuning anecdote, and it is the kind
of thing that silently produces "the RL arm doesn't work" conclusions. It is
recorded in the reproduction notes and in the paper's reproduction section.
"""

REPRO = "/opt/data/kvab/docs/reproduction.md"
TEX = "/opt/data/kvab/paper/kronos_vs_alphazerobeta.tex"

REPRO_ADD = r"""

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
"""

TEX_ADD = r"""
\subsection{A projected action space breaks the naive policy likelihood}
\label{sec:projection-bug}

Every one of the divergence failures reported above traces to one implementation
decision, and it is worth recording because it is easy to make and hard to see.
The action space is constrained by construction: the policy output is projected to
be dollar-neutral with bounded gross exposure. The natural implementation samples
from the Gaussian policy and then projects the sample. In the four runs we
attempted, every run that fed the \emph{projected} action back into the log-density
diverged, and every run that evaluated the density at the \emph{raw sample}
trained stably:

\begin{table}[t]
\centering
\footnotesize
\begin{tabular}{llrl}
\toprule
Run & log\_prob at & Non-finite updates & Loss trajectory \\
\midrule
A & projected & many & $9.6 \to 3.7\times10^{4} \to 1.5\times10^{34}$ \\
B & projected & 2543 & median $4.7$, max $2.5\times10^{35}$ \\
C & \textbf{raw sample} & \textbf{0} & $4.61 \to 4.07 \to 4.14 \to 3.99$ \\
\bottomrule
\end{tabular}
\caption{The single change that made the reproduction train. The projection is a deterministic map, so it regularly moves a sample far into the tail of the density being optimised; the clipped ratio then diverges.}
\label{tab:projbug}
\end{table}

A diagonal Gaussian has support on all of $\mathbb{R}^N$; the $\ell_1$-ball
projection does not. Evaluating the density at the projected point therefore
differentiates the likelihood of a point that may be arbitrarily improbable under
the proposal, and in the clipped objective the ratio $\exp(\log\pi_t -
\log\pi_{t-1})$ inherits that. The correct treatment of a bounded or squashed
action is to evaluate the density at the raw sample and apply the projection as a
deterministic action map after sampling. Only the raw sample lies in the support
of the density being optimised.

This interacts with the reward. The published formulation floors $\sigma_p$ at
$1\times10^{-8}$, so a near-flat book makes both the reward ratio and the
likelihood ratio ill conditioned at once, and the null portfolio becomes a
spurious attractor. That is precisely where the first run settled, at zero gross
exposure on every out-of-sample day. We therefore also report the number of
non-finite gradient updates as a training diagnostic, rather than reading a flat
learning curve as evidence about the method.
"""


def main():
    r = open(REPRO).read()
    if "D14" not in r:
        open(REPRO, "a").write(REPRO_ADD)
        print("reproduction addendum written")
    else:
        print("already present")

    s = open(TEX).read()
    if "projection-bug" not in s:
        anchor = "\\section{Computational Analysis}"
        s = s.replace(anchor, TEX_ADD + "\n" + anchor)
        open(TEX, "w").write(s)
        print("paper subsection inserted")
    else:
        print("paper subsection already present")


if __name__ == "__main__":
    main()
