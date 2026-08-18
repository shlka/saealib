---
primary_layer: layer2
related_layers: []
page_type: concept
---

# GP-UCB (Gaussian Process Upper Confidence Bound)

GP-UCB is a sequential optimization method for expensive-to-evaluate objective functions, combining a surrogate model built from Gaussian Process (GP) regression with the **Upper Confidence Bound** (UCB) acquisition function, a linear combination of the predictive mean and predictive standard deviation.

## Overview

GP-UCB extends the **UCB policy** from the multi-armed bandit problem to GP optimization.
In the bandit problem, it is known that continually choosing the arm with the highest upper confidence bound on its reward automatically balances exploration and exploitation.

GP-UCB applies this idea to GP regression over a continuous space, evaluating next the point that maximizes the **upper confidence bound** $\mu(x) + \sqrt{\beta_t}\,\sigma(x)$ of a candidate point $x$.
A point with a high predictive mean $\mu(x)$ corresponds to exploitation, while a point with a large predictive standard deviation $\sigma(x)$ corresponds to exploration, and $\beta_t$ controls the relative weight between these two terms.

The theoretical core of this method lies in choosing $\beta_t$ not as a fixed value but as a function of the iteration count $t$.
Increasing $\beta_t$ according to a specific logarithmic schedule derived from an upper bound on the information gain yields a sublinear upper bound on the cumulative regret {cite}`srinivas2012gpucb`. The concrete procedure is shown in the pseudocode below.

## Pseudocode

```{prf:algorithm} GP-UCB
:label: alg-gp-ucb

**Inputs** objective function $f$ (maximized as a reward), search domain $D$, GP prior $\mu_0=0,\sigma_0,k$, confidence parameter sequence $\beta_t$
**Output** best solution $x^*$

1. Set $t=1$
2. Choose the point $x_t = \arg\max_{x \in D} \mu_{t-1}(x) + \sqrt{\beta_t}\,\sigma_{t-1}(x)$ that maximizes the upper confidence bound
3. Observe $y_t = f(x_t) + \epsilon_t$
4. Perform a Bayesian update with the observation $y_t$, obtaining the posterior mean $\mu_t(x)$ and posterior standard deviation $\sigma_t(x)$
5. Increment $t$ by 1 and return to step 2 until the evaluation budget is reached
```

## Flowchart

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Sample initial population<br/>via LHS etc. → true evaluation"] --> ASK
    subgraph GEN["One generation (IndividualBasedStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>Generate candidates"] --> SCORE["SurrogateManager<br/>Fit GP (L4)<br/>→ Score with LCB (L2)"]
        SCORE --> SORT["Select top<br/>evaluation_ratio fraction by LCB<br/>(approximates argmax UCB)"]
        SORT --> EVAL["True evaluation →<br/>add to archive<br/>(L3)"]
        EVAL --> TELL["GA.tell()<br/>Update population"]
    end
    GEN --> TERM{"Evaluation budget<br/>reached?"}
    TERM -- "Not yet (L5)" --> ASK
    TERM -- "Reached" --> RESULT(["Best solution x*"])
```

## Configuration in saealib

The cited GP-UCB formulation is expressed as **maximizing** a reward, whereas `LowerConfidenceBound` assumes minimization and computes $\mathrm{LCB}(x) = \mu(x) - \kappa\sigma(x)$, returning it with its sign flipped so that score comparisons line up with saealib's other acquisition functions (to match saealib's overall convention that "higher score is better").

Converting $\mu(x)$ into a minimization space and then flipping the sign gives $-(\mu(x) - \kappa\sigma(x)) = -\mu(x) + \kappa\sigma(x)$, which matches the direction of the upper confidence bound $\mu(x) + \kappa\sigma(x)$ in the maximization space.
Therefore, `LowerConfidenceBound`'s `kappa` corresponds to the paper's $\sqrt{\beta_t}$.

| Role | saealib implementation | Corresponding step |
|---|---|---|
| Search algorithm | `GA` (the specific combination of crossover, mutation, and selection is not part of GP-UCB's definition) | L2 |
| Surrogate model | `SklearnGPRSurrogate` (GP regression; requires the `sklearn` extra) | L4 |
| Acquisition function | `LowerConfidenceBound` (`kappa` corresponds to the paper's $\sqrt{\beta_t}$; see the next section for details) | L2 |
| Surrogate management | `GlobalSurrogateManager` (fits the GP over the entire archive) | L2, L4 |
| Evaluation strategy | `IndividualBasedStrategy` (truly evaluates only the individuals with the top UCB scores) | L2-3 |

```python
import numpy as np
from saealib import (
    GA,
    Optimizer,
    Problem,
    IndividualBasedStrategy,
    SklearnGPRSurrogate,
    LowerConfidenceBound,
)
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.surrogate import GlobalSurrogateManager
from saealib.termination import Termination, max_fe


def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)


problem = Problem(sphere, dim=5, lb=[-5] * 5, ub=[5] * 5, n_obj=1, direction=[-1])

algorithm = GA(
    CrossoverBLXAlpha(prob=0.7, alpha=0.4),
    MutationUniform(prob_var=0.3),
    SequentialSelection(),
    TruncationSelection(),
)
surrogate_manager = GlobalSurrogateManager(SklearnGPRSurrogate())
strategy = IndividualBasedStrategy(evaluation_ratio=0.2)

opt = (
    Optimizer(problem)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_acquisition(LowerConfidenceBound(kappa=2.0))
    .set_strategy(strategy)
    .set_termination(Termination(max_fe(200)))
)
ctx = opt.run()
```

## Differences from the source

The original GP-UCB procedure finds the UCB maximizer at every iteration. saealib approximates this `argmax UCB` by selecting the top UCB candidates from the pool `GA` generates; `IndividualBasedStrategy.evaluation_ratio` lets several of the selected candidates receive true evaluation together.

For a finite set $D$, the original method chooses the following $\beta_t$, increasing logarithmically in the round index $t$ (starting at 1, incrementing once per `score()` call), to obtain a theoretical upper bound on cumulative regret:

$$\beta_t = 2 \log\left(\frac{|D|\, t^2 \pi^2}{6\delta}\right)$$

The paper also reports that cross-validating a 1/5 scale factor on this schedule performs better in practice, since the theoretical schedule as-is is often too exploratory.
saealib's default (`kappa=2.0`, fixed regardless of $t$) does not follow this schedule and carries no such guarantee.

## Parameters and variants

**κ (exploration–exploitation trade-off)**: Adjusted via `LowerConfidenceBound(kappa=...)`. The default is `2.0`.

**β_t schedule**: Pass `beta_schedule=gp_ucb_beta_schedule(domain_size, delta)` to reproduce the $\beta_t$ formula from "Differences from the source" above instead of a fixed `kappa`; `domain_size` stands in for the paper's $|D|$, since saealib's search space is continuous.

| Setting | `kappa` used | Regret guarantee |
|---|---|---|
| `beta_schedule=None` (default) | fixed `2.0` | none (fixed-weight heuristic) |
| `beta_schedule=gp_ucb_beta_schedule(domain_size, delta)` | $\sqrt{\beta_t}$, per the schedule above | reproduces the cited bound |

## Related

- [References](../references.md): Full bibliographic details for the source, and a list of sources for acquisition functions other than LCB
- [SurrogateManager](../concepts/surrogate_modeling/surrogate_manager.md): Detailed usage of `GlobalSurrogateManager`
- [AcquisitionFunction](../concepts/surrogate_modeling/acquisition_functions.md): List of acquisition functions including `LowerConfidenceBound`
- [Surrogate](../concepts/surrogate_modeling/surrogate.md): List of surrogate models including `SklearnGPRSurrogate`, and an explanation of the `sklearn` extra
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md): List of strategies including `IndividualBasedStrategy`'s `evaluation_ratio`
- [EGO](ego.md): A method with the same GP surrogate model + `IndividualBasedStrategy` configuration, replacing the acquisition function with Expected Improvement (EI)
