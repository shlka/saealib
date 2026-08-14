---
primary_layer: layer2
related_layers: []
page_type: concept
---

# MaxUnc (Uncertainty Sampling)

MaxUnc is an exploration-only acquisition function that uses a surrogate model's predictive uncertainty (standard deviation) as its criterion, selecting for the next true evaluation the candidate point the model is least confident about.

## Overview

EGO's Expected Improvement and GP-UCB's upper confidence bound were both designed to balance exploration and exploitation using both the predictive mean $\mu(x)$ and predictive standard deviation $\sigma(x)$.
MaxUnc removes the predictive-mean term entirely from this construction, using only $\sigma(x)$ as its criterion.

The procedure is structurally identical to EGO/GP-UCB.
A GP is fit to the entire archive, the point maximizing the predictive standard deviation $\sigma(x)$ is found, evaluated with the true function, and added to the archive.
Because the predictive mean is never referenced, evaluation is always biased toward the region "where the model is furthest from the training data," rather than actively seeking out promising values.

As background for this construction, Büche, Schraudolph & Koumoutsakos (2005), in a survey paper on GP-based surrogate models, proposed a merit function $f_{\mathrm{M}}(x) = \hat{t}(x) - \alpha \sigma_t(x)$ — a linear combination of the predictive mean and predictive standard deviation — and stated that larger $\alpha$ leans more toward exploration {cite}`buche2005gpes`.
The criterion MaxUnc computes, $\sigma(x)$ alone, corresponds to the limit of this merit function as $\alpha \to \infty$ — that is, the case where the predictive-mean contribution is eliminated entirely.

MaxUnc is not a criterion that directly targets improving the objective function.
As an exploration-only component that pairs with exploitation-leaning criteria like EI or UCB, it is suited to uses such as raising the surrogate model's accuracy uniformly across the whole domain, or being combined with other criteria (see also the comparison with `MeanPrediction` in the acquisition function list).
While the evaluation budget is small, it prioritizes filling in the model's unknown regions over converging on the best solution, so used alone, its final objective value may improve less than with EI or LCB.

## Pseudocode

```{prf:algorithm} MaxUnc
:label: alg-maxunc

**Inputs** objective function $f$, search domain, initial sample count $n_0$, evaluation budget $N$
**Output** evaluated archive (the set of truly evaluated points and their function values)

1. Sample an initial population of $n_0$ points, evaluate them with the true function $f$, and add them to the archive
2. Fit a GP to the entire archive, obtaining the predictive standard deviation $\sigma(x)$ at any point (the predictive mean $\mu(x)$ is not used in computing the criterion)
3. Find the point $x^* = \arg\max_x \sigma(x)$ that maximizes the predictive standard deviation
4. Evaluate $x^*$ with the true function and add it to the archive
5. Return to step 2 until the evaluation budget $N$ is reached
```

## Flowchart

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Sample initial population<br/>via LHS etc. → true evaluation<br/>(L1)"] --> ASK
    subgraph GEN["One generation (IndividualBasedStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>Generate candidates"] --> SCORE["SurrogateManager<br/>Fit GP (L2)<br/>→ Score with σ (L3)"]
        SCORE --> SORT["Select top<br/>evaluation_ratio fraction by σ<br/>(approximates argmax σ)"]
        SORT --> EVAL["True evaluation →<br/>add to archive<br/>(L4)"]
        EVAL --> TELL["GA.tell()<br/>Update population"]
    end
    GEN --> TERM{"Evaluation budget N<br/>reached?"}
    TERM -- "Not yet (L5)" --> ASK
    TERM -- "Reached" --> RESULT(["Evaluated archive"])
```

## Configuration in saealib

| Role | saealib implementation | Corresponding step |
|---|---|---|
| Search algorithm | `GA` (the specific combination of crossover, mutation, and selection is not part of MaxUnc's definition) | Candidate generation (search for argmax σ) |
| Surrogate model | `SklearnGPRSurrogate` (GP regression; requires the `sklearn` extra) | L2 |
| Acquisition function | `MaxUncertainty` (scores using only the predictive standard deviation; does not reference the predictive mean) | L3 |
| Surrogate management | `GlobalSurrogateManager` (fits the GP over the entire archive) | L2-3 |
| Evaluation strategy | `IndividualBasedStrategy` (truly evaluates only the individuals with the top σ scores) | L3-4 |

```python
import numpy as np
from saealib import (
    GA,
    Optimizer,
    Problem,
    IndividualBasedStrategy,
    SklearnGPRSurrogate,
    MaxUncertainty,
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
surrogate_manager = GlobalSurrogateManager(SklearnGPRSurrogate(), MaxUncertainty())
strategy = IndividualBasedStrategy(evaluation_ratio=0.2)

opt = (
    Optimizer(problem)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_strategy(strategy)
    .set_termination(Termination(max_fe(200)))
)
ctx = opt.run()
```

Since the specific crossover, mutation, and selection operators are not part of MaxUnc's own definition, the above is just one example, and any `Crossover`/`Mutation`/`ParentSelection`/`SurvivorSelection` can be swapped in.

Running this example with the same 200-FE evaluation budget as the EGO/GP-UCB examples, the best value may improve less than with EI/LCB, precisely because MaxUnc is exploration-only.
This is a natural consequence of MaxUnc being a criterion aimed at reducing model uncertainty rather than improving the objective function, and is the expected behavior.

## Parameters and variants

**weights (aggregating uncertainty across multiple objectives)**: Adjusted via `MaxUncertainty(weights=...)`.
In multi-objective problems, a predictive standard deviation $\sigma_1(x), \ldots, \sigma_m(x)$ is obtained for each objective, and these need to be aggregated into a single score.
The default `weights=None` uses the simple average across objectives (`std.mean(axis=1)`); passing an `np.ndarray` uses a weighted sum with those weights.

`MaxUncertainty` has no parameter analogous to EI's $\xi$ or LCB's $\kappa$ for adjusting the exploration–exploitation trade-off.
By design, since it uses $\sigma(x)$ alone as its criterion, it has no weight on the exploitation side.
To continuously adjust the exploration–exploitation weight, use [GP-UCB](gp_ucb.md)'s `LowerConfidenceBound(kappa=...)` and move toward larger `kappa`.

## Related

- [References](../references.md): Full bibliographic details for the source
- [SurrogateManager](../concepts/surrogate_modeling/surrogate_manager.md): Detailed usage of `GlobalSurrogateManager`
- [AcquisitionFunction](../concepts/surrogate_modeling/acquisition_functions.md): List of acquisition functions including `MaxUncertainty`
- [Surrogate](../concepts/surrogate_modeling/surrogate.md): List of surrogate models including `SklearnGPRSurrogate`, and an explanation of the `sklearn` extra
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md): List of strategies including `IndividualBasedStrategy`'s `evaluation_ratio`
- [EGO](ego.md): A method using the same GP surrogate model + `IndividualBasedStrategy` configuration, replacing the acquisition function with the exploitation-leaning Expected Improvement (EI)
- [GP-UCB](gp_ucb.md): A method using the `LowerConfidenceBound` acquisition function, which has the same $\mu - \kappa\sigma$ structure as Büche et al.'s merit function
