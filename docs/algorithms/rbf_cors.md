---
primary_layer: layer2
related_layers: []
page_type: concept
---

# CORS-RBF (Constrained Optimization using Response Surfaces)

CORS-RBF is a sequential optimization method for expensive-to-evaluate objective functions that selects the next evaluation point one at a time using a surrogate model built from RBF (Radial Basis Function) interpolation.
CORS-RBF is the implementation of CORS (Constrained Optimization using Response Surfaces), the framework proposed by Regis & Shoemaker (2005), realized with an RBF surrogate model.

## Overview

RBF interpolation only reconstructs a smooth surface passing through the training points, and unlike GP regression, it has no predictive variance.
Because of this, the naive approach of simply minimizing the surrogate model's prediction to choose the next evaluation point ends up repeatedly searching only around points where good values have already been observed, and can converge to a point that isn't even a local minimum of the true function.

CORS avoids this problem by building a **distance constraint** directly into the candidate-point selection itself.
The auxiliary problem solved at each iteration is a constrained optimization that, in addition to minimizing the surrogate model $\hat f_i(x)$, requires the next candidate point to be at least $\beta_i \Delta_i$ away from every existing evaluated point ($\Delta_i$ being the maximum of the minimum distances from the existing point set).
$\beta_i$ is given, iteration by iteration, as a sequence (the **search pattern**) that cycles from values near 1 (favoring global search) down to 0 (favoring local search, i.e. simply minimizing the surrogate model); this distance constraint takes over the role of exploration that a GP's predictive variance would otherwise play.

This distance constraint is not a side effect — it is the core of CORS.
It has been proven that, as long as the search pattern contains at least one nonzero value, convergence to the global minimum of any continuous function is guaranteed, regardless of the type of surrogate model or how the initial evaluation points are chosen.

The source is {cite}`regis2005cors`. The concrete procedure is shown in the pseudocode below.

## Pseudocode

```{prf:algorithm} CORS-RBF
:label: alg-rbf-cors

**Inputs** objective function $f$, search domain $\mathcal{D}$, initial evaluated point set $S_1 = \{x_1, \ldots, x_k\}$, periodic distance-parameter sequence (search pattern) $\langle \beta_1, \ldots, \beta_{N+1}=0 \rangle$
**Output** best solution $x^*$

1. Evaluate $S_1$ with the true function $f$ and set $i := 1$
2. Fit an RBF surrogate model $\hat f_i$ to the evaluated data so far, $D_i = \{(x, f(x)) \mid x \in S_i\}$
3. Solve the constrained minimization problem $\min_{x \in \mathcal{D}} \hat f_i(x) \ \mathrm{s.t.} \ \|x - x_j\| \geqslant \beta_i \Delta_i \ (j=1,\ldots,|S_i|)$ to find the candidate point $x_{k+i}$ ($\Delta_i$ being the maximum of the minimum distances from the existing evaluated point set)
4. Evaluate $x_{k+i}$ with the true function and add it to $S_{i+1} := S_i \cup \{x_{k+i}\}$
5. Update $\beta_i$ according to the periodic sequence, set $i := i+1$, and return to step 2 until the termination condition is reached
```

## Flowchart

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Sample initial population<br/>→ true evaluation<br/>(L1)"] --> ASK
    subgraph GEN["One generation (IndividualBasedStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>Generate candidate points"] --> SCORE["SurrogateManager<br/>Fit RBF (L2)<br/>→ Score with predictive mean<br/>(L3)"]
        SCORE --> SORT["Select top<br/>evaluation_ratio fraction by predictive mean"]
        SORT --> EVAL["True evaluation →<br/>add to archive<br/>(L4)"]
        EVAL --> TELL["GA.tell()<br/>Update population"]
    end
    GEN --> TERM{"Evaluation budget N<br/>reached?"}
    TERM -- "Not yet (L5)" --> ASK
    TERM -- "Reached" --> RESULT(["Best solution x*"])
```

## Configuration in saealib

| Role | saealib implementation | Corresponding step |
|---|---|---|
| Search algorithm | `GA` (the specific combination of crossover, mutation, and selection is not part of CORS's definition) | Candidate-point generation |
| Surrogate model | `RBFSurrogate` (RBF interpolation; defaults to `gaussian_kernel`, but any kernel function can be injected) | L2 |
| Acquisition function | `MeanPrediction` (uses the predictive mean directly as the score) | L3 |
| Surrogate management | `GlobalSurrogateManager` (fits the RBF over the entire archive) | L2-3 |
| Evaluation strategy | `IndividualBasedStrategy` (truly evaluates only the individuals with the top predictive-mean scores) | L3-4 |

```python
import numpy as np
from saealib import (
    GA,
    Optimizer,
    Problem,
    IndividualBasedStrategy,
    RBFSurrogate,
    gaussian_kernel,
    MeanPrediction,
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
surrogate_manager = GlobalSurrogateManager(
    RBFSurrogate(gaussian_kernel, dim=5), MeanPrediction()
)
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

Since the specific crossover, mutation, and selection operators are not part of CORS's own definition, the above is just one example, and any `Crossover`/`Mutation`/`ParentSelection`/`SurvivorSelection` can be swapped in.

## Parameters and variants

**kernel (choice of RBF kernel)**: Any kernel function can be injected via `RBFSurrogate(kernel=...)`.
The default is `gaussian_kernel`, but the paper's numerical experiments use a model combining a thin-plate-spline kernel ($\phi(r) = r^2 \log r$) with an added first-degree polynomial term $p(x)$.
Because saealib's `RBFSurrogate` is a pure RBF interpolant with no polynomial term $p(x)$ (it fits the residual after subtracting the training data's mean), swapping the kernel alone is not enough to exactly reproduce the paper's setting.

**evaluation_ratio (switching between sequential and batch evaluation)**: CORS as described in the literature is a sequential algorithm that executes steps 3-4 of the pseudocode for a single point per loop.
saealib's `IndividualBasedStrategy` is a batch extension that truly evaluates the top `evaluation_ratio` fraction of the offspring population generated by the GA's predictive-mean scores at once.
Shrinking `evaluation_ratio` down to roughly the inverse of the population size approximates one-point-at-a-time sequential evaluation.

## Related

- [References](../references.md): Full bibliographic details for the source
- [Surrogate](../concepts/surrogate_modeling/surrogate.md): List of surrogate models including `RBFSurrogate`/`gaussian_kernel`
- [AcquisitionFunction](../concepts/surrogate_modeling/acquisition_functions.md): List of acquisition functions including `MeanPrediction`
- [SurrogateManager](../concepts/surrogate_modeling/surrogate_manager.md): Detailed usage of `GlobalSurrogateManager`
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md): List of strategies including `IndividualBasedStrategy`'s `evaluation_ratio`
