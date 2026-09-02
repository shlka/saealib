---
primary_layer: layer1
related_layers: [layer2]
page_type: tutorial
---

# Single-Objective Optimization

Solve a single-objective optimization problem with an expensive-to-evaluate objective function, using `saealib`.

First define the problem, then run single-objective optimization with the high-level API's `minimize`.

For the detailed specification and customization of each component, see the [Concepts](../concepts/index.md) pages linked from the following sections.

:::{admonition} What you'll be able to do
:class: tip

By the end of this page, you'll be able to optimize an expensive single-objective function with `minimize()`.
:::

For a quick run that only requires an objective function, use the [high-level API](highlevel_api.md); to choose components by responsibility, see [Swap built-in components](component_swap.md).
This page covers running single-objective optimization with `minimize()` and selecting the algorithm, Surrogate, and evaluation Strategy with string arguments.

## Set up the problem

Assume an objective function whose single call takes a long time, like a simulation.

Here, as an example, we minimize the Sphere function as a stand-in for an expensive-to-evaluate function.

```python
import numpy as np


def expensive_func(x):
    # assume a function that is expensive to call in practice
    return np.sum(x**2)


DIM = 10
LB = [-5.0] * DIM
UB = [5.0] * DIM
```

`DIM` is the number of design-variable dimensions, and `LB`/`UB` are `DIM`-dimensional lists giving its lower and upper bounds.

The objective function is defined as a `Callable` that takes a `DIM`-dimensional design variable and returns the objective value.

## Run minimize or maximize

`minimize` is a high-level API that runs an optimization just by specifying `dim`, `lb`, and `ub`.

```python
from saealib import minimize

result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, seed=0)

print(result.x)  # optimal design variables  shape: (dim,)
print(result.f)  # optimal objective value  shape: (n_obj,)
print(result.fe)  # true function evaluations
print(result.gen)  # completed generations
```

If the maximum number of evaluations `max_fe` is omitted, `200 * dim` is used as the default.

To explicitly limit the number of evaluations, specify it as follows.

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, max_fe=500, seed=0)
```

## Switch components

`minimize` lets you switch three components — the evolutionary algorithm, the surrogate model, and the evaluation strategy — via the string-valued `algorithm`, `surrogate`, and `strategy` arguments, respectively.

For all three, you can also pass an instance directly instead of a string.

The internal behavior and customization of each component are covered on the [Algorithm](../concepts/search_algorithms/algorithm.md), [Surrogate](../concepts/surrogate_modeling/surrogate.md), and [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md) pages.

### Algorithm

The `algorithm` argument selects the evolutionary algorithm that generates candidate solutions.

| String | Class | Characteristics |
|--------|--------|------|
| `'GA'` | `GA` | Search via crossover and mutation (default) |
| `'PSO'` | `PSO` | Search via particle velocity updates |

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, algorithm="PSO", seed=0)
```

### Surrogate

The `surrogate` argument selects the surrogate model that approximates the objective function.

| String | Resolved configuration | Description |
|--------|--------|------|
| `'rbf'` | `RBFSurrogate` + `LocalSurrogateManager` (default) | Local fit over nearby points using a Gaussian RBF kernel |

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, surrogate="rbf", seed=0)
```

### Evaluation strategy

The `strategy` argument selects the evaluation strategy that decides which of the generated candidates receive a true (expensive) evaluation.

| String | Class | Behavior |
|--------|--------|------|
| `'ib'` | `IndividualBasedStrategy` | Evaluates each candidate individually with the surrogate and truly evaluates only the top `evaluation_ratio` fraction (default) |
| `'gb'` | `GenerationBasedStrategy` | Advances `gen_ctrl` generations using only the surrogate, then truly evaluates a single generation |
| `'ps'` | `PreSelectionStrategy` | Narrows down a large pool of candidates with the surrogate and truly evaluates only the top `n_select` |

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, strategy="ib", seed=0)
```

When the default `minimize()` configuration is insufficient, configure components individually and register them with `Optimizer`. [Build an Optimizer with the low-level API](lowlevel_api.md) also covers multiple Termination conditions and per-generation progress through `iterate()`.

## Related concepts and reference

- {py:func}`saealib.minimize` / {py:func}`saealib.maximize`
- {py:class}`saealib.GA` / {py:class}`saealib.PSO`
- {py:class}`saealib.IndividualBasedStrategy` / {py:class}`saealib.GenerationBasedStrategy` / {py:class}`saealib.PreSelectionStrategy`
