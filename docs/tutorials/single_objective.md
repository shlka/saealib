---
primary_layer: layer1
---

# Single-Objective Optimization

Solve a single-objective optimization problem with an expensive-to-evaluate objective function, using `saealib`.

First we define the problem and solve it with the high-level API `minimize`, then move on to the low-level API with `Optimizer`.

For the detailed specification and customization of each component, see the [Concepts](../concepts/index.md) pages linked from the following sections.

## Problem setup

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

## High-level API: minimize / maximize

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

## Switching components

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

## Low-level API: Optimizer

`minimize` wires up each component with a default combination, but doesn't let you tune individual parameters.

Instantiating components individually and assembling them into `Optimizer` removes this limitation.

`LocalSurrogateManager`にはSurrogateと学習データの作り方を渡し、AcquisitionFunctionは `Optimizer.set_acquisition()` で別に設定します。
この二つを同じコンストラクタ引数として扱わないでください。

```python
import numpy as np
from saealib import (
    Problem,
    Optimizer,
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
    RBFSurrogate,
    gaussian_kernel,
    LocalSurrogateManager,
    MeanPrediction,
    IndividualBasedStrategy,
    LHSInitializer,
    Termination,
    max_fe,
)

DIM = 10
SEED = 0

problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),  # -1: minimize
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
)

algorithm = GA(
    crossover=CrossoverBLXAlpha(0.7, 0.4),
    mutation=MutationUniform(0.3),
    parent_selection=SequentialSelection(),
    survivor_selection=TruncationSelection(),
)

surrogate_manager = LocalSurrogateManager(
    RBFSurrogate(gaussian_kernel, dim=DIM),
)

strategy = IndividualBasedStrategy(evaluation_ratio=0.1)

initializer = LHSInitializer(
    n_init_archive=5 * DIM,
    n_init_population=4 * DIM,
    seed=SEED,
)

termination = Termination(max_fe(500))

ctx = (
    Optimizer(problem, seed=SEED)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_acquisition(MeanPrediction())
    .set_strategy(strategy)
    .set_termination(termination)
    .run()
)

archive_x = ctx.archive.get_array("x")
archive_f = ctx.archive.get_array("f")[:, 0]
best_idx = int(np.argmin(archive_f))
print("Best solution:", archive_x[best_idx])
print("Objective value:", archive_f[best_idx])
print("Evaluations:", ctx.fe)
```

Pass the same value to both `Optimizer(problem, seed=SEED)` and `LHSInitializer(..., seed=SEED)` for the random seed.

`Optimizer`'s `seed` is only auto-propagated to the default `LHSInitializer` when you skip calling `set_initializer()` yourself (e.g. via `minimize`/`maximize`).

If you assemble the `Initializer` yourself, you need to pass it explicitly.

`Termination` accepts multiple conditions.

The run ends as soon as any one of the listed conditions is met (logical OR).

```python
from saealib import Termination, max_fe, max_gen

termination = Termination(max_fe(500), max_gen(200))
```

You can also add a custom condition with a lambda.

```python
termination = Termination(
    max_fe(500),
    lambda ctx: ctx.archive.get_array("f")[:, 0].min() < 1e-4,
)
```

Using `iterate()` instead of `run()` lets you obtain the context generation by generation.

This is useful for recording progress or implementing custom early stopping.

```python
optimizer = (
    Optimizer(problem, seed=SEED)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_strategy(strategy)
    .set_termination(termination)
)

history = []
for ctx in optimizer.iterate():
    best_f = ctx.archive.get_array("f")[:, 0].min()
    history.append((ctx.fe, best_f))
    print(f"gen={ctx.gen:4d}  fe={ctx.fe:4d}  best_f={best_f:.6f}")

print("Evaluations:", ctx.fe)
```

## References

- {py:func}`saealib.minimize` / {py:func}`saealib.maximize`
- {py:class}`saealib.Optimizer`
- {py:class}`saealib.GA` / {py:class}`saealib.PSO`
- {py:class}`saealib.IndividualBasedStrategy` / {py:class}`saealib.GenerationBasedStrategy` / {py:class}`saealib.PreSelectionStrategy`
- {py:class}`saealib.LocalSurrogateManager`
- {py:class}`saealib.RBFSurrogate`
- {py:class}`saealib.MeanPrediction`
- {py:class}`saealib.LHSInitializer`
- {py:class}`saealib.Termination` / {py:func}`saealib.max_fe` / {py:func}`saealib.max_gen`
