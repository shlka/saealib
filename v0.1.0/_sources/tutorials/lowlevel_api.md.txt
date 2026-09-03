---
primary_layer: layer2
page_type: guide
---

# Assembling an Optimizer with the Low-Level API

This page assumes you can express your objective function as a `Problem`.

:::{admonition} What you'll be able to do
:class: tip

By the end of this page, you'll be able to register the components the defaults don't cover on an `Optimizer` and run it with `run()` or `iterate()`.
:::

Move to the low-level API when a single run of the high-level API isn't enough.
If you only need to pick individual built-in components, start with [Swapping Built-in Components](component_swap.md); come here when you need per-generation state or want to swap components mid-run.
The import policy is collected in [Canonical Imports](../api/imports.md).

## Construct an Optimizer

Create an `Optimizer(problem)` and chain `set_*()` calls to configure the execution components.
`set_*()` is a chainable configuration API and does nothing but configure.
`run()` or `iterate()` resolves the defaults, builds the execution plan, and validates the configuration before the run begins.

```python
import numpy as np
from saealib import Optimizer, Problem, Termination, max_fe


def objective(x):
    return np.sum(x**2)


problem = Problem(
    objective,
    dim=3,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0] * 3,
    ub=[5.0] * 3,
)

optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(100)))
```

Add only the components you need, with `set_initializer()`, `set_algorithm()`, `set_surrogate_manager()`, `set_acquisition()`, `set_strategy()`, `set_evaluator()`, and so on.
Each `set_*()` returns the `Optimizer` itself, so the calls chain.

## Assemble a full configuration with a surrogate

`minimize` wires up each component with a default combination, but doesn't let you tune individual parameters.
Instantiating components individually and assembling them into `Optimizer` removes this limitation.

`LocalSurrogateManager` takes the surrogate and how its training data is built; the acquisition function is configured separately, with `Optimizer.set_acquisition()`.
Do not treat the two as the same constructor argument.

```python
import numpy as np

from saealib import (
    GA,
    IndividualBasedStrategy,
    LHSInitializer,
    Optimizer,
    Problem,
    Termination,
    max_fe,
)
from saealib.acquisition import MeanPrediction
from saealib.operators import (
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.surrogate import GaussianKernel, LocalSurrogateManager, RBFSurrogate

DIM = 10
SEED = 0

problem = Problem(
    objective,
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
    RBFSurrogate(kernel=GaussianKernel()),
)
strategy = IndividualBasedStrategy(evaluation_ratio=0.1)
initializer = LHSInitializer(
    n_init_archive=5 * DIM,
    n_init_population=4 * DIM,
    seed=SEED,
)
termination = Termination(max_fe(500))

optimizer = (
    Optimizer(problem, seed=SEED)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_acquisition(MeanPrediction())
    .set_strategy(strategy)
    .set_termination(termination)
)

ctx = optimizer.run()
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

## Combine termination conditions

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

## Choose between run and iterate

`run()` runs to termination and returns the final `OptimizationState`.
`iterate()` yields the state generation by generation, so you can record history, swap components conditionally, or implement your own early stopping inside the loop.

```{mermaid}
flowchart LR
    A["Configure the Optimizer"] --> B["run or iterate"]
    B --> C["Resolve defaults, build plan, validate"]
    C --> D["Run a generation"]
    D --> E{"Termination met?"}
    E -- No --> D
    E -- Yes --> F["OptimizationState"]
```

```python
ctx = optimizer.run()
print(ctx.fe, ctx.gen)
```

Use `iterate()` instead of `run()` when you need the state generation by generation.

```python
history = []
for ctx in optimizer.iterate():
    best_f = ctx.archive.get_array("f")[:, 0].min()
    history.append((ctx.gen, ctx.fe, float(best_f)))
    print(f"gen={ctx.gen:4d}  fe={ctx.fe:4d}  best_f={best_f:.6f}")

print("Evaluations:", ctx.fe)
```

`ctx.archive` holds the evaluated solutions, and `ctx.fe` and `ctx.gen` give the evaluation count and the number of completed generations.
To swap the strategy or the surrogate manager mid-run, read each generation's `ctx` and call `optimizer.set_*()`. The execution environment detects the change, recompiles the plan, and applies it from the next generation onward.
This works on both the Stage compatibility path and the graph-native path.
For the path by which a component itself requests a recompile, see "Behavior of runtime swapping" in [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md).
Saving and resuming a checkpoint is left to [Checkpointing](checkpoint.md).

## Related concepts and reference

- [Stage](../concepts/observation_and_state/stage.md)
- [OptimizationState](../concepts/observation_and_state/optimization_state.md)
- [Optimizer reference](../api/optimizer.md)
- [Termination reference](../api/termination.md)
- {py:class}`saealib.LocalSurrogateManager`
- {py:class}`saealib.RBFSurrogate`
- {py:class}`saealib.MeanPrediction`
- {py:class}`saealib.LHSInitializer`
- {py:class}`saealib.IndividualBasedStrategy`
- {py:class}`saealib.Termination` / {py:func}`saealib.max_fe` / {py:func}`saealib.max_gen`
