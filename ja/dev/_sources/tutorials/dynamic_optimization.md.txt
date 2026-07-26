# Dynamic Switching Based on Surrogate Accuracy

The prediction accuracy of a surrogate model changes as generations progress.

Early on, when training data is scarce, accuracy is low; as the archive fills in, accuracy rises; and once the search converges and the gap between objective values narrows, accuracy can drop again.

This page covers how to switch the evaluation strategy or the `SurrogateManager` at runtime in response to these changes.

## Problem setup

We use the same Sphere function as in the single-objective optimization tutorial.

```python
import numpy as np


def expensive_func(x):
    return np.sum(x**2)


DIM = 10
SEED = 0
```

## Tracking surrogate accuracy

Passing an `accuracy_evaluator` to `SurrogateManager` computes accuracy on every fit and records it in `surrogate_manager.last_accuracy`.

`LOOAccuracyEvaluator` computes accuracy via leave-one-out cross-validation on the current training data, without preparing any additional held-out data.

```python
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
    LHSInitializer,
    Termination,
    max_fe,
    LOOAccuracyEvaluator,
)

problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
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
    MeanPrediction(),
    accuracy_evaluator=LOOAccuracyEvaluator(),
)

initializer = LHSInitializer(
    n_init_archive=5 * DIM, n_init_population=4 * DIM, seed=SEED
)
```

`last_accuracy` is a `SurrogateAccuracy` instance, from which you can retrieve a value by specifying a metric name, e.g. `.get("spearman")`.

Note that `last_accuracy` is `None` in the first generation.

## Switching the evaluation strategy with StrategySwitcher

`StrategySwitcher(primary, fallback, metric="spearman", threshold=0.56)` returns `primary` if accuracy is at or above the threshold, and `fallback` otherwise.

The strategy returned by `switch()` takes effect from the next generation onward by passing it to `optimizer.set_strategy(...)` inside the `iterate()` loop.

```python
from saealib import PreSelectionStrategy, IndividualBasedStrategy, StrategySwitcher

ps_strategy = PreSelectionStrategy(n_candidates=40, n_select=4)
ib_strategy = IndividualBasedStrategy(evaluation_ratio=0.1)
switcher = StrategySwitcher(primary=ps_strategy, fallback=ib_strategy)

optimizer = (
    Optimizer(problem, seed=SEED)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_strategy(ib_strategy)
    .set_termination(Termination(max_fe(600)))
)

for ctx in optimizer.iterate():
    accuracy = optimizer.surrogate_manager.last_accuracy
    optimizer.set_strategy(switcher.switch(accuracy))

print(ctx.fe)
```

Running this, the search stays on `IndividualBasedStrategy` early on while accuracy is low, and switches to `PreSelectionStrategy` once accuracy exceeds the threshold.

If the search progresses, the gap between objective values narrows, and accuracy drops again, it switches back to `IndividualBasedStrategy`.

## Other switchers

Switchers with the same `switch()` interface are available besides `StrategySwitcher`.

| Class | What it switches |
|--------|--------|
| `StrategySwitcher` | Between two `OptimizationStrategy` instances (whether accuracy is at or above the threshold) |
| `ManagerSwitcher` | Between two `SurrogateManager` instances (whether accuracy is at or above the threshold) |
| `GenCtrlSwitcher` | `GenerationBasedStrategy`'s `gen_ctrl` (smooths accuracy with an exponential moving average and maps it to a continuous value) |

`GenCtrlSwitcher` returns a number, so assign it directly to the `gen_ctrl` attribute of a `GenerationBasedStrategy` instance.

```python
from saealib import GenerationBasedStrategy, GenCtrlSwitcher

gen_ctrl_switcher = GenCtrlSwitcher(gm_max=5, gm_min=0)
strategy = GenerationBasedStrategy(gen_ctrl=gen_ctrl_switcher.switch(None))

optimizer.set_strategy(strategy)
for ctx in optimizer.iterate():
    accuracy = optimizer.surrogate_manager.last_accuracy
    strategy.gen_ctrl = gen_ctrl_switcher.switch(accuracy)
```

## References

- {py:class}`saealib.StrategySwitcher` / {py:class}`saealib.ManagerSwitcher` / {py:class}`saealib.GenCtrlSwitcher`
- {py:class}`saealib.LOOAccuracyEvaluator` / {py:class}`saealib.HeldOutAccuracyEvaluator` / {py:class}`saealib.KFoldAccuracyEvaluator`
- {py:class}`saealib.SurrogateAccuracy`
- {py:class}`saealib.Optimizer`
