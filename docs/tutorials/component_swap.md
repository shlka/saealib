---
primary_layer: layer2
related_layers: []
page_type: guide
---

# Choose and swap built-in components

The prerequisite is using the high-level API or an `Optimizer` execution path.
:::{admonition} What you'll be able to do
:class: tip

By the end of this page, you'll be able to choose the built-in component for the responsibility you want to change and pass it to the appropriate swap point.
:::

See the [optimization components](../concepts/index.md) index; if you need a custom component, see [Custom components](custom_components.md).

The [Canonical Imports](../api/imports.md) page summarizes import choices.

## Choose by responsibility

Components are swap points for choosing another implementation of the same operation.
`AcquisitionFunction` and `SurrogateManager` are independent swap points; neither contains the other.

| What to change | Component to swap | Where to look |
|---|---|---|
| Search method that generates candidates | `Algorithm` | [Algorithm concept](../concepts/search_algorithms/algorithm.md), [Algorithm reference](../api/algorithms.md) |
| Crossover, mutation, and selection | `Operator` | [Crossover concept](../concepts/search_algorithms/crossover.md), [Mutation concept](../concepts/search_algorithms/mutation.md), [Operator reference](../api/operators.md) |
| Prediction model | `Surrogate` | [Surrogate concept](../concepts/surrogate_modeling/surrogate.md), [Surrogate reference](../api/surrogate.md) |
| Training data and prediction process | `SurrogateManager` | [SurrogateManager concept](../concepts/surrogate_modeling/surrogate_manager.md), [Reference](../api/surrogate.md) |
| Rule that converts predictions into candidate scores | `AcquisitionFunction` | [AcquisitionFunction concept](../concepts/surrogate_modeling/acquisition_functions.md), [Reference](../api/acquisition.md) |
| Fraction or order of surrogate candidates sent for true evaluation | `Strategy` | [Strategy concept](../concepts/execution_and_evaluation/strategies.md), [Reference](../api/strategies.md) |
| Solution ranking | `Comparator` | [Comparator concept](../concepts/problem_and_ranking/comparators.md), [Reference](../api/comparators.md) |

## Swap through the high-level API

The high-level API accepts strings or instances for `algorithm`, `surrogate`, and `strategy`.
The available strings are `"GA"` and `"PSO"` for algorithms, `"rbf"` for Surrogate, and `"ib"`, `"gb"`, and `"ps"` for Strategy.

```python
import numpy as np
from saealib import minimize


def objective(x):
    return np.sum(x**2)

result = minimize(
    objective,
    dim=3,
    lb=[-5.0] * 3,
    ub=[5.0] * 3,
    algorithm="PSO",
    strategy="ib",
    max_fe=100,
    seed=0,
    verbose=False,
)
```

A standalone Surrogate instance is wrapped in a `LocalSurrogateManager` internally.
Use `Optimizer` when configuring a Manager or AcquisitionFunction separately.

## Configure independent swap points with Optimizer

`Optimizer.set_*()` is a builder for configuring components independently.
For example, `set_surrogate_manager()` and `set_acquisition()` are separate calls.

```python
import numpy as np
from saealib import MeanPrediction, Optimizer, Problem
from saealib.surrogate import GlobalSurrogateManager, RBFSurrogate, gaussian_kernel


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
manager = GlobalSurrogateManager(RBFSurrogate(gaussian_kernel, dim=3))

optimizer = (
    Optimizer(problem, seed=0)
    .set_surrogate_manager(manager)
    .set_acquisition(MeanPrediction())
)
```

Other components required by `Optimizer` can be resolved from defaults.
Components configured with `set_*()` and omitted components are validated after `run()` or `iterate()` resolves the defaults.
When every required component is explicit, you can also call `validate()` directly before running.
Continue to [Implement custom components](custom_components.md) only when built-in components cannot meet the responsibility.

## Related concepts and reference

- [Optimization components overview](../concepts/index.md)
- [Optimizer reference](../api/optimizer.md)
- [Registry reference](../api/registry.md)
