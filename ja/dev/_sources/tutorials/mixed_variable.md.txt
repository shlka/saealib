---
primary_layer: layer1
related_layers: [layer2]
page_type: tutorial
---

# Mixed-Variable Optimization

Solve problems that include not only continuous variables but also integer and categorical variables, using `saealib`.

With `minimize()`, you can specify strings or instances for the algorithm, surrogate model, and evaluation strategy.
The examples on this page add only problem-specific settings and use defaults for everything else.

:::{admonition} What you'll be able to do
:class: tip

By the end of this page, you'll be able to define continuous, integer, and categorical variables in a `Problem` and run type-aware search.
:::

For ordinary continuous variables, use the [high-level API](highlevel_api.md); to add custom variables or operators, see [Custom components](custom_components.md).
This page defines variable types in a `Problem` and runs search with type-aware built-in operators.

## Set up the problem

Assume an objective function where each design variable is one of three types: continuous, integer, or categorical.

```python
def func(x):
    # x[0]: continuous, x[1]: integer, x[2]: categorical index
    return x[0] ** 2 + (x[1] - 3) ** 2 + (0.0 if x[2] == 1 else 5.0)
```

`x[2]` represents one of three categories (`0`, `1`, `2`), and no penalty is applied only when `1` is chosen.

## Define variables

Each dimension's type is specified with a subclass of `Variable`, passed to `Problem`'s `variables` argument.

| Class | Meaning |
|--------|------|
| `ContinuousVariable(lb, ub)` | Continuous value. A regular `Problem` with `lb`/`ub` omitted is treated as this type |
| `IntegerVariable(lb, ub)` | Integer value |
| `CategoricalVariable(categories)` | Selects one from a list of categories |

```python
import numpy as np
from saealib import Problem, ContinuousVariable, IntegerVariable, CategoricalVariable

variables = [
    ContinuousVariable(-5.0, 5.0),
    IntegerVariable(0, 10),
    CategoricalVariable([0, 1, 2]),
]

problem = Problem(
    func=func,
    dim=3,
    n_obj=1,
    direction=np.array([-1.0]),
    variables=variables,
)
```

Passing `variables` automatically derives `Problem.lb`/`Problem.ub` from each element's `lb`/`ub`, so you don't specify the `lb`/`ub` arguments.

## Run minimize

```python
from saealib import minimize

result = minimize(problem, max_fe=500, seed=0)
print(result.x, result.f)
```

`GA` comes with dedicated crossover and mutation operators for each variable type by default (`CrossoverIntegerSBX`/`MutationIntegerUniform` for integers, `CrossoverCategorical`/`MutationCategorical` for categorical variables), so passing `variables` alone gives you type-aware search.

`PSO` is a velocity-based update scheme and cannot correctly handle integer and categorical variables.

For mixed-variable problems, stick with `algorithm='GA'` (the default).

## Customize operators

The operators for integer and categorical variables can also be specified individually via `GA`'s keyword arguments.

```python
from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
    CrossoverIntegerSBX,
    CrossoverCategorical,
    MutationIntegerUniform,
    MutationCategorical,
)

ga = GA(
    crossover=CrossoverBLXAlpha(0.7, 0.4),
    mutation=MutationUniform(0.3),
    parent_selection=SequentialSelection(),
    survivor_selection=TruncationSelection(),
    integer_crossover=CrossoverIntegerSBX(0.7, eta=15.0),
    integer_mutation=MutationIntegerUniform(0.3),
    categorical_crossover=CrossoverCategorical(0.7),
    categorical_mutation=MutationCategorical(0.3),
)

result = minimize(problem, algorithm=ga, max_fe=500, seed=0)
```

See [Algorithm](../concepts/search_algorithms/algorithm.md) for details on each operator.

## Related concepts and reference

- {py:class}`saealib.Problem`
- {py:class}`saealib.ContinuousVariable` / {py:class}`saealib.IntegerVariable` / {py:class}`saealib.CategoricalVariable`
- {py:class}`saealib.GA`
- {py:func}`saealib.minimize`
