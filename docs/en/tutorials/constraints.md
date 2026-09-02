---
primary_layer: layer1
related_layers: [layer2, layer3]
page_type: tutorial
---

# Constrained Optimization

Solve a problem with constraints on the design variables using `saealib`.

With `minimize()`, you can pass a string or instance for the algorithm, surrogate, and evaluation strategy.
The examples on this page add only problem-specific settings and use defaults for everything else.

:::{admonition} What you'll be able to do
:class: tip

By the end of this page, you'll be able to define constraints on a `Problem`, check feasibility, and choose a `ConstraintHandler`.
:::

For a simple objective-function optimization, use [Single-objective optimization](single_objective.md). To implement custom constraint handling, see [Custom components](custom_components.md).
This page covers defining constraints on a `Problem`, checking feasibility, and choosing a `ConstraintHandler`.

## Set up the problem

Assume a problem that, in addition to the objective function, has an inequality constraint `g(x) <= 0` that the solution must satisfy.

```python
import numpy as np


def expensive_func(x):
    return np.sum(x**2)


def g1(x):
    # require the sum of the design variables to be at least 1
    return 1.0 - np.sum(x)


DIM = 5
LB = [-5.0] * DIM
UB = [5.0] * DIM
```

Only solutions satisfying `g1(x) <= 0` are feasible.

## Define constraints

A constraint is defined with `InequalityConstraint(func, threshold=0.0)` and passed to `Problem`'s `constraints` argument.

```python
from saealib import InequalityConstraint, Problem, minimize

constraint = InequalityConstraint(g1, threshold=0.0)

problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=LB,
    ub=UB,
    constraints=[constraint],
)

result = minimize(problem, max_fe=1000, seed=0)
print(result.x, result.f)
print(constraint.violation(result.x))  # 0.0 means the constraint is satisfied
```

To impose a constraint of the form `g(x) >= threshold`, pass `func` with its sign flipped.

For an equality constraint `h(x) = 0`, use `EqualityConstraint(func, tolerance=1e-6)` instead of combining two sign-flipped `InequalityConstraint`s.

```python
from saealib import EqualityConstraint


def h1(x):
    # require the sum of the design variables to be exactly 1
    return np.sum(x) - 1.0


equality = EqualityConstraint(h1, tolerance=1e-6)
```

`EqualityConstraint` treats solutions satisfying `|h(x)| <= tolerance` as feasible.

## Check feasibility

A solution is feasible if its constraint violation `cv` is at most `Problem`'s `eps_cv` (default `1e-6`).

```python
archive_cv = result.ctx.archive.get_array("cv")
feasible = archive_cv <= problem.eps_cv
print(f"feasible: {feasible.sum()} / {len(archive_cv)}")
```

## Customize constraint handling with ConstraintHandler

How `cv` is aggregated from multiple constraints, and how feasibility is judged from that value, is decided by `ConstraintHandler`.

If `Problem`'s `handler` argument is omitted, `StaticToleranceHandler(eps_cv=problem.eps_cv)` is used, which judges feasibility by whether `cv <= eps_cv`.

| Class | Behavior |
|--------|------|
| `StaticToleranceHandler` | Judges feasibility with a fixed tolerance `eps_cv` (default) |
| `EpsilonConstraintHandler` | Shrinks the tolerance toward 0 over generations, gradually driving solutions into the feasible region |
| `GradientRepairHandler` | Repairs infeasible solutions using the constraint gradient |

`EpsilonConstraintHandler` takes a function that receives the generation number and returns the tolerance.

```python
from saealib import EpsilonConstraintHandler


def schedule(gen):
    return max(0.0, 1.0 - gen * 0.05)


problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=LB,
    ub=UB,
    constraints=[constraint],
    handler=EpsilonConstraintHandler(schedule),
)
result = minimize(problem, max_fe=1000, seed=0)
```

## Related concepts and reference

- {py:class}`saealib.InequalityConstraint` / {py:class}`saealib.EqualityConstraint`
- {py:class}`saealib.Problem`
- {py:class}`saealib.ConstraintHandler` / {py:class}`saealib.StaticToleranceHandler` / {py:class}`saealib.EpsilonConstraintHandler` / {py:class}`saealib.GradientRepairHandler`
- {py:func}`saealib.minimize`
