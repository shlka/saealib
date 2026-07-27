# ConstraintHandler

The basic way to define constraints (`InequalityConstraint`/`EqualityConstraint`), how to specify `Problem`'s `handler` argument, and how to choose among the three built-in handlers are covered in [Constrained Optimization](../tutorials/constraints.md).
This page focuses on writing a custom `ConstraintHandler`, building on that foundation.

## ConstraintHandler's role

`ConstraintHandler` exposes its handling of constraints as a set of swappable lifecycle hooks.

```
Ask            -> [repair(x, constraints, lb, ub)]
               -> evaluate f, g
               -> [compute_cv(constraints, x, g)]        -> cv
               -> [augment_objective(f, constraints, x, g)] -> f'
Tell           -> Comparator(f', cv) with eps_cv = feasibility_threshold
Generation end -> [on_generation_end(gen, population)]
```

Only `compute_cv` is abstract; every other hook has a default implementation.
A custom `ConstraintHandler` only needs to override the hooks it needs.

**`repair(x, constraints, lb, ub, **kwargs)`**: Called after crossover and mutation, before evaluation.
The default is `np.clip(x, lb, ub)` — simply clipping to the bounds.

**`compute_cv(constraints, x, g)`** (abstract): Aggregates the set of constraints into a single `cv` value.

**`augment_objective(f, constraints, x, g)`**: Transforms the objective value using constraint information.
The default is the identity function (no transformation) — this is where you'd override to implement a penalty-function method or the augmented Lagrangian method.
None of the three built-in handlers override this hook, so it's currently left open as a future extension point.

**`feasibility_threshold`** (property): Defaults to `1e-6`.
This value is used as `Comparator`'s `eps_cv`, and is synchronized every generation during an `Optimizer` run.

**`on_generation_end(gen, population)`**: Called at the end of a generation.
The default is a no-op; used by handlers with internal state that updates an ε value per generation, such as `EpsilonConstraintHandler`.

## Hooks overridden by the built-in handlers

| Class | Hooks overridden |
|---|---|
| `StaticToleranceHandler` | `compute_cv` / `feasibility_threshold` |
| `EpsilonConstraintHandler` | `compute_cv` / `feasibility_threshold` / `on_generation_end` |
| `GradientRepairHandler` | `repair` / `compute_cv` |

`EpsilonConstraintHandler` {cite}`mezuramontes2011epsilon` updates ε per generation via a function `schedule(gen) -> float`.
Built-in schedule-generating functions `linear_epsilon_schedule(eps0, n_gen)`/`exponential_epsilon_schedule(eps0, decay)` are also exposed.

`GradientRepairHandler` {cite}`chootinan2006gradientrepair` overrides `repair()` with a simultaneous Moore-Penrose pseudoinverse update over the currently-violated constraints (equality and inequality alike).
Constraints whose `gradient()` returns `None` fall back to a forward-difference numerical approximation rather than being skipped.

## Extension points on InequalityConstraint/EqualityConstraint

`InequalityConstraint` itself also has two extension points.

**`gradient(x)`**: Returns `None` by default.
Override it to return an analytical gradient vector; otherwise `GradientRepairHandler` falls back to a numerical approximation.

**`violation_from_value(g)`**: Defines the conversion from the raw constraint value `g(x)` to a violation amount.
The default is `max(0, g - threshold)`.
`EqualityConstraint` overrides only this method, defining its own conversion, `max(0, |h(x)| - tolerance)`.

## Implementing a custom ConstraintHandler

If you need a custom constraint-handling strategy, subclass `ConstraintHandler` and implement only `compute_cv()`.
The following example is a penalty-function method that adds the constraint violation to the objective value.

```python
from saealib import ConstraintHandler


class PenaltyHandler(ConstraintHandler):
    """Adds the violation amount to the objective value as a penalty."""

    def __init__(self, penalty_coeff: float = 1e3):
        self.penalty_coeff = penalty_coeff

    def compute_cv(self, constraints, x, g):
        return sum(max(0.0, gi) for gi in g)

    def augment_objective(self, f, constraints, x, g):
        cv = self.compute_cv(constraints, x, g)
        return f + self.penalty_coeff * cv
```

While overriding `augment_objective` to add the penalty to the objective value, you can also implement `compute_cv` to always return `0` as the constraint violation, giving a configuration that handles infeasibility purely through the penalty-function method (treating every solution as feasible).

## Related components

- [Constrained Optimization](../tutorials/constraints.md): Defining constraints and basic usage of the built-in handlers
- [Problem](problem.md): How to pass the `handler` argument
- [Comparator](comparators.md): Where `feasibility_threshold` is synchronized to, as `eps_cv`

## References

- {py:class}`saealib.ConstraintHandler`
- {py:class}`saealib.StaticToleranceHandler`
- {py:class}`saealib.EpsilonConstraintHandler`
- {py:class}`saealib.GradientRepairHandler`
- {py:class}`saealib.InequalityConstraint`
- {py:class}`saealib.EqualityConstraint`
- {py:func}`saealib.linear_epsilon_schedule`
- {py:func}`saealib.exponential_epsilon_schedule`
