# Crossover

`GA` (`saealib.GA`) delegates generating offspring from selected parents to `Crossover`, a swappable operator.
To change the crossover scheme, you only need to swap out this `Crossover`, not `GA` itself.

## Crossover's role

`Crossover` requires only one method, `crossover(parent, bounds=None, rng=...)`, to be implemented.
`parent` is passed the array of parent individuals used for crossover.
By default, it receives `n_parents = 2` parents with shape `(n_parents, dim)` and returns `n_children = 2` children with shape `(n_children, dim)`.
To implement a crossover scheme other than 2-parents-2-children, override the `n_parents`/`n_children` class attributes in the subclass.
`bounds` receives the design variables' lower and upper bounds as a `(lb, ub)` tuple, or `None` for unbounded.

The individual-level probability deciding whether crossover is performed at all is judged by the caller, not inside `crossover()`.
`GA.ask()` only calls `crossover()` for pairs satisfying `ctx.rng.random() < self.crossover.prob`; pairs that don't satisfy it simply duplicate the parents as children.
In other words, while the `prob` class attribute itself is held by `Crossover`, deciding whether to perform crossover based on it is `GA`'s responsibility — `crossover()`'s implementation can always assume crossover is happening.

## Built-in Crossovers

| Class | Parameters | Characteristics |
|---|---|---|
| `CrossoverBLXAlpha` | `prob, alpha` | BLX-α crossover (introduced by Eshelman & Schaffer, 1993). Larger `alpha` lets children spread further outside the parents' value range |
| `CrossoverSBX` | `prob, eta, *, prob_var=0.5` | Simulated Binary Crossover {cite}`deb1995sbx`. Automatically switches to the bounded variant when `bounds` is finite. Larger `eta` brings children closer to the parents |
| `CrossoverUniform` | `prob, swap_rate=0.5` | Independently swaps each dimension between parents with probability `swap_rate` (introduced by Syswerda, 1989) |
| `CrossoverOnePoint` | `prob` | One-point crossover |
| `CrossoverTwoPoint` | `prob` | Two-point crossover |
| `CrossoverIntegerSBX` | `prob, eta, *, prob_var=0.5` | Performs the same computation as `CrossoverSBX` {cite}`deb1995sbx` and then rounds to an integer. For integer variables |
| `CrossoverCategorical` | `prob` | Copies each dimension's value directly from one parent or the other, 50/50. For categorical variables |

For problems with only continuous variables, pick one of these and pass it to `GA(crossover=..., ...)`.
The basic decision is: `CrossoverBLXAlpha`/`CrossoverUniform` work well for unconstrained problems, and `CrossoverSBX` when you need crossover that takes advantage of the bounds.

For problems where design variables mix integer and categorical variables, `GA` uses a different `Crossover` instance per variable type.
If the `GA` constructor's `integer_crossover`/`categorical_crossover` arguments are omitted, `CrossoverIntegerSBX`/`CrossoverCategorical` are supplied automatically (with `eta`/`prob` inherited from the continuous-variable `crossover`).
`GA.ask()` splits parent individuals into columns by variable type, applies each `Crossover` only to its corresponding columns, and then reassembles the results.
Because of this mechanism, if you pass a custom class to `integer_crossover`/`categorical_crossover`, its `n_children`/`n_parents` must match the continuous-variable `crossover`.
A mismatch raises `ConfigurationError`.

The correspondence between variable types and `Crossover` is determined by [Problem](problem.md)'s `variables` argument.

```{note}
Only `CrossoverBLXAlpha` is currently `@register()`ed; the other 6 classes are not yet registered with the Registry.
Keep this difference in mind if you resolve classes from strings via the Registry.
```

### External library adapters

`PymooCrossover(operator, *, prob=None, n_parents=None, n_children=None)` wraps an already-constructed [pymoo](https://pymoo.org/) crossover operator (e.g. `SBX()`) so existing pymoo-based research code can be reused unchanged inside `GA`.
`prob`/`n_parents`/`n_children` default to the wrapped operator's own values.

Because `crossover()` is called once per parent group while pymoo operators are written to vectorize over a whole population in one call, `PymooCrossover` calls the wrapped operator's `_do()` once per group — correct, but with per-call overhead a native saealib operator wouldn't pay.
`rng` is forwarded to the wrapped operator via pymoo's own `random_state` parameter, so results stay reproducible under saealib's seeding.

See [Installation](../getting_started/installation.md) for the `pymoo` extra.

## Extension hooks

If you just want to add post-processing, such as rounding values that fall outside the bounds, you can add it to an existing `Crossover` instance with `with_post(fn)` instead of creating a new subclass.
`with_post` doesn't modify the original instance — it returns a copy with `fn` added.

```python
import numpy as np
from saealib import CrossoverBLXAlpha

base = CrossoverBLXAlpha(prob=1.0, alpha=0.5)


def clip_to_bounds(offspring, parents, rng, ctx=None):
    return np.clip(offspring, -1.0, 1.0)


repaired = base.with_post(clip_to_bounds)
```

`fn`'s signature is `fn(offspring, parents, rng, ctx) -> np.ndarray`, receiving the result of the existing hook (by default an identity function that does nothing) and returning an additional transformation.
Calling `with_post` multiple times chains the hooks in the order called.

## Implementing a custom Crossover

If you need a custom crossover scheme, subclass `Crossover` and implement only `crossover()`.
The following example is a simple crossover that returns the average of two parents as both children, unchanged.

```python
import numpy as np
from saealib import Crossover


class AverageCrossover(Crossover):
    def __init__(self, prob: float = 1.0):
        super().__init__()
        self.prob = prob

    def crossover(self, parent, bounds=None, rng=np.random.default_rng()):
        mean = parent.mean(axis=0)
        return np.array([mean, mean])
```

If you want `n_parents`/`n_children` to be something other than 2, or want to add custom rounding logic that uses `bounds`, override the class attributes and add a reference to `bounds` inside `crossover()`.

## Related components

- [Algorithm](algorithm.md): How `GA` combines `Crossover`
- [Mutation](mutation.md): The paired operator called next, after crossover
- [ParentSelection](parent_selection.md): The operator that selects the parents passed to `Crossover`
- [Problem](problem.md): Defining integer and categorical variables, and their correspondence with mixed-variable Crossovers
- [Extension guidelines](extension_guidelines.md): The general design philosophy behind `with_post`-style hooks

## References

- {py:class}`saealib.Crossover`
- {py:class}`saealib.CrossoverBLXAlpha`
- {py:class}`saealib.CrossoverSBX`
- {py:class}`saealib.CrossoverUniform`
- {py:class}`saealib.CrossoverOnePoint`
- {py:class}`saealib.CrossoverTwoPoint`
- {py:class}`saealib.CrossoverIntegerSBX`
- {py:class}`saealib.CrossoverCategorical`
- {py:class}`saealib.PymooCrossover`
