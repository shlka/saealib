---
primary_layer: layer2
---

# Dominator

Comparators in the [ParetoComparator](comparators.md) family (`NSGA2Comparator`/`HypervolumeComparator`/`NSGA3Comparator`/`RNSGA2Comparator`/`EpsilonDominanceComparator`) delegate the definition of the dominance relation itself to `Dominator`, a swappable component.
When you want to use a preference relation other than Pareto dominance, you only need to swap out the `dominator` argument, not the Comparator itself.

## Dominator's role

`Dominator` requires only one method, `dominance_matrix(f, direction=None)`, to be implemented.
`f` is a matrix of objective values with shape `(n, n_obj)`; it returns a boolean matrix of shape `(n, n)` where `D[i, j]` indicates "does row `i` dominate row `j`."
`f` is assumed to contain no NaN; the caller is responsible for guaranteeing this.

There's also `dominates(fa, fb, direction=None) -> bool`, which compares just two points, but this is a default implementation derived from `dominance_matrix` — when implementing a custom `Dominator`, implementing `dominance_matrix` alone automatically guarantees consistency with `dominates`.
Internally, it stacks `fa`/`fb` into two rows and calls `dominance_matrix`.
If `fa` contains NaN, it always returns `False` (does not dominate), while NaN on the `fb` side is replaced with `+inf` and treated as an individual that "is always dominated" — an asymmetric treatment.

## Built-in Dominators

| Class | Parameters | Characteristics |
|---|---|---|
| `ParetoDominator` | None | Standard Pareto dominance. Considered dominance when at or below on every objective, and strictly less on at least one objective (default) |
| `EpsilonDominator` | `eps, mode="additive"` | Quantizes into ε-boxes and then delegates internally to `ParetoDominator` |

`EpsilonDominator` implements the ε-dominance of {cite}`laumanns2002epsilon`.
There are two quantization modes.

- **additive** (default): Computes each objective's box index as `floor(f_i / eps_i)`
- **multiplicative**: Computes it as `floor(log(f_i) / log(1 + eps_i))`. Assumes all objective values are positive; raises `ValueError` otherwise

`eps` is specified as a scalar or an array of shape `(n_obj,)`, and every element must be positive (otherwise `ValueError` at construction time).

[EpsilonDominanceComparator](comparators.md) is a thin wrapper that simply passes `EpsilonDominator(eps, mode)` to `ParetoComparator`'s `dominator` argument.

```{note}
For new code, use `ParetoDominator().dominates(...)`/`ParetoDominator().dominance_matrix(...)` directly.
```

## Implementing a custom Dominator

If you need a custom dominance relation, subclass `Dominator` and implement only `dominance_matrix()`.
The following example applies a scaling factor to each objective before applying Pareto dominance.

```python
import numpy as np
from saealib import Dominator, ParetoDominator


class WeightedDominator(Dominator):
    """Applies a scaling factor to each objective, then applies standard Pareto dominance."""

    def __init__(self, scale):
        self.scale = np.asarray(scale, dtype=float)
        self._pareto = ParetoDominator()

    def dominance_matrix(self, f, direction=None):
        return self._pareto.dominance_matrix(f * self.scale, direction)
```

As with `EpsilonDominator`, delegating to the existing `ParetoDominator` on the converted values means you don't have to write the objective-value comparison logic yourself.

## Related components

- [Comparator](comparators.md): Injects a `Dominator` via the `ParetoComparator` family's `dominator` argument
- [NonDominatedSorter](nondominated_sorting.md): The paired swap point that controls how individuals are sorted into fronts, alongside `Dominator`
- [Population](../observation_and_state/population.md): `ParetoArchive` uses `Dominator` to determine non-dominated solutions

## References

- {py:class}`saealib.Dominator`
- {py:class}`saealib.ParetoDominator`
- {py:class}`saealib.EpsilonDominator`
