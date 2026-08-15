---
primary_layer: layer3
page_type: concept
---

# SurvivorSelection

`GA` (`saealib.GA`) delegates choosing which individuals survive into the next generation from a selection pool to `SurvivorSelection`, a swappable operator.
To change the generational-replacement scheme, you only need to swap out this `SurvivorSelection`, not `GA` itself.

## SurvivorSelection's role

`SurvivorSelection` requires only one method, `select(ctx, pool, n_survivors) -> np.ndarray`, to be implemented.
`pool` is a combined `Population` built by `Algorithm` — for example, the parent and offspring populations together — and it returns the indices of the `n_survivors` individuals that survive from it.

Whether it's a $(\mu+\lambda)$ scheme (selecting from a pool of parents and offspring combined) or a $(\mu,\lambda)$ scheme (selecting from a pool of offspring only) is determined by how `Algorithm` builds `pool` — that is, what it includes.
`SurvivorSelection`'s interface itself doesn't distinguish between the two schemes.

## Built-in SurvivorSelections

| Class | Parameters | Characteristics |
|---|---|---|
| `TruncationSelection` | `randomize_ties=False` | Truncation selection: sorts via `ctx.comparator.sort_population(pool)` and takes the top `n_survivors` |

Setting `randomize_ties=True` shuffles individuals tied at the truncation boundary (where `compare_population` returns `0`) before truncating.
With the default `False`, it's a deterministic truncation that uses the order returned by `sort_population` as-is.
Because this tie-breaking consumes `ctx.rng`, note that using `randomize_ties=True` also affects the random state at checkpoint resumption.

`TruncationSelection` is `@register()`ed.

## Implementing a custom SurvivorSelection

If you need a custom generational-replacement scheme, subclass `SurvivorSelection` and implement only `select()`.
The following example is a survivor selection that always keeps the single best individual, choosing the rest at random.

```python
import numpy as np
from saealib import SurvivorSelection


class ElitistSurvivorSelection(SurvivorSelection):
    """Always keeps the single best individual, choosing the rest at random."""

    def select(self, ctx, pool, n_survivors):
        sorted_idx = ctx.comparator.sort_population(pool)
        best = sorted_idx[:1]
        rest_pool = sorted_idx[1:]
        rest = ctx.rng.choice(rest_pool, size=n_survivors - 1, replace=False)
        return np.concatenate([best, rest])
```

Schemes that don't assume a ranking from `sort_population`, such as tournament-style survivor selection or age-based replacement, can also be implemented within the same `select()` signature.

## Related components

- [Algorithm](algorithm.md): How `GA.tell()` builds `pool` and calls `SurvivorSelection`
- [Comparator](../problem_and_ranking/comparators.md): Ranking individuals via `sort_population`/`compare_population`

## References

- {py:class}`saealib.SurvivorSelection`
- {py:class}`saealib.TruncationSelection`
