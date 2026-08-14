---
primary_layer: layer2
---

# NonDominatedSorter

Comparators in the [ParetoComparator](comparators.md) family (`NSGA2Comparator`/`HypervolumeComparator`/`NSGA3Comparator`/`RNSGA2Comparator`/`EpsilonDominanceComparator`) delegate sorting the population into fronts to `sorter`, a swappable argument.
Where [Dominator](dominance.md) handles "the definition of the dominance relation between two points," `sorter` is a second, independent swap axis: "how to use that dominance relation to assign individuals to fronts."

## NonDominatedSorter's role

`NonDominatedSorter` is defined not as an abstract base class but as a `Protocol` (structural subtyping).
Any callable satisfying the signature `__call__(f, direction=None, *, dominator=None) -> tuple[ranks, fronts]` can be passed as `sorter` without inheriting from any class.

`ranks` is an array giving each individual's front number (`0` is best); `fronts` satisfies the contract that `fronts[i]` is the list of individual indices belonging to front `i`.

## Built-in NonDominatedSorters

`non_dominated_sort`/`dda_non_dominated_sort` are both functions satisfying the `(ranks, fronts)` contract above (with no class hierarchy of their own).

| Function | Algorithm |
|---|---|
| `non_dominated_sort` | {cite}`deb2002nsga2`'s non-dominated sorting. A front-peeling approach that strips off one front at a time |
| `dda_non_dominated_sort` | The Dominance-Degree Approach (Zhou et al., 2017 proposed the dominance degree matrix; Mishra & Senwar, 2020 proposed DDA-ENS's front assignment) |

`non_dominated_sort`'s complexity is O(MN²), but because the dominance matrix is built vectorized with NumPy, it runs fast in practice.
`dda_non_dominated_sort` is guaranteed to return exactly the same `(ranks, fronts)` as `non_dominated_sort`, making it a drop-in alternative provided for scalability when the number of individuals `N` or objectives `M` is large (`M > 100`).

Both handle rows containing NaN the same way.
Rows containing NaN are excluded from normal front splitting, and are appended one individual at a time as a sentinel front after the final front.

`sorter` and `dominator` are independent swap axes.
Both `non_dominated_sort`/`dda_non_dominated_sort` merely call `dominator.dominance_matrix()` internally, and take no part in defining the dominance relation itself.
If `dominator` is omitted, `ParetoDominator` is used as the default.

## Auxiliary functions

Auxiliary functions used inside the internal implementation of Pareto-family Comparators are also available standalone as public API.

**`crowding_distance(f_front)`**: Computes crowding distance within a single front. Used by `NSGA2Comparator`.
Boundary solutions (those achieving the minimum or maximum of each objective) are assigned `inf`.

**`crowding_distance_all_fronts(f, fronts)`**: Applies `crowding_distance` to every front returned by `non_dominated_sort`.

**`spea2_fitness(f, direction=None, dominator=None)`**: The fitness computation {cite}`zitzler2001spea2` used by `SPEA2Comparator`.
Computed from three components: strength (the number of individuals dominated), raw fitness (the sum of the strengths of the individuals that dominate it), and density (the reciprocal of the $k$-nearest-neighbor distance).

```{warning}
`spea2_fitness`'s return value follows the convention that lower is better, the opposite of saealib's overall "higher is better" score convention.
Be careful not to pass it directly to another Comparator.
```

## How to extend NonDominatedSorter

Because `NonDominatedSorter` is a Protocol, writing a single function that satisfies the `(ranks, fronts)` contract is enough to use as a custom implementation.
There's no need to inherit from a base class.

```python
import numpy as np
from saealib import non_dominated_sort


def logged_non_dominated_sort(f, direction=None, *, dominator=None):
    """The minimal function satisfying the Protocol, simply adding pre-processing to the existing implementation."""
    print(f"sorting {len(f)} individuals")
    return non_dominated_sort(f, direction, dominator=dominator)
```

Passing it as `ParetoComparator(sorter=logged_non_dominated_sort, ...)` lets you swap the sorting scheme without changing an existing `Comparator` implementation.

## Related components

- [Dominator](dominance.md): The definition of the dominance relation between two points, paired with `sorter`
- [Comparator](comparators.md): The `ParetoComparator` family of Comparators, which have a `sorter` argument

## References

- {py:class}`saealib.NonDominatedSorter`
- {py:func}`saealib.non_dominated_sort`
- {py:func}`saealib.dda_non_dominated_sort`
- {py:func}`saealib.crowding_distance`
- {py:func}`saealib.crowding_distance_all_fronts`
- {py:func}`saealib.spea2_fitness`
