---
primary_layer: layer2
---

# Comparator

`Problem` delegates deciding the relative superiority of solutions to `Comparator`, a swappable top-level component.
Pass it via `Problem`'s `comparator` argument.

## Comparator's role

`Comparator` has four methods that are all abstract, including `__init__` itself.

- **`__init__(weights, eps_cv, eps_obj, direction=None)`**: Holds `weights`/`eps_cv`/`eps_obj`/`direction`
- **`sort_population(population) -> np.ndarray`**: Returns an index array ordering the entire population from best to worst
- **`compare_population(population, idx_a, idx_b) -> int`**: Compares two individuals within a population (`-1`: a is better, `1`: b is better, `0`: equal)
- **`compare(fa, cv_a, fb, cv_b) -> int`**: A lightweight version that compares two points directly from their objective values and constraint violation, without going through `Population`

Having `__init__` be an abstract method is different from other components.
Subclasses implementing a custom `Comparator` must always call `super().__init__(weights, eps_cv, eps_obj, direction=...)`.

## Built-in Comparators

| Class | When to use |
|---|---|
| `SingleObjectiveComparator` | Single-objective problems |
| `WeightedSumComparator` | Scalarization via weighted linear combination. Works for both single- and multi-objective |
| `ParetoComparator` | Ranks by dominance relation only (no secondary metric such as crowding) |
| `NSGA2Comparator` | Pareto rank + crowding distance {cite}`deb2002nsga2` |
| `SPEA2Comparator` | SPEA2 fitness, which depends on the entire population {cite}`zitzler2001spea2` |
| `HypervolumeComparator` | Front rank + exclusive HV contribution (SMS-EMOA style) {cite}`beume2007smsemoa` |
| `EpsilonDominanceComparator` | Ranking via ε-dominance {cite}`laumanns2002epsilon` |
| `NSGA3Comparator` | Niche preservation via reference points {cite}`deb2014nsga3` |
| `RNSGA2Comparator` | Preference guidance via user-specified reference points {cite}`deb2006rnsga2` |

`SingleObjectiveComparator(direction=None, *, eps_cv=1e-6, eps_obj=1e-6)` allows `direction` to be omitted, in which case it's treated as minimization.

`WeightedSumComparator(direction, *, eps_cv=1e-6, eps_obj=1e-6)` requires `direction`; omitting it raises `TypeError`.
This class is the sole exception where the passed `direction` is used directly as the scalarization weights (`score = f @ direction`).
Not just the sign but also the magnitude functions as a weight, which is a treatment specific to this class, different from the general role split described in [Problem](problem.md) where "`direction` is sign only, and weight magnitude is a separate concept."

`ParetoComparator(direction=None, *, eps_cv=1e-6, eps_obj=1e-6, sorter=non_dominated_sort, dominator=None)` ranks the population by dominance relation alone.
It's the common base of `NSGA2Comparator`/`HypervolumeComparator`/`NSGA3Comparator`/`RNSGA2Comparator`/`EpsilonDominanceComparator`, and can also be used standalone as a concrete class.
The `dominator` argument injects a [Dominator](dominance.md) and the `sorter` argument injects a [NonDominatedSorter](nondominated_sorting.md) — independent swap points from each other.

`NSGA2Comparator` adds secondary ranking by crowding distance on top of `ParetoComparator`.
The sort result is stored in `Population`'s cache (`get_cache`/`set_cache`) and reused within the generation until the population changes.

### Population-relative Comparators

`SPEA2Comparator` and `HypervolumeComparator` both raise `NotImplementedError` when `compare()` is called.
This is because both SPEA2's fitness and exclusive HV contribution are metrics that depend on the entire population, and cannot be computed from just two points.
This is not a bug but an intentional design, marked by the class attribute `is_population_relative=True`.

These Comparators can't be used in situations that need a two-point comparison (`compare()`) alone, such as PSO's pbest update or `PairwiseComparisonSet`.
Use `ParetoComparator` instead in such situations.
`compare_population()` (comparison via population indices) is defined on both classes, so things like tournament selection work as-is.

`HypervolumeComparator`'s HV computation performs O(N) leave-one-out evaluations per front.
For problems with many objectives, this computational cost grows large.

```{note}
Separate from `HypervolumeComparator`'s internal implementation, there is a public function `saealib.hypervolume(f, reference_point)`.
This can be used standalone as a performance metric for evaluating results after optimization, and is unrelated to `HypervolumeComparator`.
See [Utils](../../api/utils.md) for details.
```

### Comparators using reference points

`NSGA3Comparator(reference_points, direction=None, *, ...)` requires `reference_points` (`shape (n_ref, n_obj)`, points on the unit simplex).
Normally, you pass points generated uniformly via `saealib.utils.weight_vectors.uniform_weight_vectors(n_obj, n_divisions)`.
The `rng` property is lazily generated; during an `Optimizer` run, `Runner` injects a random number generator spawned from `ctx.rng`.
This internal rng is not included when a checkpoint is saved, and is re-spawned on resume.

Unlike `NSGA3Comparator`, `RNSGA2Comparator(reference_points, epsilon=0.001, direction=None, *, ...)` doesn't require the reference points to lie on the unit simplex — you can specify the objective values you actually want (an aspiration point) directly.
`epsilon` is the radius of the ε-clearing that thins out solutions close to the same reference point.

`EpsilonDominanceComparator(eps, mode="additive", direction=None, *, ...)` is a thin wrapper that simply swaps `ParetoComparator`'s `dominator` for an [EpsilonDominator](dominance.md).

[DecompositionComparator](decomposition.md) is a Comparator that ranks using MOEA/D-style scalarization.
See that page for details.

## Implementing a custom Comparator

If you need a custom ranking scheme, subclass `Comparator` and implement all four methods.
`__init__` must always call `super().__init__(weights, eps_cv, eps_obj, direction=...)`.

```python
import numpy as np
from saealib import Comparator


class RandomComparator(Comparator):
    """A simple example that always considers only feasibility."""

    def __init__(self, direction=None, *, eps_cv=1e-6, eps_obj=1e-6):
        super().__init__(np.empty(0), eps_cv, eps_obj, direction=direction)

    def sort_population(self, population):
        cv = population.get_array("cv")
        return np.argsort(cv)

    def compare_population(self, population, idx_a, idx_b):
        cv = population.get_array("cv")
        return self.compare(None, cv[idx_a], None, cv[idx_b])

    def compare(self, fa, cv_a, fb, cv_b):
        if cv_a > self.eps_cv and cv_b <= self.eps_cv:
            return 1
        if cv_b > self.eps_cv and cv_a <= self.eps_cv:
            return -1
        return 0
```

When implementing a metric that depends on the entire population, as `SPEA2Comparator`/`HypervolumeComparator` do, you can use the design pattern of setting the class attribute `is_population_relative = True` and having `compare()` raise a `NotImplementedError` explaining why.

## Related components

- [Dominator](dominance.md): The `dominator` argument of the `ParetoComparator` family
- [NonDominatedSorter](nondominated_sorting.md): The `sorter` argument of the `ParetoComparator` family
- [Decomposition](decomposition.md): The scalarization functions used by `DecompositionComparator`
- [Problem](problem.md): How to pass the `comparator` argument, and the default-selection rules
- [ParentSelection](../search_algorithms/parent_selection.md) / [SurvivorSelection](../search_algorithms/survivor_selection.md): Operators that use `Comparator` to select individuals

## References

- {py:class}`saealib.Comparator`
- {py:class}`saealib.SingleObjectiveComparator`
- {py:class}`saealib.WeightedSumComparator`
- {py:class}`saealib.ParetoComparator`
- {py:class}`saealib.NSGA2Comparator`
- {py:class}`saealib.SPEA2Comparator`
- {py:class}`saealib.HypervolumeComparator`
- {py:class}`saealib.EpsilonDominanceComparator`
- {py:class}`saealib.NSGA3Comparator`
- {py:class}`saealib.RNSGA2Comparator`
- {py:func}`saealib.hypervolume`
