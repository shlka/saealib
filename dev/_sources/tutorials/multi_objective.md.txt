# Multi-Objective Optimization

Solve multi-objective optimization problems with trade-offs between objectives, using `saealib`.

Regardless of the number of objectives, switching algorithms, surrogates, and the evaluation strategy works the same as in "Switching components" in [Single-Objective Optimization](single_objective.md).

This page covers what's specific to problems with two or more objectives: choosing a comparator and extracting the Pareto front.

## Problem setup

When multiple objective functions are in a trade-off relationship, there exist solutions where improving one worsens another.

Under this relationship, the set of solutions not dominated by any other solution in every objective is called the **Pareto front**.

Here, as an example, we minimize the ZDT1 function built into `saealib`.

```python
from saealib.benchmarks import zdt1

problem = zdt1(n_var=10)
```

`zdt1` is a `Problem` instance returning a two-objective benchmark problem with a convex Pareto front.

## High-level API: minimize

Passing a `Problem` instance directly carries over the number of objectives from it.

```python
from saealib import minimize

result = minimize(problem, max_fe=2000, seed=0)

print(result.x.shape)  # (n_pareto, dim)
print(result.f.shape)  # (n_pareto, n_obj)
```

Where `result.x`/`result.f` were a single point in the single-objective case, in the multi-objective case they become multiple solutions forming the Pareto front.

## Choosing a comparator

In multi-objective optimization, `Comparator` decides the relative superiority between candidate solutions.

If `Problem`'s `comparator` argument is omitted, one is auto-selected based on the number of objectives (`SingleObjectiveComparator` when `n_obj == 1`, `NSGA2Comparator` when `n_obj > 1`).

| Class | Behavior |
|--------|------|
| `NSGA2Comparator` | Diversity maintenance via non-dominated sorting and crowding distance (default) |
| `SPEA2Comparator` | Fitness based on strength of dominance and neighborhood density |
| `HypervolumeComparator` | Superiority judged by hypervolume contribution |
| `EpsilonDominanceComparator` | Superiority judged by epsilon-dominance |
| `NSGA3Comparator` | Diversity maintenance via reference points. Requires `reference_points` |
| `RNSGA2Comparator` | Concentrates solutions near specified reference points. Requires `reference_points` |

`comparator` can be swapped as an attribute of the `Problem` instance.

```python
from saealib.comparators import HypervolumeComparator

problem.comparator = HypervolumeComparator()
result = minimize(problem, max_fe=2000, seed=0)
```

## Extracting the Pareto front

After running, `result.ctx.pareto_archive` holds the final Pareto front.

```python
pareto_x = result.ctx.pareto_archive.get_array("x")
pareto_f = result.ctx.pareto_archive.get_array("f")
```

To compute the Pareto front from an arbitrary array of objective values, you can use `non_dominated_sort` directly.

```python
from saealib.comparators import non_dominated_sort

archive_f = result.ctx.archive.get_array("f")
ranks, fronts = non_dominated_sort(archive_f, direction=problem.direction)
front0_f = archive_f[fronts[0]]  # first non-dominated front
```

## References

- {py:func}`saealib.minimize`
- {py:class}`saealib.Problem`
- {py:class}`saealib.NSGA2Comparator` / {py:class}`saealib.SPEA2Comparator` / {py:class}`saealib.HypervolumeComparator` / {py:class}`saealib.EpsilonDominanceComparator` / {py:class}`saealib.NSGA3Comparator` / {py:class}`saealib.RNSGA2Comparator`
- {py:func}`saealib.non_dominated_sort`
