---
primary_layer: layer1
---

# Multi-Objective Optimization

Solve multi-objective optimization problems with trade-offs between objectives, using `saealib`.

`minimize()`は目的関数の数に応じたデフォルトのコンポーネントを使います。
アルゴリズム、代理モデル、評価戦略を個別に設定する場合は `Optimizer.set_*()` を使います。

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

## Optimizerによる構成

`Optimizer(problem)`では、各コンポーネントを独立して設定できます。
`set_*()`は連鎖呼び出しのために同じ `Optimizer` を返し、`run()`または `iterate()`で最適化を実行します。

```python
from saealib import Optimizer, Termination, max_fe

optimizer = (
    Optimizer(problem, seed=0)
    .set_termination(Termination(max_fe(2000)))
)
ctx = optimizer.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

未設定のコンポーネントはデフォルトに解決されます。
多目的の順位付けは、次のComparatorの選択で設定します。

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
