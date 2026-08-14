---
primary_layer: layer2
---

# Population

`Population` は、個体の集合と評価結果を扱うためのドメインデータ構造です。
旧来のベクトル経路では `x`、`f`、`g`、`cv` の列を持つ構造化配列として使います。
現行のgraph-native経路では、個体の表現を `GenomeBatch` と `SearchSpace` が定め、`Population` の `x` は利用できる場合の互換性用ビューです。

`Initializer` はProblemのSearchSpaceからGenomeを生成し、候補IDとともに状態やArchiveへ登録します。
したがって、すべてのProblemを浮動小数点の `x` 配列へ変換できるとは限りません。

## Populationの表現

`Population` は、Genome、目的値、制約値、制約違反、Algorithm固有の補助属性を保持します。
Genomeは `DenseVectorBatch`、`ObjectBatch`、PermutationBatchなどの `GenomeBatch` 実装で表せます。
数値ベクトルとして安全に取得できる場合だけ、`x` またはdense numeric viewを使います。

`Population` は通常インスタンスとして使い、独自の個体管理が必要な場合だけサブクラス化します。

属性スキーマは `PopulationAttribute(name, dtype, shape, default)` のリストで定義します。
旧来のベクトル経路では `x`、`f`、`g`、`cv` と、`Algorithm.get_required_attrs(problem)` が返す補助属性をこのスキーマへ追加します。
Genome-native経路では、GenomeBatchとSearchSpaceの契約が表現の構造を担い、列スキーマだけで個体を表しません。

## Main attributes and methods

| Method | Role |
|---|---|
| `get_array(key)` | Gets an attribute's raw array as a write-protected view |
| `update_array(key, value)` | Updates an attribute array in bulk |
| `get(key, default=None)` | A safe getter that returns `default` if the attribute doesn't exist |
| `append(element=None, **kwargs)` | Adds a single individual |
| `extend(other)` | Adds a batch of individuals from another `Population` or a dict |
| `extract(indices)` | Extracts a subset as a new `Population`, via an index array or slice |
| `truncate(new_size)` / `delete(index)` / `clear()` | Resizing and deletion |
| `reorder(order)` / `argsort(name, reverse=False)` | Reordering |
| `empty_like(capacity=None)` | Creates an empty `Population` with the same schema |
| `set_cache(key, value)` / `get_cache(key)` | Holds a computed result as a cache valid until the Population changes |
| `pop[i]` / `pop[a:b]` | Returns an `Individual` for a single int, or a `Population` subset for a slice |
| `len(pop)` | The number of individuals |

The cache from `set_cache`/`get_cache` is automatically invalidated whenever an operation that changes the population — `append`/`delete`/`update_array`, etc. — is called.
[NSGA2Comparator](../problem_and_ranking/comparators.md) uses this mechanism when reusing its front-splitting and crowding-distance computation results within a generation.

```python
import numpy as np
from saealib import Population, PopulationAttribute

attrs = [
    PopulationAttribute("x", np.float64, (2,)),
    PopulationAttribute("f", np.float64, (1,)),
    PopulationAttribute("cv", np.float64, ()),
]
pop = Population(attrs, init_capacity=4)
pop.append(x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0)
pop.append(x=np.array([0.3, 0.4]), f=np.array([2.0]), cv=0.0)

pop.x  # design-variable array with shape (2, 2)
pop[0]  # an Individual view of the first individual
pop[0:1]  # a Population containing only the first individual
```

### Individual

`Individual` is a lightweight view of a single individual, obtained via `pop[i]`.
It doesn't duplicate the actual data — it holds only a reference to the source `Population` and its own index.

You can read and write values via `get_readonly_value(key)`/`update_value(key, value)`, or equivalently via attribute access such as `ind.x`/`ind.f = ...`.
Using a stale `Individual` after the source `Population`'s structure (number of individuals or ordering) has changed raises an exception, as an invalid reference.

## Archive

`Archive` is a concrete class mixing `ArchiveMixin` into `Population`, used to accumulate evaluated solutions without duplicates.

`add(element, **kwargs)` takes almost the same arguments as `append`, but differs in that it ignores duplicate solutions.
The attribute used for duplicate detection is specified via the `key_attr` argument (default `"x"`), with `atol`/`rtol` adjusting the tolerance.
`get_knn(x, k)` provides nearest-neighbor search via a kd-tree (lazily built on first call), used by [LocalSurrogateManager](../surrogate_modeling/surrogate_manager.md)'s default `training_set` when gathering per-candidate local training data.

```python
from saealib import Archive

arc = Archive(attrs, init_capacity=4, key_attr="x")
arc.add(x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0)
arc.add(
    x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0
)  # the duplicate solution is ignored
idx, dist = arc.get_knn(np.array([0.1, 0.2]), k=1)
```

## ParetoArchive

`ParetoArchive` is a concrete class mixing `ParetoMixin` into `Population`, continuously maintaining a non-dominated solution set.

Every time a new solution is added, existing solutions it dominates are removed, and if the new solution is itself dominated by an existing one, the new solution is discarded.
Dominance is judged using a feasibility-first scheme.
A feasible solution (`cv <= eps_cv`) always dominates an infeasible one; [Dominator](../problem_and_ranking/dominance.md)'s `dominates` is used only when both are feasible.

The `dominator` argument lets you swap out the definition of the dominance relation.
`eps_cv`'s default is `0.0` (only strictly feasible solutions are considered acceptable), but during an `Optimizer` run, this value is overwritten every generation with `problem.handler.feasibility_threshold`.
The `0.0` default only matters when using `ParetoArchive` standalone, detached from `Optimizer`.

## A limited extension point

`ArchiveMixin`/`ParetoMixin` are designed on the assumption that they're mixed into a subclass of `Population` via multiple inheritance.
If you need custom population-management logic, you can define a new class combining these mixins (`class MyArchive(ArchiveMixin, Population): ...`).
You can also override [Algorithm](../search_algorithms/algorithm.md)'s `population_class`/`archive_class` to swap the Population/Archive that `Initializer` generates for your own custom subclass.

## Related components

- [Initializer](../execution_and_evaluation/initialization.md): Constructs `Population`/`Archive`/`ParetoArchive` at the start of a run
- [OptimizationState](optimization_state.md): The state object holding the constructed `Population`/`Archive`/`ParetoArchive`
- [Algorithm](../search_algorithms/algorithm.md): Swaps the concrete classes via `population_class`/`archive_class`
- [Comparator](../problem_and_ranking/comparators.md): Reuses sort results via `set_cache`/`get_cache`
- [Dominance](../problem_and_ranking/dominance.md): The `Dominator` `ParetoArchive` uses to judge non-dominated solutions
- [SurrogateManager](../surrogate_modeling/surrogate_manager.md): Gathering local training data via `Archive.get_knn`

## References

- {py:class}`saealib.Population`
- {py:class}`saealib.Individual`
- {py:class}`saealib.PopulationAttribute`
- {py:class}`saealib.Archive`
- {py:class}`saealib.ArchiveMixin`
- {py:class}`saealib.ParetoArchive`
- {py:class}`saealib.ParetoMixin`
