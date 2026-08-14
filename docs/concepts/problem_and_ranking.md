---
primary_layer: cross
related_layers: [layer2, layer3]
page_type: entry
---

# 問題定義と順位付け

Problemは、評価対象の関数、目的方向、探索空間、制約を一つの最適化対象としてまとめます。
制約処理は候補の実行可能性と制約違反を計算し、その結果をComparatorが目的値と合わせて比較します。
複数目的ではDominatorが優越関係を定め、NonDominatedSorterがその関係から非劣解の階層を作ります。
Decompositionは複数の目的をスカラー化し、分解に基づく比較を可能にします。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`compass;sd-mr-1` 問題（Problem）
:link: problem_and_ranking/problem
:link-type: doc
目的関数、変数、目的方向、制約、探索空間を定義します。
:::

:::{grid-item-card} {fa}`filter;sd-mr-1` 制約処理（ConstraintHandler）
:link: problem_and_ranking/constraints
:link-type: doc
制約違反の集約、実行可能性の判定、修復を差し替えます。
:::

:::{grid-item-card} {fa}`arrow-down-up-across-line;sd-mr-1` 比較器（Comparator）
:link: problem_and_ranking/comparators
:link-type: doc
目的値と制約違反から解を順位付けします。
:::

:::{grid-item-card} {fa}`crown;sd-mr-1` 優越関係（Dominator）
:link: problem_and_ranking/dominance
:link-type: doc
Pareto優越やε優越など、解どうしの優越関係を定めます。
:::

:::{grid-item-card} {fa}`arrow-down-wide-short;sd-mr-1` 非劣解ソーター（NonDominatedSorter）
:link: problem_and_ranking/nondominated_sorting
:link-type: doc
優越関係を使って集団を非劣解フロントへ分けます。
:::

:::{grid-item-card} {fa}`scissors;sd-mr-1` 分解（Decomposition）
:link: problem_and_ranking/decomposition
:link-type: doc
複数目的をスカラー化する分解関数を提供します。
:::

::::

```{toctree}
:hidden:

problem_and_ranking/problem
problem_and_ranking/search_space
problem_and_ranking/constraints
problem_and_ranking/comparators
problem_and_ranking/dominance
problem_and_ranking/nondominated_sorting
problem_and_ranking/decomposition
```
