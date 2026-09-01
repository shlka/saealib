---
primary_layer: cross
related_layers: [layer2, layer3]
page_type: entry
---

# 探索アルゴリズム

`Algorithm`は候補を生成する`ask`と、評価結果を受け取る`tell`を担います。
`Crossover`と`Mutation`が候補を変化させ、`ParentSelection`が交叉に使う親を選びます。
生成された候補のうち次世代へ残す個体は`SurvivorSelection`が選びます。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`dna;sd-mr-1` 探索アルゴリズム
:link: search_algorithms/algorithm
:link-type: doc
候補生成と評価結果の消費を分離した探索アルゴリズムの契約を提供します。
:::

:::{grid-item-card} {fa}`code-fork;sd-mr-1` 交叉
:link: search_algorithms/crossover
:link-type: doc
選択された親から子候補を生成します。
:::

:::{grid-item-card} {fa}`bolt-lightning;sd-mr-1` 突然変異
:link: search_algorithms/mutation
:link-type: doc
交叉後の候補へ確率的な変化を加えます。
:::

:::{grid-item-card} {fa}`hand-pointer;sd-mr-1` 親選択
:link: search_algorithms/parent_selection
:link-type: doc
集団から交叉に使う親の組を選びます。
:::

:::{grid-item-card} {fa}`user-check;sd-mr-1` 生存選択
:link: search_algorithms/survivor_selection
:link-type: doc
親と子などの選択プールから次世代の個体を選びます。
:::

::::

```{toctree}
:hidden:

search_algorithms/algorithm
search_algorithms/crossover
search_algorithms/mutation
search_algorithms/parent_selection
search_algorithms/survivor_selection
```
