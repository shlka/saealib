---
primary_layer: cross
related_layers: [layer2, layer3]
page_type: entry
---

# Search algorithms

`Algorithm` handles `ask`, which generates candidates, and `tell`, which receives evaluation results.
`Crossover` and `Mutation` change candidates, and `ParentSelection` chooses the parents used for crossover.
`SurvivorSelection` chooses which generated candidates remain for the next generation.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`dna;sd-mr-1` Algorithm
:link: search_algorithms/algorithm
:link-type: doc
Provides a search-algorithm contract that separates candidate generation from evaluation-result consumption.
:::

:::{grid-item-card} {fa}`code-fork;sd-mr-1` Crossover
:link: search_algorithms/crossover
:link-type: doc
Generates offspring from selected parents.
:::

:::{grid-item-card} {fa}`bolt-lightning;sd-mr-1` Mutation
:link: search_algorithms/mutation
:link-type: doc
Adds probabilistic changes to candidates after crossover.
:::

:::{grid-item-card} {fa}`hand-pointer;sd-mr-1` ParentSelection
:link: search_algorithms/parent_selection
:link-type: doc
Chooses parent groups for crossover from the population.
:::

:::{grid-item-card} {fa}`user-check;sd-mr-1` SurvivorSelection
:link: search_algorithms/survivor_selection
:link-type: doc
Chooses next-generation individuals from a pool such as parents and offspring.
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
