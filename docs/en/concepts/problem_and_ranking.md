---
primary_layer: cross
related_layers: [layer2, layer3]
page_type: entry
---

# Problem definition and ranking

`Problem` brings the objective function, objective directions, SearchSpace, and constraints together as one optimization target.
Constraint handling computes candidate feasibility and violations, and `Comparator` compares those results with objective values.
For multiple objectives, `Dominator` defines dominance and `NonDominatedSorter` builds layers of non-dominated solutions from that relation.
`Decomposition` scalarizes multiple objectives to enable decomposition-based comparison.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`compass;sd-mr-1` Problem
:link: problem_and_ranking/problem
:link-type: doc
Defines the objective function, variables, objective directions, constraints, and SearchSpace.
:::

:::{grid-item-card} {fa}`square-root-variable;sd-mr-1` SearchSpace
:link: problem_and_ranking/search_space
:link-type: doc
Defines candidate representation and the services needed for sampling and validation.
:::

:::{grid-item-card} {fa}`filter;sd-mr-1` Constraint handling
:link: problem_and_ranking/constraints
:link-type: doc
Replaces violation aggregation, feasibility checks, and repair.
:::

:::{grid-item-card} {fa}`arrow-down-up-across-line;sd-mr-1` Comparator
:link: problem_and_ranking/comparators
:link-type: doc
Ranks solutions from objective values and constraint violations.
:::

:::{grid-item-card} {fa}`crown;sd-mr-1` Dominator
:link: problem_and_ranking/dominance
:link-type: doc
Defines dominance relations between solutions, such as Pareto and ε-dominance.
:::

:::{grid-item-card} {fa}`arrow-down-wide-short;sd-mr-1` Non-dominated sorter
:link: problem_and_ranking/nondominated_sorting
:link-type: doc
Splits a population into non-dominated fronts using the dominance relation.
:::

:::{grid-item-card} {fa}`scissors;sd-mr-1` Decomposition
:link: problem_and_ranking/decomposition
:link-type: doc
Provides decomposition functions that scalarize multiple objectives.
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
