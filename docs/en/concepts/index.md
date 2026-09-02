---
primary_layer: cross
related_layers: [layer2, layer3, layer4]
page_type: entry
---

# Optimization components

This section organizes the components that make up an optimization, including Component, Problem, Population, and Comparator.
The pages grouped by responsibility provide details about the built-in components.
See [Framework](../framework/index.md) for how contracts become execution plans.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`compass;sd-mr-1` Problem definition and ranking
:link: problem_and_ranking
:link-type: doc
Problem definition, constraint handling, ranking, and non-dominated solutions.
:::

:::{grid-item-card} {fa}`dna;sd-mr-1` Search algorithms
:link: search_algorithms
:link-type: doc
Candidate generation, mutation, parent selection, and survivor selection.
:::

:::{grid-item-card} {fa}`brain;sd-mr-1` Surrogate modeling
:link: surrogate_modeling
:link-type: doc
Training data, prediction, acquisition scoring, and model switching.
:::

:::{grid-item-card} {fa}`play;sd-mr-1` Execution and evaluation
:link: execution_and_evaluation
:link-type: doc
Initialization, candidate evaluation, generation processing, and termination.
:::

:::{grid-item-card} {fa}`eye;sd-mr-1` Observation and state
:link: observation_and_state
:link-type: doc
Event observation, the Stage compatibility boundary, execution state, and population data.
:::

:::{grid-item-card} {fa}`wand-magic-sparkles;sd-mr-1` Extension guidelines
:link: extension_guidelines
:link-type: doc
How to choose among swapping components, hooks, Stages, Callbacks, and framework extensions.
:::

::::

For your first optimization, see the [Quickstart](../getting_started/quickstart.md); to choose an extension point, see the [Extension guidelines](extension_guidelines.md).
See the [API reference](../api/index.md) for public import paths for each type.

```{toctree}
:hidden:

extension_guidelines
problem_and_ranking
search_algorithms
surrogate_modeling
execution_and_evaluation
observation_and_state
```
