---
primary_layer: cross
related_layers: [layer2, layer3]
page_type: entry
---

# Observation and state

CallbackManager emits events at execution boundaries so external code can observe progress and results.
Stage is the compatibility boundary that passes existing generation processing and `OptimizationState`; it differs from the state boundary of a graph-native component.
`OptimizationState` holds execution values and the values needed for resumption, while `Population` manages population data such as individuals, objectives, and constraints.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`eye;sd-mr-1` CallbackManager
:link: observation_and_state/callbacks
:link-type: doc
Record events and observe processing during execution.
:::

:::{grid-item-card} {fa}`circle-info;sd-mr-1` Diagnostics and observation
:link: observation_and_state/diagnostics
:link-type: doc
Choose the right mechanism for errors, warnings, logs, history, callbacks, and compiler diagnostics.
:::

:::{grid-item-card} {fa}`comment-dots;sd-mr-1` Feedback
:link: observation_and_state/feedback
:link-type: doc
Match candidates to observations and pass them in a form the Algorithm can use.
:::

:::{grid-item-card} {fa}`bars-staggered;sd-mr-1` Stage
:link: observation_and_state/stage
:link-type: doc
A compatibility execution unit for composing generation processing.
:::

:::{grid-item-card} {fa}`hard-drive;sd-mr-1` OptimizationState
:link: observation_and_state/optimization_state
:link-type: doc
Holds execution state, results, and values needed for checkpoints.
:::

:::{grid-item-card} {fa}`users;sd-mr-1` Population
:link: observation_and_state/population
:link-type: doc
Manages individuals and evaluation results as a Population and Archive.
:::

::::

```{toctree}
:hidden:

observation_and_state/callbacks
observation_and_state/diagnostics
observation_and_state/feedback
observation_and_state/stage
observation_and_state/optimization_state
observation_and_state/population
```
