---
primary_layer: cross
related_layers: [layer2, layer3]
page_type: entry
---

# Surrogate modeling

`TrainingSet` builds training data from an Archive or Population, and `Surrogate` predicts objective values from that data.
`SurrogateManager` coordinates when fitting and prediction occur, while `AcquisitionFunction` converts predictions into scores for candidate selection.
`AccuracyBasedSurrogateSwitcher` switches the `SurrogateManager` or `OptimizationStrategy` according to prediction-accuracy evaluations.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`brain;sd-mr-1` Surrogate
:link: surrogate_modeling/surrogate
:link-type: doc
Fits a model from training data and returns predictions for candidates.
:::

:::{grid-item-card} {fa}`sitemap;sd-mr-1` SurrogateManager
:link: surrogate_modeling/surrogate_manager
:link-type: doc
Coordinates model fitting and batch prediction.
:::

:::{grid-item-card} {fa}`database;sd-mr-1` TrainingSet
:link: surrogate_modeling/training_set
:link-type: doc
Builds training inputs and labels from an Archive or Population.
:::

:::{grid-item-card} {fa}`calculator;sd-mr-1` AcquisitionFunction
:link: surrogate_modeling/acquisition_functions
:link-type: doc
Converts predictions into scalar scores for candidate selection.
:::

:::{grid-item-card} {fa}`toggle-on;sd-mr-1` AccuracyBasedSurrogateSwitcher
:link: surrogate_modeling/surrogate_switching
:link-type: doc
Evaluates Surrogate accuracy and switches the configuration used during a run.
:::

::::

```{toctree}
:hidden:

surrogate_modeling/surrogate
surrogate_modeling/surrogate_manager
surrogate_modeling/training_set
surrogate_modeling/acquisition_functions
surrogate_modeling/surrogate_switching
```
