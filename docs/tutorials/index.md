# Tutorials

Setup guides for common optimization scenarios, from the high-level API to manually assembling an `Optimizer`.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`bullseye;sd-mr-1` Single-Objective Optimization
:link: single_objective
:link-type: doc

Solve an expensive single-objective problem step by step, from `minimize()` to manually assembling an `Optimizer`.
:::

:::{grid-item-card} {fa}`layer-group;sd-mr-1` Multi-Objective Optimization
:link: multi_objective
:link-type: doc

Solve a problem with trade-offs between objectives and extract the Pareto front.
:::

:::{grid-item-card} {fa}`shield-halved;sd-mr-1` Constrained Optimization
:link: constraints
:link-type: doc

Define inequality constraints and control how infeasible solutions are handled.
:::

:::{grid-item-card} {fa}`shapes;sd-mr-1` Mixed-Variable Optimization
:link: mixed_variable
:link-type: doc

Solve problems that include integer and categorical variables alongside continuous variables.
:::

:::{grid-item-card} {fa}`arrows-rotate;sd-mr-1` Dynamic Switching
:link: dynamic_optimization
:link-type: doc

Switch the evaluation strategy or `SurrogateManager` at runtime based on the surrogate's prediction accuracy.
:::

:::{grid-item-card} {fa}`floppy-disk;sd-mr-1` Reproducibility and Checkpointing
:link: checkpoint
:link-type: doc

Make long-running optimizations reproducible and resumable.
:::

:::{grid-item-card} {fa}`file-lines;sd-mr-1` Logging Progress
:link: logging
:link-type: doc

Record optimization progress with the standard `logging` module.
:::

:::{grid-item-card} {fa}`plug;sd-mr-1` Integrating External Libraries
:link: external_libraries
:link-type: doc

Incorporate models from external libraries such as scikit-learn as surrogates.
:::

::::

```{toctree}
:hidden:

single_objective
multi_objective
constraints
mixed_variable
dynamic_optimization
checkpoint
logging
external_libraries
```
