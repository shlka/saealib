---
primary_layer: cross
related_layers: []
page_type: entry
---

# Tutorials

These pages show implementations for specific optimization scenarios.
Choose a page for single-objective, multi-objective, constrained, mixed-variable, or another goal.
Each page is an example for one scenario and does not assume a particular configuration depth.

## Optimization scenarios

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`bullseye;sd-mr-1` Single-Objective Optimization
:link: single_objective
:link-type: doc

Solve an expensive single-objective problem with `minimize()`, choosing the algorithm, surrogate, and evaluation strategy.
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

Incorporate scikit-learn and PyTorch models as surrogates, and pymoo operators, algorithms, and problems, through adapters.
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

## Configuration and control

These guides cover choosing built-in components, assembling the low-level API, and adding Hooks or Callbacks to an execution path.
The guides contain the details; this page provides an overview and links.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`puzzle-piece;sd-mr-1` Choose and swap built-in components
:link: component_swap
:link-type: doc
Choose the built-in component for a responsibility.
:::

:::{grid-item-card} {fa}`sliders;sd-mr-1` Assemble Optimizer with the low-level API
:link: lowlevel_api
:link-type: doc
Assemble a full configuration with a surrogate, combine termination conditions, and inspect each generation with `iterate()`.
:::

:::{grid-item-card} {fa}`eye;sd-mr-1` Choose a Hook, Stage, or Callback
:link: interface_hooks
:link-type: doc
Choose how to extend and observe an execution path.
:::

:::{grid-item-card} {fa}`rocket;sd-mr-1` Run an optimization with the high-level API
:link: highlevel_api
:link-type: doc
Run an optimization through the high-level API and read its Result.
:::

:::{grid-item-card} {fa}`puzzle-piece;sd-mr-1` Implement a custom component
:link: custom_components
:link-type: doc
Implement, register, and test a custom component that follows an existing contract. The guide also covers porting external Operators to native code.
:::

::::

```{toctree}
:hidden:

component_swap
lowlevel_api
interface_hooks
highlevel_api
custom_components
```
