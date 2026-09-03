---
primary_layer: cross
---

# What is saealib?

saealib is a Python library for evolutionary computation with surrogate models.
It targets optimization problems whose objective evaluations are expensive.

## What is SAEA?

Evolutionary computation repeatedly generates candidate solutions, evaluates them with an objective function, and uses the results to create the next candidates.
When objective evaluations take time or money, directly evaluating many candidates becomes a major burden.

SAEA uses a surrogate model that estimates objective values from past evaluations to narrow the candidates sent for direct evaluation.
Surrogate predictions are inexpensive but approximate, so final decisions use direct objective evaluations.

```{mermaid}
flowchart TD
    A[Generate candidates] --> B[Predict with surrogate model]
    B --> C[Select candidates for true evaluation]
    C --> D[Evaluate with objective function]
    D --> E[Update model and search]
    E --> A
```

## What saealib provides

Candidate generation, surrogate modeling, and selection of candidates for evaluation are separate and replaceable.
`OptimizationStrategy` is the independent strategy that decides which candidates receive direct evaluation.

## Read next

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {fa}`rocket;sd-mr-1` Quickstart
:link: ../getting_started/quickstart
:link-type: doc
Run your first optimization with the smallest configuration.
:::

:::{grid-item-card} {fa}`graduation-cap;sd-mr-1` Tutorials
:link: ../tutorials/index
:link-type: doc
Learn task-oriented usage from examples.
:::

:::{grid-item-card} {fa}`cubes;sd-mr-1` Optimization components
:link: ../concepts/index
:link-type: doc
Review the roles of the components that make up saealib.
:::

:::{grid-item-card} {fa}`diagram-project;sd-mr-1` Algorithms
:link: ../algorithms/index
:link-type: doc
Review the configurations of the available algorithms.
:::

::::
