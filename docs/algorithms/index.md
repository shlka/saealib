---
primary_layer: layer2
related_layers: []
page_type: entry
---

# Algorithms

Pages summarizing how named algorithms from the literature are reproduced as combinations of saealib components.
Each page has two parts: a theoretical overview of the algorithm (a general explanation independent of saealib), and how to configure it in saealib (the component combination and Python code).
Full bibliographic details for the sources are collected in [References](../references.md), and each page in this section links there.

When a configuration does not exactly match the theoretical definition, each page states the implementation differences, such as acquisition-function constraints, selection or archive handling, and evaluation order.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} EGO
:link: ego
:link-type: doc

Sequential model-based optimization combining Gaussian Process regression with the Expected Improvement acquisition function.
:::

:::{grid-item-card} GP-UCB
:link: gp_ucb
:link-type: doc

Sequential model-based optimization combining Gaussian Process regression with the Upper Confidence Bound acquisition function.
:::

:::{grid-item-card} MaxUnc
:link: maxunc
:link-type: doc

Exploration-only sequential model-based optimization based solely on the predictive uncertainty of Gaussian Process regression.
:::

:::{grid-item-card} CORS-RBF
:link: rbf_cors
:link-type: doc

Sequential model-based optimization combining an RBF-interpolation surrogate model with a distance constraint from existing evaluated points.
:::

:::{grid-item-card} NSGA-II
:link: nsga2
:link-type: doc

Multi-objective genetic algorithm based on non-dominated sorting and crowding distance. The basis for saealib's multi-objective comparators.
:::

:::{grid-item-card} SPEA2
:link: spea2
:link-type: doc

Multi-objective genetic algorithm combining dominance-based strength and density into a fitness value, with a fixed-size archive.
:::

:::{grid-item-card} NSGA-III
:link: nsga3
:link-type: doc

Multi-objective genetic algorithm that uses reference-direction niche preservation.
Primarily targets many-objective optimization, while exposing reference-direction behavior with three objectives.
:::

:::{grid-item-card} SMS-EMOA
:link: sms_emoa
:link-type: doc

Steady-state multi-objective evolutionary algorithm that incorporates dominated hypervolume directly into the selection criterion.
:::

::::

```{toctree}
:hidden:

ego
gp_ucb
maxunc
rbf_cors
nsga2
spea2
nsga3
sms_emoa
```
