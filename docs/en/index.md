---
primary_layer: cross
related_layers: []
page_type: entry
---

# saealib

saealib is a Python optimization library for objective functions whose evaluations take substantial time or cost.
It combines a model that approximates the objective from past evaluations with direct evaluation to reduce the number of expensive evaluations.

```{button-ref} getting_started/quickstart
:ref-type: doc
:color: primary
:shadow:
:class: sd-mr-2
```
```{button-ref} getting_started/what_is_saealib
:ref-type: doc
:color: secondary
:outline:
```

## Documentation

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`flag;sd-mr-1` Getting started
:link: getting_started/index
:link-type: doc
Learn the common ways to get started.
:::
:::{grid-item-card} {fa}`book-open;sd-mr-1` Tutorials
:link: tutorials/index
:link-type: doc
Follow task-oriented procedures.
:::
:::{grid-item-card} {fa}`cubes;sd-mr-1` Optimization components
:link: concepts/index
:link-type: doc
Explore the pieces that make up an optimization.
:::
:::{grid-item-card} {fa}`sitemap;sd-mr-1` Framework
:link: framework/index
:link-type: doc
Explore contracts, ComponentGraph, Compiler, and Runtime.
:::
:::{grid-item-card} {fa}`diagram-project;sd-mr-1` Algorithms
:link: algorithms/index
:link-type: doc
Review algorithm configurations and sources.
:::
:::{grid-item-card} {fa}`code;sd-mr-1` API reference
:link: api/index
:link-type: doc
Browse the public API.
:::
::::

## Minimal example

```python
from saealib import minimize
from saealib.benchmarks import rastrigin

result = minimize(rastrigin(n_var=10), max_fe=300, seed=0)
print(result.x, result.f)
```

Configure which components to use and how to combine them.
Configuration conflicts are validated before optimization starts.
See [What is saealib?](getting_started/what_is_saealib.md) and [Optimization components](concepts/index.md) for details.

Installation instructions are in [Installation](getting_started/installation.md), and implementation sources are listed in [References](references.md).

```{toctree}
:hidden:

getting_started/index
tutorials/index
concepts/index
framework/index
algorithms/index
references
api/index
```
