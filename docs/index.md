# saealib

A comprehensive library for **surrogate-assisted evolutionary algorithms (SAEA)**, implemented in Python.

Designed for optimization problems where the objective function is expensive to evaluate, `saealib` provides a modular framework that combines evolutionary algorithms, surrogate models, and model management strategies.

```{button-ref} getting_started/quickstart
:ref-type: doc
:color: primary
:shadow:
:class: sd-mr-2

Quickstart →
```
```{button-ref} getting_started/what_is_saealib
:ref-type: doc
:color: secondary
:outline:

What is saealib?
```

---

::::{grid} 1 2 2 3
:gutter: 3
:margin: 4 4 0 0

:::{grid-item-card} {fa}`bolt;sd-mr-1` High-level API
:link: getting_started/quickstart
:link-type: doc

```python
from saealib import minimize

result = minimize(func, dim=5,
                  lb=-5, ub=5)
```

A boilerplate-free high-level API via `minimize()` / `maximize()`.
:::

:::{grid-item-card} {fa}`sliders;sd-mr-1` Low-level API
:link: components/index
:link-type: doc

The `Optimizer` builder and `iterate()` generator enable per-generation inspection and custom loop control for research use.
:::

:::{grid-item-card} {fa}`puzzle-piece;sd-mr-1` Extensibility
:link: components/index
:link-type: doc

Every concept has an abstract base class and can be swapped at construction time.
This makes it possible to express any SAEA variant without forking the library.
:::

::::

---

## Documentation

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`flag;sd-mr-1` Getting Started
:link: getting_started/index
:link-type: doc

For first-time saealib users. Covers installation, basic usage, and core concepts.
:::

:::{grid-item-card} {fa}`book-open;sd-mr-1` Tutorials
:link: tutorials/index
:link-type: doc

Setup guides for specific scenarios: single-/multi-objective optimization, constraints, checkpointing.
:::

:::{grid-item-card} {fa}`cubes;sd-mr-1` Components
:link: components/index
:link-type: doc

Detailed usage of each component and guidelines for extending it.
:::

:::{grid-item-card} {fa}`diagram-project;sd-mr-1` Algorithms
:link: algorithms/index
:link-type: doc

How named algorithms from the literature are reproduced as combinations of saealib components.
:::

:::{grid-item-card} {fa}`bookmark;sd-mr-1` References
:link: references
:link-type: doc

A bibliography of the theoretical sources for the implemented algorithms, operators, and comparison methods.
:::

:::{grid-item-card} {fa}`code;sd-mr-1` API Reference
:link: api/index
:link-type: doc

The complete specification of every class and function.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

getting_started/index
tutorials/index
components/index
algorithms/index
references
api/index
```
