---
primary_layer: cross
related_layers: []
page_type: entry
---

# Getting started

Learn what you need to start using saealib and how to run a simple optimization.

## Start with the basics

Start here for installation and a simple implementation.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`circle-info;sd-mr-1` What is saealib?
:link: what_is_saealib
:link-type: doc
Learn saealib's purpose and overall structure.
:::

:::{grid-item-card} {fa}`download;sd-mr-1` Installation
:link: installation
:link-type: doc
Install saealib in your environment.
:::

:::{grid-item-card} {fa}`rocket;sd-mr-1` Quickstart
:link: quickstart
:link-type: doc
Run your first optimization.
:::

::::

(choose-your-layer)=
## Choose how to use saealib

saealib offers four ways to work, depending on how much of the configuration you assemble yourself.

| Layer | Usage | First step |
|---|---|---|
| Layer 1: Use | Run an optimization with the defaults. Define a problem and pass it to `minimize()` or `maximize()`. | [Quickstart](quickstart.md), [Tutorials](../tutorials/index.md) |
| Layer 2: Compose | Choose built-in components and combine them with `Optimizer`. Reproduce algorithms from the literature. | [Optimization components](../concepts/index.md), [Algorithms](../algorithms/index.md) |
| Layer 3: Extend Components | Replace an existing responsibility with a custom implementation by subclassing an abstract base. | [Implement a custom component](../tutorials/custom_components.md) |
| Layer 4: Extend Framework | Extend the meaning of contracts, graphs, compilers, and execution itself. | [Framework extensions](../framework/extensions.md) |

Start at Layer 1 and move down the layers when the defaults no longer cover your needs.
To choose a layer by the part you want to change, see [Choosing an extension point](../concepts/extension_guidelines.md).

```{toctree}
:hidden:

what_is_saealib
installation
quickstart
```
