---
primary_layer: cross
related_layers: [layer1, layer2, layer3, layer4]
page_type: entry
---

# API reference

The API reference is organized around public facades.
Deep implementation-module paths are not documented as canonical imports.

## Public API layers

Use the following public layers when choosing an import path.

- **Root convenience API**: Stable, everyday entry points.

  ```python
  from saealib import minimize, Problem, GA
  ```

- **Public namespaces**: Public components grouped by domain.

  ```python
  from saealib.surrogate import RBFSurrogate
  from saealib.operators import MutationPolynomial
  ```

Import SearchSpace, execution, and Feedback names from the public namespaces `saealib.space`, `saealib.execution`, and `saealib.policies`, respectively.

- **Framework extension API**: Contracts and composition elements for extensions.

  ```python
  from saealib.core import Component, ComponentGraph, ComponentContract
  ```

- Implementation modules are not substitutes for public facades.

Names exported from a namespace are not automatically re-exported from the root.
Preserving an existing root import and adding a new name to the root are separate decisions.

Add a new root export only when it is broadly useful, expected to remain stable, and clearly improves common imports.
Place other names in the corresponding public namespace or framework extension API.

```{toctree}
:maxdepth: 2

imports
core
execution
space
feedback
highlevel
optimizer
exceptions
registry
problem
variables
comparators
decomposition
population
algorithms
operators
surrogate
acquisition
strategies
initialization
evaluation
termination
callbacks
pipeline
stages
utils
../references
```
