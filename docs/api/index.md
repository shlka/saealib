# API Reference

## API layers

Use the following four canonical layers when choosing an import path:

- **Root convenience API** — stable, common entry points:

  ```python
  from saealib import minimize, Problem, GA
  ```

- **Public namespaces** — public components grouped by domain:

  ```python
  from saealib.surrogate import RBFSurrogate
  from saealib.operators import MutationPolynomial
  ```

- **Framework extension API** — contracts and composition primitives for extensions:

  ```python
  from saealib.core import Component, ComponentGraph, ComponentContract
  ```

- **Internal implementation modules** — modules such as
  `saealib.core.compiler.compiler` are unsupported and may change without notice.

Namespace exports do not automatically become root exports. Preserve-root compatibility
means existing root imports remain available; adding a name to a namespace is not, by
itself, a reason to add it to `saealib`.

Add a new root convenience export only when the API is broadly useful, expected to be
stable, and materially improves the common import path. Otherwise, keep it in the
appropriate public namespace or framework extension API.

```{toctree}
:maxdepth: 2

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
```
