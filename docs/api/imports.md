# Canonical Imports

saealib is still pre-1.0, so public APIs may evolve before the first stable
release. Within that constraint, use these four layers when choosing an import
path:

## 1. Root convenience API

Use the root package for common, user-facing entry points and convenience
examples:

```python
from saealib import minimize, Problem, GA
```

Root exports are intended to make the common path easy. An existing root import
is preserved as part of the root compatibility surface, but a name being
available elsewhere does not automatically make it a root export.

## 2. Domain namespaces

Use public namespaces when an object belongs to a particular domain. Examples
include `saealib.surrogate`, `saealib.acquisition`, and
`saealib.operators`:

```python
from saealib.surrogate import RBFSurrogate
from saealib.acquisition import MeanPrediction
from saealib.operators import MutationPolynomial
```

Other public domain namespaces follow the same rule. Prefer the namespace
export over a deeper module path when both are available.

## 3. Framework and runtime extension APIs

Use `saealib.core` as the framework extension facade for contracts and
composition primitives:

```python
from saealib.core import Component, ComponentGraph, ComponentContract
```

Use `saealib.execution` for runtime extension points. Its public runtime API
includes `RuntimeRegistry`, `RuntimeRegistration`, `RuntimeFactory`, and
`create_runtime`:

```python
from saealib.execution import (
    RuntimeFactory,
    RuntimeRegistration,
    RuntimeRegistry,
    create_runtime,
)
```

These facades are the supported starting points for framework and runtime
extensions. Their detailed behavior may still change while saealib remains
pre-1.0.

## 4. Deep implementation paths

Paths below the public facades, such as `saealib.core.compiler.compiler`, are
deep implementation paths. They are non-canonical and have no compatibility
guarantee; do not use them as the basis for application imports or extension
documentation.
