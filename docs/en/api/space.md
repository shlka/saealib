---
primary_layer: layer4
related_layers: [layer3]
page_type: reference
---

# SearchSpace API

The SearchSpace API is the public facade for candidate representations and space-specific services.
Use names from `saealib.space` rather than the space implementation modules.

## Spaces, services, and adapters

### Search spaces

Search spaces provide `GenomeBatch` representations.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.SearchSpace
   saealib.space.ServiceRegistry
   saealib.space.ValidationResult
   saealib.core.contracts.RepresentationSpec
   saealib.space.VectorSpace
   saealib.space.ObjectSpace
   saealib.space.SequenceSpace
   saealib.space.PermutationSpace
```

### SearchSpace services

SearchSpace services such as sampling and validation are capabilities provided by a space.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.GenomeCodec
   saealib.space.SamplingService
   saealib.space.ValidationService
```

### Adapters

`EvaluationAdapter` is an adapter at the evaluation boundary that converts a `GenomeBatch` into an `EvaluationPayload`.
`FeatureEncoder` is an adapter subtype exposed as `saealib.space.FeatureEncoder`. It converts a `GenomeBatch` into a `FeatureBatch` and determines which features a surrogate can learn. It differs in kind from space capabilities such as `SamplingService`.
In the current implementation, the `SurrogateManager` contract declares `ServiceRequirement("FeatureEncoder")`, and `VectorSpace` registers a default encoder as a service, so numeric vector spaces resolve it without extra configuration. `ObjectSpace`, `PermutationSpace`, and `SequenceSpace` fail to resolve it unless the user provides a `FeatureEncoder`. The user decides which input is passed to the surrogate.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.FeatureEncoder
```
