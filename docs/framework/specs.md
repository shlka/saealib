---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# Specs

Contract and SearchSpace boundaries pass connection conditions and representations as declarative values rather than processing them directly.
This page calls these declarative values specs.
Specs are immutable values, not Component subclasses, that the Compiler uses to determine connections and requirements.

## Specs' role

Specs own the boundary that passes connection conditions, representations, and required services to the Compiler rather than performing the processing itself.
They do not hold Component execution state; `ComponentContract` and `SearchSpace` own the declarative values.

## PortSpec and PortContract

`PortSpec` is one port declaration containing a port name, direction, `DataSpec`, cardinality, required services, and optionality.
`PortContract` is the set of input and output ports belonging to one role.
Port names must be unique within each direction.

The Compiler uses `ONE`, `MANY`, and `OPTIONAL` cardinalities to determine whether a provided value satisfies the consumer's requirement.
A connection is incompatible when its direction, registered data kind, schema binding, or cardinality does not match.

## DataSpec and ServiceRequirement

`DataSpec` represents a registered nominal data kind and schema binding.
Fixed values, variables, containment conditions, and product bindings can unify the same schema variable across ports.
DataSpec kind compatibility, schema unification, and service resolution are separate checks.

`ServiceRequirement` declares a named service required by a port or Component.
SearchSpace services such as `SamplingService` and `ValidationService` are matched as space capabilities.
`EvaluationAdapter` is an adapter at the evaluation boundary, while `FeatureEncoder` is an adapter subtype exposed as `saealib.space.FeatureEncoder`.
`FeatureEncoder` converts a `GenomeBatch` to a `FeatureBatch` and provides the semantic transformation that determines features a surrogate model can learn. It differs in nature from space capabilities such as `SamplingService`.
In the current implementation, the SurrogateManager contract declares `ServiceRequirement("FeatureEncoder")`, and `VectorSpace` registers a default FeatureEncoder as a service, so numeric vector spaces resolve it without extra configuration. `ObjectSpace`, `PermutationSpace`, and `SequenceSpace` raise an error unless the user supplies a FeatureEncoder. The user determines the input passed to the surrogate model.

## RepresentationSpec boundary

`RepresentationSpec` is the candidate-representation specification on the SearchSpace side.
It describes candidate type, shape, and representation, but is not part of a ComponentContract's ports, state, or execution.
Therefore, SearchSpace services, `EvaluationAdapter`, `FeatureEncoder`, and the RepresentationSpec held by the space are separate boundaries that the Compiler connects.

| Declarative value | Owner | Boundary using it |
|---|---|---|
| `PortSpec`, `PortContract` | `ComponentContract` | Graph connections and port compatibility |
| `DataSpec` | Port or contract | Data kind and schema binding unification |
| `ServiceRequirement` | Component or port | Resolving services provided by SearchSpace and Problem |
| `RepresentationSpec` | `SearchSpace` | Candidate representation and representation-service consistency |

## Construction time and use time

`PortSpec`, `DataSpec`, and `ServiceRequirement` are created when a ComponentContract is built and used by the Compiler when connecting the graph.
The contract or SearchSpace owns these values; the Runtime does not rewrite them during execution.

## Invariants and diagnostics

An unregistered kind, undefined cardinality, mismatched schema binding, or missing required service becomes a Diagnostic.
The implementation reports these with codes such as `unknown_data_spec`, `unknown_cardinality`, `unknown_schema_variable`, and `unresolved_service`.

## Extension points

When adding a data kind or service, state its registration and compatibility rules explicitly and do not connect it through an implicit type conversion.

## Related pages

See the [API reference](../api/index.md) for concrete port-declaration types and public import paths.
See [SearchSpace](../concepts/problem_and_ranking/search_space.md) for extending candidate representations and [Framework extensions](extensions.md) for extending Compiler rules.

## References

- {py:class}`saealib.core.PortSpec`
- {py:class}`saealib.core.PortContract`
- {py:class}`saealib.core.DataSpec`
