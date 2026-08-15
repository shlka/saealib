---
primary_layer: layer4
related_layers: [layer3]
page_type: entry
---

# Framework

The saealib framework places Component contracts in a ComponentGraph and provides the execution foundation for the Compiler to turn them into a verified ExecutablePlan.
It treats contract structure and execution flow as separate relationships.

## Concept pages

Components return their static contract through `contract()`.
ComponentContract describes contract inclusion for the components a Component holds and the capabilities it requires, not a Component inheritance hierarchy.
`PortSpec`, `DataSpec`, and `StateContract` are declarative values that compose a contract, not Component subclasses.

The following pages explain the concepts needed for extensions by responsibility.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`puzzle-piece;sd-mr-1` Component
:link: component
:link-type: doc
The Component Protocol, `contract()`, and PartSpec.
:::

:::{grid-item-card} {fa}`file-signature;sd-mr-1` ComponentContract
:link: contract
:link-type: doc
Contract elements and invariants.
:::

:::{grid-item-card} {fa}`plug;sd-mr-1` Specs
:link: specs
:link-type: doc
Declarative values for ports, data, services, and compatibility.
:::

:::{grid-item-card} {fa}`diagram-project;sd-mr-1` Graph
:link: graph
:link-type: doc
Nodes, edges, state bindings, and structured regions.
:::

:::{grid-item-card} {fa}`gears;sd-mr-1` Compiler
:link: compiler
:link-type: doc
Verification, resolution, diagnostics, and ExecutablePlan.
:::

:::{grid-item-card} {fa}`microchip;sd-mr-1` Runtime
:link: runtime
:link-type: doc
ExecutablePlan execution, state application, resumption, and asynchronous waiting.
:::

:::{grid-item-card} {fa}`square-root-variable;sd-mr-1` SearchSpace
:link: ../concepts/problem_and_ranking/search_space
:link-type: doc
Candidate representation, services, and the RepresentationSpec boundary.
:::

:::{grid-item-card} {fa}`comment-dots;sd-mr-1` Feedback
:link: ../concepts/observation_and_state/feedback
:link-type: doc
Candidate IDs and the correspondence between observations, true values, and predictions.
:::

:::{grid-item-card} {fa}`database;sd-mr-1` OptimizationState
:link: ../concepts/observation_and_state/optimization_state
:link-type: doc
The ownership boundary between Stage-compatible and graph-native state.
:::

::::

## Execution flow

The runtime relationship is `Component → ComponentNode → ComponentGraph → Compiler → ExecutablePlan → ExecutionRuntime`.
This flow is separate from the contract tree; ComponentContract does not directly represent execution order.

See [OptimizationState](../concepts/observation_and_state/optimization_state.md) and [Runtime](runtime.md) for detailed state boundaries, and [SearchSpace](../concepts/problem_and_ranking/search_space.md) and [Feedback](../concepts/observation_and_state/feedback.md) for candidate-representation and observation boundaries.

## Extension paths

When you only need to swap the behavior of a built-in component, choose the corresponding extension page.
When you need a new contract, candidate representation, Graph connection, Compiler rule, or Runtime semantic, see [Framework extensions](extensions.md).
See the [API reference](../api/index.md) for import paths for public types.

```{toctree}
:hidden:

component
contract
specs
graph
compiler
runtime
extensions
```
