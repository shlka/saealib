---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# Graph

Placing Component contracts into execution order and data relationships produces a `ComponentGraph`.
`ComponentGraph` is an immutable graph holding that placement.
ComponentContract, which holds the contract tree, and ComponentGraph, which holds execution relationships between nodes, are separate structures.

## ComponentGraph's role

ComponentGraph owns the boundary that places Component contracts into execution order and data relationships.
It separates the contract tree held by ComponentContract from the node relationships held by the Graph and passes a self-contained structure to the Compiler.

## Nodes and references

`ComponentNode` holds a Component instance, component ID, role, resolved services, and a contract snapshot for compilation.
`NodeRef` references a node by component ID and optional role, normalizing endpoints for edges and StateBindings.
`NodeRef` distinguishes connection targets even when the same Component is placed under different roles.

## Edges and state

`DataEdge` is a data connection from a source port to a target port.
`ControlEdge` passes no data; it is a control dependency that makes source completion precede target execution.
`StateBinding` connects a node to actual typed StateKeys, mapping the contract's state declarations to runtime state.

The Graph holds nodes, edges, state bindings, and entry points as values.

## Structured regions

`StructuredRegion` holds nested Sequence, Repeat, Loop, and Branch structures together with the state effects read by the region.
A Loop is not represented as an ordinary Graph cycle; it is held in a form whose condition and region effects the Compiler can verify.
After lowering a structured region, the mapping between the execution tree passed to ExecutablePlan and its state effects remains intact.

## Construction time and use time

The Graph is created when a Pipeline or Graph builder places Components.
The Compiler reads contract snapshots and verifies entry points, edge endpoints, port compatibility, state bindings, and structured regions.
When a Compiler rule changes the Graph during Resolution, it declares the changed locations as claims; undeclared changes and conflicts become Diagnostics.

## Invariants and diagnostics

The Graph holds nodes, edges, state bindings, and entry points as values; endpoints referring to unknown nodes become Diagnostics such as `invalid_graph_edge` or `invalid_entry_point`.
Undeclared changes or conflicts from resolution rules become `unclaimed_rewrite` or `conflicting_rewrite` Diagnostics.

## Extension points

```python
from saealib.core import ComponentContract, ComponentGraph
from saealib.core.compiler import ComponentNode


class Empty:
    def contract(self) -> ComponentContract:
        return ComponentContract()


node = ComponentNode(component_id="root", component=Empty())
graph = ComponentGraph(nodes=(node,), entry_points=("root",))
```

This example creates a Graph with one node as its entry point.
See the [API reference](../api/index.md) for building actual ports and edges with the public API.
See [Framework extensions](extensions.md) when adding Graph connection rules.
Preserve endpoint and entry-point invariants and do not bypass Compiler verification.

## Related pages

[Contract](contract.md) explains contract inclusion, and [Compiler](compiler.md) explains Graph resolution and verification.

## References

- {py:class}`saealib.core.ComponentGraph`
- {py:class}`saealib.core.Component`
