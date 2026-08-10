from __future__ import annotations

from saealib.core.compiler import (
    Compiler,
    ComponentGraph,
    ComponentNode,
    ControlEdge,
    DataEdge,
    NodeRef,
    StructuredGraph,
)
from saealib.core.contracts import (
    MANY,
    ComponentContract,
    DataSpec,
    PortContract,
    PortDirection,
    PortSpec,
)
from saealib.pipeline import Pipeline


class _Component:
    def __init__(
        self,
        name: str,
        *,
        output: str | None = None,
        input_kind: str | None = None,
        optional: bool = False,
    ) -> None:
        self.name = name
        self._output = output
        self._input_kind = input_kind
        self._optional = optional

    def contract(self) -> ComponentContract:
        outputs = (
            ()
            if self._output is None
            else (
                PortSpec(
                    name="value",
                    direction=PortDirection.OUTPUT,
                    data=DataSpec(kind=self._output),
                    cardinality=MANY,
                ),
            )
        )
        inputs = (
            ()
            if self._input_kind is None
            else (
                PortSpec(
                    name="value",
                    direction=PortDirection.INPUT,
                    data=DataSpec(kind=self._input_kind),
                    cardinality=MANY,
                    optional=self._optional,
                ),
            )
        )
        return ComponentContract(
            ports={"main": PortContract(inputs=inputs, outputs=outputs)}
        )


def _compile(*components: _Component):
    return Compiler().compile_pipeline(Pipeline(list(components)))


def _edge_tuple(edge: DataEdge) -> tuple[str, str, str, str]:
    return (
        edge.source.component_id,
        edge.source.role or "",
        edge.target.component_id,
        edge.target.role or "",
    )


def test_structured_dataflow_adds_one_compatible_edge() -> None:
    plan = _compile(
        _Component("producer", output="Population"),
        _Component("consumer", input_kind="Population"),
    )

    assert isinstance(plan.graph, StructuredGraph)
    assert [_edge_tuple(edge) for edge in plan.graph.data_edges] == [
        ("producer", "main", "consumer", "main")
    ]
    assert not plan.diagnostics


def test_required_input_without_compatible_upstream_is_reported() -> None:
    plan = _compile(_Component("consumer", input_kind="Population"))

    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["unresolved_input"]


def test_multiple_compatible_upstream_producers_are_ambiguous() -> None:
    plan = _compile(
        _Component("first", output="Population"),
        _Component("second", output="Population"),
        _Component("consumer", input_kind="Population"),
    )

    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["ambiguous_input"]
    assert not plan.graph.data_edges


def test_optional_input_without_compatible_upstream_stays_unconnected() -> None:
    plan = _compile(_Component("consumer", input_kind="Population", optional=True))

    assert not plan.graph.data_edges
    assert not plan.diagnostics


def test_incompatible_upstream_producer_is_not_selected() -> None:
    plan = _compile(
        _Component("producer", output="FeatureBatch"),
        _Component("consumer", input_kind="Population"),
    )

    assert not plan.graph.data_edges
    assert [diagnostic.code for diagnostic in plan.diagnostics] == ["unresolved_input"]


def test_direct_component_graph_is_not_auto_resolved() -> None:
    producer = _Component("producer", output="Population")
    consumer = _Component("consumer", input_kind="Population")
    graph = ComponentGraph(
        nodes=(
            ComponentNode(component_id="producer", component=producer),
            ComponentNode(component_id="consumer", component=consumer),
        ),
        control_edges=(
            ControlEdge(
                source=NodeRef(component_id="producer"),
                target=NodeRef(component_id="consumer"),
            ),
        ),
        entry_points=(NodeRef(component_id="producer"),),
    )
    assert not Compiler().compile(graph).graph.data_edges
