from dataclasses import replace

from saealib.core.compiler import (
    Compiler,
    ComponentGraph,
    ComponentNode,
    ControlEdge,
    NodeRef,
    StateBinding,
)
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import SURROGATES_DEFAULT, USER_DATA, StateKey


class _Component:
    def __init__(self, contract):
        self._contract = contract

    def contract(self):
        return self._contract


def _graph(*items, edges=(), bindings=()):
    nodes = tuple(
        ComponentNode(component_id=name, component=_Component(contract))
        for name, contract in items
    )
    return ComponentGraph(
        nodes=nodes,
        control_edges=tuple(edges),
        state_bindings=tuple(bindings),
        entry_points=(NodeRef(component_id=items[0][0]),),
    )


def _codes(graph, *, context=None):
    return [item.code for item in Compiler().compile(graph, context).diagnostics]


def test_ordered_write_then_read_is_valid():
    writer = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    reader = ComponentContract(state=StateContract(reads=(USER_DATA,)))
    graph = _graph(
        ("writer", writer),
        ("reader", reader),
        edges=(
            ControlEdge(
                source=NodeRef(component_id="writer"),
                target=NodeRef(component_id="reader"),
            ),
        ),
    )
    assert "concurrent_state_read_write" not in _codes(graph)
    assert "unreachable_state_read" not in _codes(graph)


def test_read_before_write_and_unwritten_read_are_diagnosed():
    reader = ComponentContract(state=StateContract(reads=(USER_DATA,)))
    writer = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    before = _graph(
        ("reader", reader),
        ("writer", writer),
        edges=(
            ControlEdge(
                source=NodeRef(component_id="reader"),
                target=NodeRef(component_id="writer"),
            ),
        ),
    )
    unwritten = _graph(("reader", reader))
    assert "unreachable_state_read" in _codes(before)
    assert "unreachable_state_read" in _codes(unwritten)


def test_read_write_without_an_initial_value_is_uninitialized():
    graph = _graph(
        (
            "node",
            ComponentContract(
                state=StateContract(reads=(USER_DATA,), writes=(USER_DATA,))
            ),
        )
    )
    assert "uninitialized_state_write" in _codes(graph)


def test_unordered_writes_and_read_write_are_diagnosed():
    writer = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    reader = ComponentContract(state=StateContract(reads=(USER_DATA,)))
    writes = _graph(("left", writer), ("right", writer))
    read_write = _graph(("writer", writer), ("reader", reader))
    assert "concurrent_state_write" in _codes(writes)
    assert "concurrent_state_read_write" in _codes(read_write)


def test_non_enumerable_reads_use_the_graph_state_universe():
    opaque = ComponentContract(state=StateContract(reads_enumerable=False))
    writer = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    graph = _graph(("opaque", opaque), ("writer", writer))
    assert "concurrent_state_read_write" in _codes(graph)
    ordered = replace(
        graph,
        control_edges=(
            ControlEdge(
                source=NodeRef(component_id="writer"),
                target=NodeRef(component_id="opaque"),
            ),
        ),
    )
    assert "concurrent_state_read_write" not in _codes(ordered)


def test_bindings_keep_same_contract_nodes_in_distinct_namespaces():
    contract = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    first_key = StateKey(namespace="user", name="first", schema_version=1)
    second_key = StateKey(namespace="user", name="second", schema_version=1)
    graph = _graph(
        ("node_a", contract),
        ("node_b", contract),
        bindings=(
            StateBinding(node=NodeRef(component_id="node_a"), state_key=first_key),
            StateBinding(node=NodeRef(component_id="node_b"), state_key=second_key),
        ),
    )
    plan = Compiler().compile(graph)
    assert "concurrent_state_write" not in _codes(graph)
    assert not [
        item for item in plan.diagnostics if item.code == "concurrent_state_write"
    ]


def test_bindings_qualify_declared_surrogate_keys_without_string_parsing():
    contract = ComponentContract(state=StateContract(writes=(SURROGATES_DEFAULT,)))
    cheap = StateKey(namespace="surrogates", name="stage:cheap", schema_version=1)
    rich = StateKey(namespace="surrogates", name="stage:rich", schema_version=1)
    graph = _graph(
        ("cheap", contract),
        ("rich", contract),
        bindings=(
            StateBinding(node=NodeRef(component_id="cheap"), state_key=cheap),
            StateBinding(node=NodeRef(component_id="rich"), state_key=rich),
        ),
    )
    plan = Compiler().compile(graph)
    assert not [
        item for item in plan.diagnostics if item.code == "concurrent_state_write"
    ]


def test_endpoint_roles_still_order_state_effects_by_node():
    writer = ComponentContract(state=StateContract(writes=(USER_DATA,)))
    reader = ComponentContract(state=StateContract(reads=(USER_DATA,)))
    graph = _graph(
        ("writer", writer),
        ("reader", reader),
        edges=(
            ControlEdge(
                source=NodeRef(component_id="writer", role="producer"),
                target=NodeRef(component_id="reader", role="consumer"),
            ),
        ),
    )
    assert "unreachable_state_read" not in _codes(graph)
