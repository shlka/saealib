"""Compiler diagnostics for CORS decision semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from saealib.core.compiler.diagnostics import (
    ContractPath,
    Diagnostic,
    Severity,
)
from saealib.core.compiler.graph import ComponentGraph

CORS_NONSEQUENTIAL_MESSAGE = (
    "CORSDistance is used outside the source-faithful sequential one-candidate "
    "evaluation path. The CORS score may not directly determine the true-evaluated "
    "point, multiple candidates may share one decision, or distinct decisions may "
    "overlap. This configuration is supported, but does not reproduce the "
    "sequential CORS procedure."
)


def _stage(component: object) -> object:
    """Return the compatibility stage hidden behind a graph adapter."""
    return getattr(component, "stage", component)


def _requires_sequential_decisions(acquisition: object) -> bool:
    """Check CORS metadata without importing concrete acquisition classes."""
    if getattr(acquisition, "requires_sequential_decisions", False):
        return True
    children = getattr(acquisition, "acquisitions", None)
    if isinstance(children, Mapping):
        return any(_requires_sequential_decisions(child) for child in children.values())
    return False


def _cors_nodes(graph: ComponentGraph) -> tuple[tuple[str, object], ...]:
    result: list[tuple[str, object]] = []
    for node in graph.nodes:
        stage = _stage(node.component)
        acquisition = getattr(stage, "_acquisition", None)
        if acquisition is not None and _requires_sequential_decisions(acquisition):
            result.append((node.component_id, acquisition))
    return tuple(result)


def _planner_nodes(graph: ComponentGraph) -> tuple[tuple[str, object, object], ...]:
    result: list[tuple[str, object, object]] = []
    for node in graph.nodes:
        stage = _stage(node.component)
        planner = getattr(stage, "_planner", None)
        if planner is not None:
            result.append((node.component_id, planner, stage))
    return tuple(result)


def _owner_node_ids(graph: ComponentGraph, owner_id: str) -> frozenset[str]:
    """Return a stage node and its decomposed part nodes."""
    prefix = f"{owner_id}__"
    return frozenset(
        node.component_id
        for node in graph.nodes
        if node.component_id == owner_id or node.component_id.startswith(prefix)
    )


def _owner_id(graph: ComponentGraph, component_id: str) -> str:
    """Map a decomposed part node back to its owning stage node."""
    matches = [
        node.component_id
        for node in graph.nodes
        if component_id == node.component_id
        or component_id.startswith(f"{node.component_id}__")
    ]
    return min(matches, key=len) if matches else component_id


def _data_reachable(graph: ComponentGraph, starts: frozenset[str]) -> frozenset[str]:
    """Follow data edges from the supplied graph nodes."""
    reachable = set(starts)
    changed = True
    while changed:
        changed = False
        for edge in graph.data_edges:
            source = edge.source.component_id
            target = edge.target.component_id
            if source in reachable and target not in reachable:
                reachable.add(target)
                changed = True
    return frozenset(reachable)


def _planner_reachable(
    graph: ComponentGraph, acquisition_id: str, planner_id: str
) -> bool:
    """Return whether a CORS score data-flows to one planner."""
    starts = _owner_node_ids(graph, acquisition_id)
    targets = _owner_node_ids(graph, planner_id)
    return bool(starts and targets and _data_reachable(graph, starts) & targets)


def _surrogate_generation_nodes(graph: ComponentGraph) -> frozenset[str]:
    """Return nodes in a GenerationBasedStrategy surrogate-only region.

    The generation-based pipeline names the branch that feeds
    ``TellStage(channel=SURROGATE)`` ``surrogate_generations``.  Its acquisition
    score therefore does not feed the true-evaluation planner, even when that
    planner selects only one candidate.
    """
    result: set[str] = set()

    def visit(current: ComponentGraph) -> None:
        for region_node in getattr(current, "region_nodes", ()):
            region = getattr(region_node, "region", None)
            body = getattr(region, "body", None)
            if getattr(region, "region_id", None) == "surrogate_generations":
                result.update(node.component_id for node in getattr(body, "nodes", ()))
            if isinstance(body, ComponentGraph):
                visit(body)
            otherwise = getattr(region, "otherwise", None)
            if isinstance(otherwise, ComponentGraph):
                visit(otherwise)

    visit(graph)
    return frozenset(result)


def _candidate_count(
    graph: ComponentGraph, planner_node_ids: frozenset[str] | None = None
) -> int | None:
    """Return an ask count for one planner's upstream data-flow branch."""
    if planner_node_ids is None:
        candidate_nodes = frozenset(node.component_id for node in graph.nodes)
    else:
        reverse_reachable = set(planner_node_ids)
        changed = True
        while changed:
            changed = False
            for edge in graph.data_edges:
                if edge.target.component_id in reverse_reachable and (
                    edge.source.component_id not in reverse_reachable
                ):
                    reverse_reachable.add(edge.source.component_id)
                    changed = True
        candidate_nodes = frozenset(reverse_reachable)
    owner_ids = {_owner_id(graph, component_id) for component_id in candidate_nodes}
    counts: list[int] = []
    for node in graph.nodes:
        if node.component_id not in owner_ids:
            continue
        stage = _stage(node.component)
        count = getattr(stage, "_n_offspring", None)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            counts.append(count)
    if counts:
        return max(counts)
    return None


def _selects_multiple(
    planner: object, candidate_count: int | None
) -> Literal["single", "multiple", "unknown"]:
    """Classify the statically known unique-candidate cardinality."""
    planner_name = type(planner).__name__
    if planner_name == "RepeatedEvaluation":
        # RepeatedEvaluation uses EvaluateAll().plan(), then replicates the
        # complete candidate batch.  Its unique-candidate count is therefore
        # the static candidate count, not the replicate count.
        if candidate_count is None:
            return "unknown"
        if candidate_count > 1:
            return "multiple"
        if candidate_count == 1:
            return "single"
        return "unknown"
    if planner_name == "TopKEvaluation":
        k = getattr(planner, "k", None)
        if not isinstance(k, int) or isinstance(k, bool):
            return "unknown"
        if k > 1:
            return "multiple"
        if k == 1:
            return "single"
        return "unknown"
    if planner_name == "RatioEvaluation":
        ratio = getattr(planner, "ratio", None)
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
            return "unknown"
        if candidate_count is None:
            return "unknown"
        selected_count = max(1, int(ratio * candidate_count))
        return "multiple" if selected_count > 1 else "single"
    if planner_name == "EvaluateAll":
        if candidate_count is None:
            return "unknown"
        if candidate_count > 1:
            return "multiple"
        if candidate_count == 1:
            return "single"
        return "unknown"
    return "unknown"


class CORSNonSequentialEvaluationRule:
    """Warn when a CORS graph is non-canonical or statically non-sequential."""

    namespace = "core"
    name = "cors_nonsequential_evaluation"
    phase: Literal["verification"] = "verification"

    def apply(self, context) -> object:
        """Return one advisory diagnostic for a supported semantic extension."""
        from saealib.core.compiler.compiler import VerificationResult

        cors_nodes = _cors_nodes(context.graph)
        if not cors_nodes:
            return VerificationResult()
        planners = _planner_nodes(context.graph)
        reachable_planners = tuple(
            (component_id, planner, stage)
            for component_id, planner, stage in planners
            if any(
                _planner_reachable(context.graph, cors_id, component_id)
                for cors_id, _ in cors_nodes
            )
        )
        static_batch = any(
            _selects_multiple(
                planner,
                _candidate_count(
                    context.graph, _owner_node_ids(context.graph, component_id)
                ),
            )
            == "multiple"
            for component_id, planner, _ in reachable_planners
        )
        surrogate_generation_nodes = _surrogate_generation_nodes(context.graph)
        surrogate_generation_cors = any(
            component_id in surrogate_generation_nodes for component_id, _ in cors_nodes
        )
        if not static_batch and not surrogate_generation_cors:
            return VerificationResult()
        acquisition_id = cors_nodes[0][0]
        related = tuple(
            ContractPath(components=(component_id,))
            for component_id, _, _ in reachable_planners
        )
        return VerificationResult(
            diagnostics=(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="cors_nonsequential_evaluation",
                    message=CORS_NONSEQUENTIAL_MESSAGE,
                    path=ContractPath(components=(acquisition_id,)),
                    related=related,
                    resolutions=(
                        "Use a configuration where CORSDistance directly selects one "
                        "true-evaluated candidate per sequential decision, or accept "
                        "the supported extension.",
                    ),
                ),
            )
        )


__all__ = ["CORS_NONSEQUENTIAL_MESSAGE", "CORSNonSequentialEvaluationRule"]
