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


def _candidate_count(graph: ComponentGraph) -> int | None:
    """Return an explicit ask count when the graph exposes one."""
    counts: list[int] = []
    for node in graph.nodes:
        stage = _stage(node.component)
        count = getattr(stage, "_n_offspring", None)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            counts.append(count)
    if counts:
        return max(counts)
    return None


def _selects_multiple(planner: object, candidate_count: int | None) -> bool:
    """Return whether a known planner can select multiple unique candidates."""
    planner_name = type(planner).__name__
    if planner_name == "RepeatedEvaluation":
        # RepeatedEvaluation uses EvaluateAll().plan(), then replicates the
        # complete candidate batch.  Its unique-candidate count is therefore
        # the static candidate count, not the replicate count.
        return candidate_count is None or candidate_count > 1
    if planner_name == "TopKEvaluation":
        k = getattr(planner, "k", None)
        return isinstance(k, int) and not isinstance(k, bool) and k > 1
    if planner_name == "RatioEvaluation":
        ratio = getattr(planner, "ratio", None)
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
            return False
        if candidate_count is None:
            return ratio > 0.0
        return max(1, int(ratio * candidate_count)) > 1
    if planner_name == "EvaluateAll":
        return candidate_count is None or candidate_count > 1
    return False


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
        candidate_count = _candidate_count(context.graph)
        static_batch = any(
            _selects_multiple(planner, candidate_count) for _, planner, _ in planners
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
            for component_id, _, _ in planners[:1]
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
