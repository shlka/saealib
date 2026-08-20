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
    "CORSDistance is used with an evaluation configuration that may select "
    "multiple candidates per decision. This configuration is supported, but "
    "uses one beta value for the whole decision and does not reproduce the "
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
        # Repeated requests intentionally describe one candidate observed more
        # than once, not a multi-candidate decision.
        return False
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
    """Warn when a CORS graph is statically non-sequential."""

    namespace = "core"
    name = "cors_nonsequential_evaluation"
    phase: Literal["verification"] = "verification"

    def apply(self, context) -> object:
        """Return one advisory diagnostic for a supported batch extension."""
        from saealib.core.compiler.compiler import VerificationResult

        cors_nodes = _cors_nodes(context.graph)
        if not cors_nodes:
            return VerificationResult()
        planners = _planner_nodes(context.graph)
        candidate_count = _candidate_count(context.graph)
        static_batch = any(
            _selects_multiple(planner, candidate_count) for _, planner, _ in planners
        )
        async_overlap = (
            context.compile_context.async_max_pending is not None
            and context.compile_context.async_max_pending > 1
        )
        if not static_batch and not async_overlap:
            return VerificationResult()
        acquisition_id = cors_nodes[0][0]
        related = tuple(
            ContractPath(components=(component_id,))
            for component_id, _, _ in planners[:1]
        )
        if async_overlap:
            related += (ContractPath(components=("async_evaluation_scheduler",)),)
        return VerificationResult(
            diagnostics=(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="cors_nonsequential_evaluation",
                    message=CORS_NONSEQUENTIAL_MESSAGE,
                    path=ContractPath(components=(acquisition_id,)),
                    related=related,
                    resolutions=(
                        "Use one true-evaluated candidate per decision, or accept "
                        "the supported batch extension.",
                    ),
                ),
            )
        )


__all__ = ["CORS_NONSEQUENTIAL_MESSAGE", "CORSNonSequentialEvaluationRule"]
