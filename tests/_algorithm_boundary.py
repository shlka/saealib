"""Small test helpers for exercising the canonical ask/tell boundary."""

from __future__ import annotations

from typing import Any, cast

from saealib.algorithms.base import ProposalRequest
from saealib.context import OptimizationState
from saealib.core.contracts import FeedbackBatch
from saealib.core.state import StateView


def view(
    algorithm: Any,
    state: OptimizationState,
    provider: Any | None = None,
) -> StateView:
    """Bind an optimization state and dispatch sink to a read-only view."""
    dispatch = getattr(provider, "dispatch", None)
    if not callable(dispatch):
        dispatch = None
    return state._store.view(
        algorithm.contract().state,
        context=state,
        dispatch=dispatch,
    )


def ask(
    algorithm: Any,
    state: OptimizationState,
    provider: Any | None = None,
    *,
    n_offspring: int | None = None,
) -> Any:
    """Call a proposer through the canonical request/state boundary."""
    proposal = algorithm.ask(
        ProposalRequest(n_offspring=n_offspring),
        view(algorithm, state, provider),
    )
    return proposal.candidates


def tell(
    algorithm: Any,
    state: OptimizationState,
    offspring: Any,
    provider: Any | None = None,
) -> Any:
    """Call a feedback consumer with the evaluated batch bound in state."""
    state.offspring = offspring
    return algorithm.tell(
        cast(FeedbackBatch, object()),
        view(algorithm, state, provider),
    )
