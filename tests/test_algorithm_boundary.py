"""Guards for the canonical ask/tell algorithm boundary."""

from __future__ import annotations

import inspect
from typing import get_type_hints

from saealib.algorithms import GA, PSO, Algorithm, AskTellAlgorithm, GenomeGA
from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.core.contracts import FeedbackBatch, ProposalBatch
from saealib.core.state import StateView
from saealib.core.state.patch import StatePatch
from saealib.core.state.store import StateStore
from saealib.stages import AskStage, TellStage


def test_builtin_algorithms_expose_only_canonical_ask_tell_signatures() -> None:
    for algorithm_type in (GA, PSO, GenomeGA, PymooAlgorithm):
        assert issubclass(algorithm_type, AskTellAlgorithm)
        ask = inspect.signature(algorithm_type.ask)
        tell = inspect.signature(algorithm_type.tell)
        ask_hints = get_type_hints(algorithm_type.ask)
        tell_hints = get_type_hints(algorithm_type.tell)
        assert tuple(ask.parameters) == ("self", "request", "state")
        assert tuple(tell.parameters) == ("self", "feedback", "state")
        assert ask_hints["return"] is ProposalBatch
        assert tell_hints["return"] is StatePatch


def test_algorithm_base_has_no_legacy_seam() -> None:
    import saealib.algorithms.base as base
    import saealib.core.state.store as store

    assert not any("legacy" in name.lower() for name in base.__dict__)
    assert not any("legacy" in name.lower() for name in store.__dict__)


def test_stages_keep_a_native_ask_tell_algorithm() -> None:
    class Native(Algorithm):
        def get_required_attrs(self, problem):
            return []

        @property
        def population_class(self):
            return object

        @property
        def archive_class(self):
            return object

        def ask(self, request, state):
            assert isinstance(request, type(request))
            assert isinstance(state, StateView)
            raise AssertionError("not executed in this guard")

        def tell(self, feedback, state):
            assert isinstance(feedback, FeedbackBatch)
            assert isinstance(state, StateView)
            raise AssertionError("not executed in this guard")

    algorithm = Native()
    assert AskStage(algorithm)._algorithm is algorithm
    assert TellStage(algorithm)._algorithm is algorithm


def test_state_view_can_carry_runtime_context_without_widening_reads() -> None:
    marker = object()
    events: list[object] = []
    view = StateStore().view((), context=marker, dispatch=events.append)
    assert view.context is marker
    view.dispatch("event")
    assert events == ["event"]
