from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import get_type_hints

from saealib.context import OptimizationState
from saealib.core.contracts import ComponentContract, StateContract
from saealib.core.state import USER_DATA, StatePatch, StateStore, StateView
from saealib.core.state.context import RuntimeContext
from saealib.pipeline import Pipeline, Stage
from saealib.stages import stage_component


class _StatefulStage(Stage):
    name = "stateful"

    def __init__(self) -> None:
        super().__init__()
        self.seen = None
        self.inner_context = None

    def contract(self) -> ComponentContract:
        return ComponentContract(
            state=StateContract(reads=(USER_DATA,), writes=(USER_DATA,))
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        self.seen = state
        self.inner_context = state._store.view((USER_DATA,), context=state).context
        value = state.get_state(USER_DATA)
        state.set_state(USER_DATA, value + 1)
        return state.replace(data={"value": value + 2})


class _AsyncStatefulStage(Stage):
    name = "async_stateful"

    def __init__(self) -> None:
        super().__init__()
        self.received = None

    def contract(self) -> ComponentContract:
        return ComponentContract(
            state=StateContract(reads=(USER_DATA,), writes=(USER_DATA,))
        )

    def execute(self, state: OptimizationState) -> OptimizationState:
        return state

    async def execute_async(
        self,
        state: OptimizationState,
        *,
        scheduler=None,
        feedback_builder=None,
        algorithm=None,
        callback_manager=None,
        strategy=None,
    ) -> OptimizationState:
        self.received = (
            scheduler,
            feedback_builder,
            algorithm,
            callback_manager,
            strategy,
        )
        state.set_state(USER_DATA, state.get_state(USER_DATA) + 1)
        return state


def _view() -> tuple[StateView, SimpleNamespace]:
    store = StateStore({USER_DATA: 1})
    owner = SimpleNamespace(_store=store)
    return store.view((USER_DATA,), context=RuntimeContext(owner)), owner


def test_stage_adapter_uses_state_view_boundary() -> None:
    stage = _StatefulStage()
    adapter = stage_component(stage)

    assert adapter._execution_mode == "graph-native"
    assert (
        get_type_hints(adapter.execute)["return"] is StatePatch
        and get_type_hints(adapter.execute)["state"] is StateView
    )
    view, _ = _view()

    patch = adapter.execute(view)

    assert isinstance(patch, StatePatch)
    assert patch.writes[USER_DATA] == {"value": 3}
    assert not isinstance(stage.seen, OptimizationState)
    assert isinstance(stage.seen, object)
    assert isinstance(stage.inner_context, RuntimeContext)


def test_factory_can_be_used_as_a_structured_pipeline_component() -> None:
    adapter = stage_component(_StatefulStage())
    pipeline = Pipeline([adapter])

    assert pipeline.stages == [adapter]
    assert adapter.contract().state.reads == (USER_DATA,)
    assert adapter.contract().state.writes == (USER_DATA,)


def test_async_stage_adapter_converts_transaction_and_filters_kwargs() -> None:
    stage = _AsyncStatefulStage()
    adapter = stage_component(stage)
    view, _ = _view()
    services = {
        "scheduler": object(),
        "feedback_builder": object(),
        "algorithm": object(),
        "callback_manager": object(),
        "strategy": object(),
        "ignored": object(),
    }

    patch = asyncio.run(adapter.execute_async(view, **services))

    assert patch.writes[USER_DATA] == 2
    assert stage.received == tuple(
        services[name] for name in services if name != "ignored"
    )
