from __future__ import annotations

import pytest

from saealib import Branch, Condition, Loop, Pipeline, Repeat
from saealib.core import ComponentContract, StateContract, lower_pipeline
from saealib.core.compiler.regions import BranchRegion, LoopRegion, RepeatRegion
from saealib.core.state.keys import RUNTIME_RNG
from saealib.exceptions import ValidationError


class _Component:
    def __init__(self, name: str) -> None:
        self.name = name

    def contract(self) -> ComponentContract:
        return ComponentContract()


class _Condition(Condition):
    def __init__(self, state: StateContract | None = None) -> None:
        self._state = state or StateContract()

    def contract(self) -> StateContract:
        return self._state

    def evaluate(self, context: object) -> bool:
        del context
        return True


def test_pipeline_is_structural_and_accepts_steps_keyword() -> None:
    pipeline = Pipeline(steps=[_Component("one")])

    assert not hasattr(pipeline, "execute")
    assert pipeline.stages is pipeline.steps
    assert pipeline["one"].name == "one"


def test_structural_dsl_values_lower_to_regions_without_cycles() -> None:
    condition = _Condition()
    pipeline = Pipeline(
        [
            Repeat(Pipeline([_Component("repeat_body")]), 2, name="repeat"),
            Loop(_Component("loop_body"), until=condition, name="loop"),
            Branch(
                condition,
                then=_Component("then_body"),
                else_=_Component("else_body"),
                name="branch",
            ),
        ],
        name="outer",
    )

    graph = lower_pipeline(pipeline)

    assert [type(node.region) for node in graph.regions] == [
        RepeatRegion,
        LoopRegion,
        BranchRegion,
    ]
    assert [node.region.qualified_id for node in graph.regions] == [
        "outer.repeat",
        "outer.loop",
        "outer.branch",
    ]
    assert not any(
        edge.source.component_id == edge.target.component_id
        for edge in graph.control_edges
    )


@pytest.mark.parametrize("count", [-1, True, "2"])
def test_repeat_rejects_invalid_count(count: object) -> None:
    with pytest.raises(ValidationError):
        Repeat(_Component("body"), count)  # type: ignore[arg-type]


def test_loop_and_branch_validate_conditions() -> None:
    with pytest.raises(ValidationError):
        Loop(_Component("body"), until=object())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Branch(object(), then=_Component("body"))  # type: ignore[arg-type]


def test_branch_effect_includes_both_alternatives() -> None:
    graph = lower_pipeline(
        Pipeline(
            [
                Branch(
                    _Condition(StateContract(reads=(RUNTIME_RNG,))),
                    then=_Component("then"),
                    else_=_Component("else"),
                )
            ]
        )
    )

    assert graph.effect.reads == (RUNTIME_RNG,)
    assert graph.regions[0].region.effect.reads == (RUNTIME_RNG,)
