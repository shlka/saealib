"""Tests for validation-time contract diagnostics."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from pymoo.algorithms.soo.nonconvex.ga import GA as PymooGA  # noqa: N811

from saealib.acquisition.base import CompositeAcquisition
from saealib.acquisition.mean import CORSDistance, MeanPrediction
from saealib.algorithms.pymoo_algorithm import PymooAlgorithm
from saealib.core.compiler import (
    Compiler,
    ContractPath,
    Diagnostic,
    DiagnosticBag,
    Severity,
)
from saealib.core.compiler.compiler import (
    CompileContext,
    RuleContext,
    VerificationResult,
)
from saealib.core.compiler.cors_diagnostics import CORSNonSequentialEvaluationRule
from saealib.core.compiler.diagnostics import DIAGNOSTIC_CODES
from saealib.core.compiler.graph import ComponentGraph, ComponentNode, DataEdge, NodeRef
from saealib.core.contracts import ComponentContract
from saealib.execution.evaluator import SerialEvaluator
from saealib.execution.scheduler import AsyncEvaluationScheduler
from saealib.islands import IslandModel
from saealib.optimizer import Optimizer
from saealib.policies.evaluation import (
    RatioEvaluation,
    RepeatedEvaluation,
    TopKEvaluation,
)
from saealib.problem import Problem
from saealib.strategies.direct import DirectStrategy
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import GaussianKernel
from saealib.termination import Termination, max_fe


def _problem() -> Problem:
    return Problem(
        func=lambda x: np.array([float(np.sum(np.asarray(x) ** 2))]),
        dim=2,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-1.0, -1.0],
        ub=[1.0, 1.0],
    )


def _default_optimizer() -> Optimizer:
    optimizer = Optimizer(_problem())
    optimizer._resolve_defaults()
    return optimizer


def _diagnostic(severity: Severity, message: str) -> Diagnostic:
    return Diagnostic(
        severity=severity,
        code="contract_unavailable",
        message=message,
        path=ContractPath(components=("algorithm",)),
    )


def test_error_contract_diagnostic_fails_validate(monkeypatch) -> None:
    optimizer = _default_optimizer()
    finding = _diagnostic(Severity.ERROR, "fatal contract finding")
    monkeypatch.setattr(
        optimizer,
        "contract_diagnostics",
        lambda: DiagnosticBag((finding,)),
    )

    issues = optimizer.validate()

    assert str(finding) in issues
    assert optimizer.last_contract_diagnostics == (finding,)


def test_warning_and_info_are_visible_without_failing_validate(monkeypatch) -> None:
    optimizer = _default_optimizer()
    warning = _diagnostic(Severity.WARNING, "advisory contract finding")
    info = _diagnostic(Severity.INFO, "informational contract finding")
    monkeypatch.setattr(
        optimizer,
        "contract_diagnostics",
        lambda: DiagnosticBag((warning, info)),
    )

    assert optimizer.validate() == []
    assert optimizer.last_contract_diagnostics == (warning, info)


def test_pymoo_complete_batch_with_async_runtime_is_diagnosed() -> None:
    optimizer = Optimizer(_problem()).set_algorithm(
        PymooAlgorithm(PymooGA(pop_size=4), allow_partial_tell=False)
    )
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer._resolve_defaults()

    diagnostics = tuple(optimizer.contract_diagnostics())

    assert [diagnostic.code for diagnostic in diagnostics] == [
        "pymoo_partial_feedback_unsupported"
    ]
    assert diagnostics[0].severity is Severity.ERROR
    assert diagnostics[0].path == ContractPath(components=("algorithm",))
    assert diagnostics[0].related == (
        ContractPath(components=("async_evaluation_scheduler",)),
    )
    assert any(
        "pymoo_partial_feedback_unsupported" in issue for issue in optimizer.validate()
    )


def test_pymoo_partial_tell_opt_in_matches_async_runtime() -> None:
    optimizer = Optimizer(_problem()).set_algorithm(
        PymooAlgorithm(PymooGA(pop_size=4), allow_partial_tell=True)
    )
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator())
    )
    optimizer._resolve_defaults()

    assert tuple(optimizer.contract_diagnostics()) == ()


@pytest.mark.parametrize("kind", ("default", "surrogate", "async"))
def test_supported_component_configurations_have_no_diagnostics(kind: str) -> None:
    optimizer = Optimizer(_problem())
    if kind == "surrogate":
        optimizer.set_surrogate(RBFSurrogate(kernel=GaussianKernel()))
    elif kind == "async":
        optimizer.set_async_evaluation_scheduler(
            AsyncEvaluationScheduler(SerialEvaluator())
        )
    optimizer._resolve_defaults()

    assert tuple(optimizer.contract_diagnostics()) == ()


def test_island_component_configurations_have_no_diagnostics() -> None:
    model = IslandModel(
        (
            Optimizer(_problem()).set_strategy(DirectStrategy(n_offspring=4)),
            Optimizer(_problem()).set_strategy(DirectStrategy(n_offspring=4)),
        ),
        migration_interval=0,
    )

    for optimizer in model.optimizers:
        optimizer._resolve_defaults()
        assert tuple(optimizer.contract_diagnostics()) == ()


def test_pymoo_diagnostic_code_is_persistently_registered() -> None:
    assert DIAGNOSTIC_CODES.get("pymoo_partial_feedback_unsupported") is not None


def test_cors_composite_diagnostic_code_is_persistently_registered() -> None:
    assert DIAGNOSTIC_CODES.get("cors_composite_usage") is not None


def test_contract_diagnostics_are_not_called_per_generation(monkeypatch) -> None:
    optimizer = _default_optimizer()
    optimizer.set_termination(Termination(max_fe(1)))
    calls = 0
    real_contract_diagnostics = optimizer.contract_diagnostics

    def count_calls():
        nonlocal calls
        calls += 1
        return real_contract_diagnostics()

    monkeypatch.setattr(optimizer, "contract_diagnostics", count_calls)

    optimizer.run()

    assert calls == 1


def test_optimizer_compiles_once_per_run_and_retains_plan(monkeypatch) -> None:
    optimizer = _default_optimizer()
    optimizer.set_termination(Termination(max_fe(1)))
    calls = 0
    original_compile = Compiler.compile

    def count_compile(self, graph, context=None):
        nonlocal calls
        calls += 1
        return original_compile(self, graph, context)

    monkeypatch.setattr(Compiler, "compile", count_compile)

    optimizer.run()

    assert calls == 1
    assert optimizer.executable_plan is not None
    assert optimizer.describe() == optimizer.executable_plan.describe()


def _cors_optimizer(*, n_select: int, planner=None, n_candidates: int = 4) -> Optimizer:
    optimizer = _default_optimizer()
    optimizer.set_acquisition(CORSDistance(search_pattern=(0.9, 0.0)))
    optimizer.set_strategy(
        PreSelectionStrategy(n_candidates=n_candidates, n_select=n_select)
    )
    if planner is not None:
        optimizer.set_evaluation_planner(planner)
    return optimizer


def test_cors_compiler_warns_for_static_multi_candidate_top_k() -> None:
    optimizer = _cors_optimizer(n_select=2, planner=TopKEvaluation(2))

    plan = optimizer._compile_plan()

    assert plan is not None
    diagnostics = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "cors_nonsequential_evaluation"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].severity is Severity.WARNING
    assert (
        diagnostics[0].message
        == "CORSDistance is used outside the source-faithful sequential one-candidate "
        "evaluation cadence. Multiple candidates may share one decision, or distinct "
        "decisions may overlap. This configuration is supported, but does not "
        "reproduce the sequential CORS procedure."
    )
    assert diagnostics[0].resolutions == (
        "Use a configuration where CORSDistance directly selects one "
        "true-evaluated candidate per sequential decision, or accept the "
        "supported extension.",
    )


def test_cors_compiler_keeps_single_candidate_configuration_clean() -> None:
    optimizer = _cors_optimizer(n_select=1, planner=TopKEvaluation(1))

    plan = optimizer._compile_plan()

    assert plan is not None
    assert not any(
        diagnostic.code == "cors_nonsequential_evaluation"
        for diagnostic in plan.diagnostics
    )


def test_cors_composite_usage_warns_for_single_candidate_top_k() -> None:
    optimizer = _cors_optimizer(n_select=1, planner=TopKEvaluation(1))
    optimizer.set_acquisition(
        CompositeAcquisition(
            {
                "cors": CORSDistance(search_pattern=(0.9, 0.0)),
                "mean": MeanPrediction(),
            },
            combine_fn=lambda scores: scores[0] + scores[1],
        )
    )

    plan = optimizer._compile_plan()

    assert plan is not None
    diagnostics = [
        diagnostic
        for diagnostic in plan.diagnostics
        if diagnostic.code == "cors_composite_usage"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].severity is Severity.WARNING
    assert "combine_fn" in diagnostics[0].message
    assert "-inf" in diagnostics[0].message
    assert "CORSDistance alone" in diagnostics[0].message


def test_cors_composite_usage_is_clean_for_standalone_cors() -> None:
    optimizer = _cors_optimizer(n_select=1, planner=TopKEvaluation(1))

    plan = optimizer._compile_plan()

    assert plan is not None
    assert not any(
        diagnostic.code == "cors_composite_usage" for diagnostic in plan.diagnostics
    )


class _DiagnosticGraphComponent:
    def __init__(self, *, acquisition=None, planner=None, n_offspring=None):
        self._acquisition = acquisition
        self._planner = planner
        self._n_offspring = n_offspring

    def contract(self):
        return ComponentContract()


def test_cors_compiler_ignores_batch_planner_on_an_independent_branch() -> None:
    """Only planners reached by the CORS score can make it non-sequential."""
    graph = ComponentGraph(
        nodes=(
            ComponentNode(
                component_id="cors",
                component=_DiagnosticGraphComponent(
                    acquisition=CORSDistance(search_pattern=(0.9, 0.0))
                ),
            ),
            ComponentNode(
                component_id="cors_ask",
                component=_DiagnosticGraphComponent(n_offspring=1),
            ),
            ComponentNode(
                component_id="cors_planner",
                component=_DiagnosticGraphComponent(planner=TopKEvaluation(1)),
            ),
            ComponentNode(
                component_id="other_ask",
                component=_DiagnosticGraphComponent(n_offspring=10),
            ),
            ComponentNode(
                component_id="other_planner",
                component=_DiagnosticGraphComponent(planner=RatioEvaluation(1.0)),
            ),
        ),
        data_edges=(
            DataEdge(
                source=NodeRef(component_id="cors"),
                target=NodeRef(component_id="cors_planner"),
                source_port="scores",
                target_port="acquisition",
            ),
            DataEdge(
                source=NodeRef(component_id="cors_ask"),
                target=NodeRef(component_id="cors_planner"),
                source_port="candidates",
                target_port="candidates",
            ),
            DataEdge(
                source=NodeRef(component_id="other_ask"),
                target=NodeRef(component_id="other_planner"),
                source_port="candidates",
                target_port="candidates",
            ),
        ),
        entry_points=(NodeRef(component_id="cors"),),
    )
    result = cast(
        VerificationResult,
        CORSNonSequentialEvaluationRule().apply(
            RuleContext(graph, CompileContext(), DiagnosticBag())
        ),
    )

    assert not any(
        diagnostic.code == "cors_nonsequential_evaluation"
        for diagnostic in result.diagnostics
    )


def test_cors_compiler_keeps_individual_based_single_candidate_clean() -> None:
    optimizer = _default_optimizer()
    optimizer.set_acquisition(CORSDistance(search_pattern=(0.9, 0.0)))
    optimizer.set_strategy(IndividualBasedStrategy(evaluation_ratio=0.25))
    optimizer.set_evaluation_planner(TopKEvaluation(1))

    plan = optimizer._compile_plan()

    assert plan is not None
    assert not any(
        diagnostic.code == "cors_nonsequential_evaluation"
        for diagnostic in plan.diagnostics
    )


def test_individual_based_default_ratio_compiler_is_clean() -> None:
    """An unknown static ask count must not turn the default ratio into a warning."""
    optimizer = _default_optimizer()
    optimizer.set_acquisition(CORSDistance(search_pattern=(0.9, 0.0)))
    # IndividualBasedStrategy supplies its standard RatioEvaluation; no planner
    # override is used here.  At runtime this configuration evaluates one
    # candidate, while AskStage intentionally does not expose a static count.
    optimizer.set_strategy(IndividualBasedStrategy(evaluation_ratio=0.1))

    plan = optimizer._compile_plan()

    assert plan is not None
    assert not any(
        diagnostic.code == "cors_nonsequential_evaluation"
        for diagnostic in plan.diagnostics
    )


def test_cors_compiler_warns_for_generation_based_surrogate_cors() -> None:
    optimizer = _default_optimizer()
    optimizer.set_acquisition(CORSDistance(search_pattern=(0.9, 0.0)))
    optimizer.set_strategy(GenerationBasedStrategy(gen_ctrl=1))
    optimizer.set_evaluation_planner(TopKEvaluation(1))

    plan = optimizer._compile_plan()

    assert plan is not None
    assert any(
        diagnostic.code == "cors_nonsequential_evaluation"
        for diagnostic in plan.diagnostics
    )


def test_cors_compiler_warns_for_known_ratio_batch() -> None:
    optimizer = _cors_optimizer(n_select=1, planner=RatioEvaluation(0.75))

    plan = optimizer._compile_plan()

    assert plan is not None
    assert any(
        diagnostic.code == "cors_nonsequential_evaluation"
        for diagnostic in plan.diagnostics
    )


def test_cors_compiler_warns_for_repeated_full_candidate_batch() -> None:
    optimizer = _cors_optimizer(n_select=1, planner=RepeatedEvaluation(2))

    plan = optimizer._compile_plan()

    assert plan is not None
    assert any(
        diagnostic.code == "cors_nonsequential_evaluation"
        for diagnostic in plan.diagnostics
    )


def test_cors_compiler_keeps_repeated_single_candidate_clean() -> None:
    optimizer = _cors_optimizer(
        n_select=1,
        n_candidates=1,
        planner=RepeatedEvaluation(2),
    )

    plan = optimizer._compile_plan()

    assert plan is not None
    assert not any(
        diagnostic.code == "cors_nonsequential_evaluation"
        for diagnostic in plan.diagnostics
    )


def test_cors_compiler_keeps_async_overlap_capacity_clean() -> None:
    optimizer = _cors_optimizer(n_select=1, planner=TopKEvaluation(1))
    optimizer.set_async_evaluation_scheduler(
        AsyncEvaluationScheduler(SerialEvaluator(), max_pending=2)
    )

    plan = optimizer._compile_plan()

    assert plan is not None
    assert not any(
        diagnostic.code == "cors_nonsequential_evaluation"
        for diagnostic in plan.diagnostics
    )
