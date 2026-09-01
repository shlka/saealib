import logging

import numpy as np
from integration.test_async_workflow import (
    ReattachEvaluator,
    make_state,
    requests,
)

from saealib import minimize
from saealib.execution import AsyncEvaluationScheduler


def test_minimize_logs_resolved_bundled_defaults_and_run_boundaries(caplog):
    with caplog.at_level(logging.DEBUG, logger="saealib.optimizer"):
        minimize(
            lambda x: np.sum(x**2),
            dim=1,
            lb=[-1.0],
            ub=[1.0],
            max_fe=1,
            seed=0,
            verbose=False,
        )

    records = [
        record for record in caplog.records if record.name == "saealib.optimizer"
    ]
    resolved = next(
        record
        for record in records
        if record.levelno == logging.DEBUG
        and record.getMessage().startswith("Resolved defaults")
    )
    message = resolved.getMessage()
    assert "bundled_preset=ga_rbf_ib" in message
    for component_type in (
        "GA",
        "IndividualBasedStrategy",
        "LocalSurrogateManager",
        "MeanPrediction",
        "RatioEvaluation",
        "ComparatorWorstFallback",
    ):
        assert component_type in message
    assert any(
        record.levelno == logging.INFO
        and "Optimization run started" in record.getMessage()
        for record in records
    )
    assert any(
        record.levelno == logging.INFO
        and "Optimization run finished" in record.getMessage()
        for record in records
    )


def test_async_scheduler_logs_timeout_warning(caplog):
    class CancellingEvaluator(ReattachEvaluator):
        def cancel(self, handle):
            del handle
            return True

    with caplog.at_level(logging.WARNING, logger="saealib.execution.scheduler"):
        scheduler = AsyncEvaluationScheduler(CancellingEvaluator(), timeout=0)
        state = scheduler.submit(make_state(), [requests()[0]])
        scheduler.poll(state, wait=False)

    warnings = [
        record
        for record in caplog.records
        if record.name == "saealib.execution.scheduler"
        and record.levelno == logging.WARNING
    ]
    assert any(
        "Evaluation timed out" in record.getMessage()
        and "request_id=0" in record.getMessage()
        for record in warnings
    )
