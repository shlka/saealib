"""InitialEvaluationEvent callbacks example.

Demonstrates how to hook into the initial archive evaluation phase via
``InitialEvaluationStartEvent`` and ``InitialEvaluationEndEvent``.

Two callbacks are registered:
- Start: logs the number and range of sampled candidates.
- End:   logs initial archive statistics and removes outliers whose
         objective value exceeds mean + 2 * std (outlier trimming).

A ``JoblibEvaluator`` is used so the initial batch evaluation runs in
parallel across all available cores.
"""

import logging

import numpy as np
from opfunu.cec_based import cec2015

from saealib import (
    GA,
    CrossoverBLXAlpha,
    IndividualBasedStrategy,
    LHSInitializer,
    MutationUniform,
    Optimizer,
    Problem,
    RBFSurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    gaussian_kernel,
    max_fe,
)
from saealib.callback import InitialEvaluationEndEvent, InitialEvaluationStartEvent
from saealib.execution.evaluator import JoblibEvaluator

logging.basicConfig(level=logging.INFO)
logging.getLogger("saealib.surrogate.rbf").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


def on_initial_evaluation_start(event: InitialEvaluationStartEvent) -> None:
    """Log the number and range of sampled candidates."""
    x = event.candidates_x
    logger.info(
        "Initial evaluation: %d candidates, x in [%.3f, %.3f]",
        len(x),
        x.min(),
        x.max(),
    )


def on_initial_evaluation_end(event: InitialEvaluationEndEvent) -> None:
    """Log archive statistics and remove outliers beyond mean + 2*std."""
    archive = event.archive
    f_vals = np.array([archive[i].f[0] for i in range(len(archive))])

    logger.info(
        "Initial archive: best=%.4f  worst=%.4f  mean=%.4f",
        f_vals.min(),
        f_vals.max(),
        f_vals.mean(),
    )

    # Remove outliers: keep individuals within mean + 2*std
    threshold = f_vals.mean() + 2.0 * f_vals.std()
    keep_idx = [i for i, v in enumerate(f_vals) if v <= threshold]
    if len(keep_idx) < len(archive):
        kept = archive.extract(keep_idx)
        archive.clear()
        archive.extend(kept)
        logger.info(
            "Trimmed %d outlier(s); archive size: %d -> %d",
            len(f_vals) - len(keep_idx),
            len(f_vals),
            len(archive),
        )


def main():
    """Run SAGA-RBF optimization with initial evaluation callbacks."""
    dim = 10
    seed = 1
    knn = 50
    rsm = 0.1
    ub = [100] * dim
    lb = [-100] * dim

    f1 = cec2015.F12015(ndim=10)

    problem = Problem(
        func=f1.evaluate,
        dim=dim,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=lb,
        ub=ub,
    )

    opt = (
        Optimizer(problem)
        .set_initializer(
            LHSInitializer(n_init_archive=5 * dim, n_init_population=4 * dim, seed=seed)
        )
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4),
                mutation=MutationUniform(mutation_rate=0.3),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_termination(Termination(max_fe(200 * dim)))
        .set_surrogate(RBFSurrogate(gaussian_kernel, dim), n_neighbors=knn)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=rsm))
        .set_evaluator(JoblibEvaluator())
    )

    opt.cbmanager.register(InitialEvaluationStartEvent, on_initial_evaluation_start)
    opt.cbmanager.register(InitialEvaluationEndEvent, on_initial_evaluation_end)

    opt.run()


if __name__ == "__main__":
    main()
