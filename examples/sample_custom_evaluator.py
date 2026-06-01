"""Custom Evaluator example: parallel batch evaluation.

Demonstrates how to plug a custom Evaluator into the optimizer via
``Optimizer.set_evaluator()``. The ``ParallelEvaluator`` below evaluates the
candidate batch concurrently with a thread pool; swap in a process pool,
joblib, or an MPI backend for true parallelism with picklable objectives.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

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
    RBFsurrogate,
    SequentialSelection,
    Termination,
    TruncationSelection,
    gaussian_kernel,
    max_fe,
)
from saealib.execution.evaluator import EvaluationResult, Evaluator

logging.basicConfig(level=logging.INFO)
logging.getLogger("saealib.surrogate.rbf").setLevel(logging.CRITICAL)


class ParallelEvaluator(Evaluator):
    """Evaluate a batch of candidates concurrently with a thread pool.

    Parameters
    ----------
    max_workers : int or None
        Maximum number of worker threads. None lets the executor choose.
    """

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def evaluate_batch(self, x: np.ndarray, problem: Problem) -> EvaluationResult:
        """Evaluate every row of ``x`` in parallel, preserving input order."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(x)
        f = np.empty((n, problem.n_obj), dtype=float)
        g = np.empty((n, problem.n_constraints), dtype=float)
        cv = np.zeros(n, dtype=float)

        def _eval(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            g_i, cv_i = problem.evaluate_constraints(xi)
            return problem.evaluate(xi), g_i, cv_i

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for i, (f_i, g_i, cv_i) in enumerate(pool.map(_eval, x)):
                f[i] = f_i
                g[i] = g_i
                cv[i] = cv_i

        return EvaluationResult(f=f, g=g, cv=cv)


def main():
    """Run SAGA-RBF optimization with a custom parallel evaluator."""
    # parameters
    dim = 10
    seed = 1
    knn = 50
    rsm = 0.1
    ub = [100] * dim
    lb = [-100] * dim

    # benchmark function
    f1 = cec2015.F12015(ndim=10)

    problem = Problem(
        func=f1.evaluate,
        dim=dim,
        n_obj=1,
        weight=np.array([-1.0]),
        lb=lb,
        ub=ub,
    )
    initializer = LHSInitializer(
        n_init_archive=5 * dim,
        n_init_population=4 * dim,
        seed=seed,
    )
    algorithm = GA(
        crossover=CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4),
        mutation=MutationUniform(mutation_rate=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    )
    termination = Termination(max_fe(200 * dim))
    surrogate = RBFsurrogate(gaussian_kernel, dim)
    strategy = IndividualBasedStrategy(evaluation_ratio=rsm)

    opt = (
        Optimizer(problem)
        .set_initializer(initializer)
        .set_algorithm(algorithm)
        .set_termination(termination)
        .set_surrogate(surrogate, n_neighbors=knn)
        .set_strategy(strategy)
        .set_evaluator(ParallelEvaluator(max_workers=4))
    )
    opt.run()


if __name__ == "__main__":
    main()
