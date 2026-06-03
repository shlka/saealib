"""
Evaluator abstraction.

An ``Evaluator`` is the single entry point through which Strategies and
Initializers turn a batch of design vectors into objective values, raw
constraint values, and aggregate constraint violations. Centralizing
evaluation here enables pluggable execution backends (serial, parallel, ...)
without touching the pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from saealib.problem import Problem


@dataclass
class EvaluationResult:
    """
    Batched result of evaluating a set of design vectors.

    Attributes
    ----------
    f : np.ndarray
        Objective values. shape = (n, n_obj)
    g : np.ndarray
        Raw constraint values. shape = (n, n_constraints).
        Has shape (n, 0) when the problem defines no constraints.
    cv : np.ndarray
        Aggregate constraint violation per candidate. shape = (n, )
        All zeros when the problem defines no constraints.
    """

    f: np.ndarray
    g: np.ndarray
    cv: np.ndarray


class Evaluator(ABC):
    """Base class for batch evaluators."""

    @abstractmethod
    def evaluate_batch(self, x: np.ndarray, problem: Problem) -> EvaluationResult:
        """
        Evaluate a batch of design vectors.

        Parameters
        ----------
        x : np.ndarray
            Design vectors to evaluate. shape = (n, dim)
        problem : Problem
            The optimization problem providing the objective and constraints.

        Returns
        -------
        EvaluationResult
            Batched objective values, raw constraint values, and violations.
        """
        ...


class SerialEvaluator(Evaluator):
    """Default evaluator: evaluates each candidate sequentially."""

    def evaluate_batch(self, x: np.ndarray, problem: Problem) -> EvaluationResult:
        """
        Evaluate each row of ``x`` one at a time.

        Produces results equivalent to the per-candidate evaluation loops that
        previously lived in each Strategy and Initializer.

        Parameters
        ----------
        x : np.ndarray
            Design vectors to evaluate. shape = (n, dim)
        problem : Problem
            The optimization problem providing the objective and constraints.

        Returns
        -------
        EvaluationResult
            Batched objective values, raw constraint values, and violations.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(x)
        n_constraints = problem.n_constraints

        f = np.empty((n, problem.n_obj), dtype=float)
        g = np.empty((n, n_constraints), dtype=float)
        cv = np.zeros(n, dtype=float)

        for i, xi in enumerate(x):
            g_i, cv_i = problem.evaluate_constraints(xi)
            f[i] = problem.evaluate(xi, g_i)
            g[i] = g_i
            cv[i] = cv_i

        return EvaluationResult(f=f, g=g, cv=cv)
