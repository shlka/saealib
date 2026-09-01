"""Adapter for registered DEAP mutation callables."""

from __future__ import annotations

from collections.abc import MutableSequence
from dataclasses import replace
from typing import Protocol

import numpy as np

from saealib.core.contracts import ComponentContract
from saealib.exceptions import ValidationError
from saealib.operators._deap_rng import seeded_global_random
from saealib.operators.mutation import Mutation


class _DeapMutationLike(Protocol):
    """Structural interface of a registered DEAP mutation callable."""

    def __call__(self, individual: MutableSequence[float]) -> tuple[object]: ...


class DeapMutation(Mutation):
    """
    Wrap a registered DEAP mutation callable.

    Parameters
    ----------
    operator : callable
        DEAP callable with its hyperparameters and bounds already bound.
    prob : float, optional
        Individual-level mutation probability, by default 1.0.

    Notes
    -----
    The adapter supports fixed-length numeric mutable sequences only. GP
    trees, variable-length and permutation genomes, and custom DEAP creator
    classes are outside its scope. The callable owns its hyperparameters and
    bounds; ``mutate_range`` is accepted only for the base interface. DEAP
    mutates its arguments in place, so the adapter passes fresh list copies
    and uses one Python ``random`` bridge for all gated rows.
    """

    def __init__(self, operator: _DeapMutationLike, *, prob: float = 1.0) -> None:
        try:
            import deap as _deap

            del _deap
        except ImportError as exc:
            raise ImportError(
                "DeapMutation requires DEAP; install it with "
                "pip install 'saealib[deap]'"
            ) from exc
        self.operator = operator
        self.prob = prob
        self.prob_var = None

    def contract(self) -> ComponentContract:
        """Return a mutation contract without a bounds service requirement."""
        contract = super().contract()
        port = contract.ports["mutation"]
        inputs = tuple(
            replace(port_input, required_services=()) for port_input in port.inputs
        )
        return replace(contract, ports={"mutation": replace(port, inputs=inputs)})

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """Apply the DEAP callable to each row selected by ``prob``.

        Parameters
        ----------
        candidates_batch : numpy.ndarray
            Candidate individuals with shape ``(n, dim)``.
        mutate_range : tuple
            Accepted for the base interface and intentionally unused.
        rng : numpy.random.Generator, optional
            Source of gate draws and the bridged DEAP seed.

        Returns
        -------
        numpy.ndarray
            Mutated candidates with shape ``(n, dim)``.
        """
        try:
            candidates = np.asarray(candidates_batch, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError("DeapMutation candidates must be numeric.") from exc
        if candidates.ndim != 2:
            raise ValidationError("DeapMutation expects shape (n, dim).")
        n, dim = candidates.shape
        result = candidates.copy()
        gated = rng.random(n) < self.prob
        if not np.any(gated):
            return result
        with seeded_global_random(rng):
            for i in np.flatnonzero(gated):
                result[i] = self._validate_result(
                    self.operator(candidates[i].tolist()), dim
                )
        return result

    @staticmethod
    def _validate_result(result: object, dim: int) -> np.ndarray:
        """Validate and coerce the individual returned by DEAP."""
        if not isinstance(result, tuple) or len(result) != 1:
            raise ValidationError("DeapMutation operator must return a 1-tuple.")
        try:
            individual = np.asarray(result[0], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "DeapMutation returned a non-numeric individual."
            ) from exc
        if individual.shape != (dim,):
            raise ValidationError(
                "DeapMutation returned an individual with the wrong shape."
            )
        return individual
