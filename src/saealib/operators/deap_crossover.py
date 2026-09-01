"""Adapter for registered DEAP crossover callables."""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Protocol

import numpy as np

from saealib.exceptions import ValidationError
from saealib.operators._deap_rng import seeded_global_random
from saealib.operators.crossover import Crossover


class _DeapCrossoverLike(Protocol):
    """Structural interface of a registered DEAP crossover callable."""

    def __call__(
        self, ind1: MutableSequence[float], ind2: MutableSequence[float]
    ) -> tuple[object, object]: ...


class DeapCrossover(Crossover):
    """
    Wrap a registered DEAP crossover callable.

    Parameters
    ----------
    operator : callable
        DEAP callable with its hyperparameters and bounds already bound.
    prob : float, optional
        Individual-level crossover probability, by default 1.0.

    Notes
    -----
    The adapter supports fixed-length numeric mutable sequences only. GP
    trees, variable-length and permutation genomes, and custom DEAP creator
    classes are outside its scope. The callable owns its hyperparameters and
    bounds; ``bounds`` is accepted only for the base interface. DEAP mutates
    its arguments in place, so the adapter passes fresh list copies and uses
    one Python ``random`` bridge for the whole batch.
    """

    def __init__(self, operator: _DeapCrossoverLike, *, prob: float = 1.0) -> None:
        try:
            import deap as _deap

            del _deap
        except ImportError as exc:
            raise ImportError(
                "DeapCrossover requires DEAP; install it with "
                "pip install 'saealib[deap]'"
            ) from exc
        self.operator = operator
        self.prob = prob

    def crossover_batch(
        self,
        parents_batch: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """Apply the DEAP callable once to each pair in ``parents_batch``.

        Parameters
        ----------
        parents_batch : numpy.ndarray
            Parent pairs with shape ``(n_pair, 2, dim)``.
        bounds : tuple of numpy.ndarray or None, optional
            Accepted for the base interface and intentionally unused.
        rng : numpy.random.Generator, optional
            Source of the bridged DEAP seed.

        Returns
        -------
        numpy.ndarray
            Offspring with shape ``(n_pair, 2, dim)``.
        """
        try:
            parents = np.asarray(parents_batch, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError("DeapCrossover parents must be numeric.") from exc
        if parents.ndim != 3 or parents.shape[1] != 2:
            raise ValidationError("DeapCrossover expects shape (n_pair, 2, dim).")
        n_pair, _, dim = parents.shape
        offspring = np.empty((n_pair, 2, dim), dtype=float)
        if n_pair == 0:
            return offspring
        with seeded_global_random(rng):
            for i in range(n_pair):
                result = self.operator(parents[i, 0].tolist(), parents[i, 1].tolist())
                child1, child2 = self._validate_result(result, dim)
                offspring[i, 0] = child1
                offspring[i, 1] = child2
        return offspring

    @staticmethod
    def _validate_result(result: object, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Validate and coerce the two children returned by DEAP."""
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValidationError("DeapCrossover operator must return a 2-tuple.")
        children: list[np.ndarray] = []
        for child in result:
            try:
                array = np.asarray(child, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "DeapCrossover returned a non-numeric child."
                ) from exc
            if array.shape != (dim,):
                raise ValidationError(
                    "DeapCrossover returned a child with the wrong shape."
                )
            children.append(array)
        return children[0], children[1]
