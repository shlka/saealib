"""Adapter exposing a registered DEAP crossover callable as saealib's Crossover."""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Any, Protocol

import numpy as np

from saealib.exceptions import ValidationError
from saealib.operators._deap_rng import seeded_global_random
from saealib.operators.crossover import Crossover


class _DeapCrossoverLike(Protocol):
    """Structural interface of a toolbox-registered DEAP crossover callable.

    Matches the calling convention of DEAP's built-in crossover operators
    (e.g. ``deap.tools.cxSimulatedBinaryBounded``, ``deap.tools.cxBlend``)
    once registered with all of their own hyperparameters (``eta``, ``low``,
    ``up``, ``alpha``, ...) already bound -- e.g. via
    ``toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0,
    low=lb, up=ub)`` (pass ``toolbox.mate`` as ``operator``), or an
    equivalent ``functools.partial``. This mirrors an already-constructed
    pymoo ``Crossover`` instance in :class:`PymooCrossover`: the operator
    itself, not this adapter, owns those hyperparameters.
    """

    def __call__(
        self,
        ind1: MutableSequence[float],
        ind2: MutableSequence[float],
    ) -> tuple[Any, Any]:
        """Cross two individuals in place and return them as a 2-tuple."""
        ...


class DeapCrossover(Crossover):
    """
    Adapter wrapping a registered DEAP crossover callable as a saealib ``Crossover``.

    Lets researchers who already have a DEAP crossover operator (e.g.
    ``tools.cxSimulatedBinaryBounded``, ``tools.cxBlend``) registered on a
    ``Toolbox`` reuse it unchanged inside saealib's ``GA``.

    Parameters
    ----------
    operator : callable
        An already-registered DEAP crossover callable (see
        ``_DeapCrossoverLike``), e.g. ``toolbox.mate`` after
        ``toolbox.register("mate", tools.cxSimulatedBinaryBounded,
        eta=20.0, low=lb, up=ub)``.
    prob : float, optional
        Individual-level crossover probability, gated by saealib's ``GA``
        before ``crossover_batch()``/``crossover()`` is even called (on the
        gated subset in batch mode, per gated pair in sequential mode).
        Defaults to 1.0.

    Notes
    -----
    **Scope.** This adapter targets fixed-length numeric mutable sequences
    only -- the ``ind1``/``ind2`` arguments DEAP's crossover operators
    mutate in place. DEAP also supports GP trees, variable-length genomes,
    permutation genomes, and custom ``creator`` classes; none of that is in
    scope here.

    **Fixed 2-parent, 2-child shape.** Unlike :class:`PymooCrossover`, this
    class does not expose ``n_parents``/``n_children`` constructor
    overrides: every DEAP crossover operator in this family takes exactly
    two individuals and returns exactly two, matching the ``Crossover`` ABC's
    own defaults (``n_parents = n_children = 2``).

    **Bounds are not forwarded.** ``crossover_batch()``'s own ``bounds``
    parameter is accepted for interface uniformity with the ``Crossover``
    ABC, but is **unused**: the wrapped DEAP callable must already have its
    own ``low``/``up`` (or equivalent) bound at registration time, and it is
    the caller's responsibility to keep those in sync with the problem's
    actual bounds. There is no reliable way to forward ``bounds`` into an
    arbitrary registered callable's keyword arguments without knowing its
    parameter names in advance.

    When binding ``low``/``up`` at registration, pass plain Python
    sequences (e.g. ``lb.tolist()``), not ``np.ndarray``: DEAP's own bounded
    operators branch on ``isinstance(low, collections.abc.Sequence)``,
    which a raw ``np.ndarray`` does not satisfy, silently changing how the
    bound is broadcast per dimension.

    **In-place mutation is copied away.** DEAP's crossover operators mutate
    their ``ind1``/``ind2`` arguments in place and return them (as a
    2-tuple). Each row of ``parents_batch`` is copied into a fresh Python
    list (via ``.tolist()``) before the call, so the caller's own array is
    never mutated by the wrapped operator. The returned tuple's arity and
    each child's shape/dtype are validated after the call; a malformed
    return raises :class:`~saealib.exceptions.ValidationError`.

    **RNG bridging.** DEAP's named operators call Python's global ``random``
    module directly -- there is no injectable generator to pass in. Each
    ``crossover_batch()`` call seeds a fresh ``random`` state (derived from
    ``rng``, consuming one ``rng`` draw) via
    :func:`saealib.operators._deap_rng.seeded_global_random`, lets the whole
    batch's DEAP calls run under that one seeded state, then restores the
    previous global ``random`` state in a ``finally`` block -- even if the
    wrapped operator raises. This swap is process-global and **not safe
    under concurrent/multi-threaded use**.

    For ``n_pair > 1``, a loop of separate single-pair ``crossover_batch``
    calls (equivalently, separate :meth:`crossover` calls) does not
    reproduce the same per-row results as one batched call with the same
    seeded ``rng``: each separate call draws its own fresh seed and starts a
    new ``random`` stream, whereas one batched call draws a single seed and
    lets DEAP's per-row calls consume one continuous stream across all rows.
    """

    def __init__(self, operator: _DeapCrossoverLike, *, prob: float = 1.0) -> None:
        super().__init__()
        self.operator = operator
        self.prob = prob

    def crossover_batch(
        self,
        parents_batch: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute crossover on a batch of parent pairs via the wrapped DEAP operator.

        The wrapped operator is called once per pair, under a single seeded
        ``random`` state.

        Parameters
        ----------
        parents_batch : np.ndarray
            Batch of parent pairs. shape = (n_pair, 2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Accepted for interface uniformity with the ``Crossover`` ABC,
            but unused -- see the class Notes.
        rng : np.random.Generator, optional
            Used to derive the seed for the wrapped operator's calls (see
            the class Notes on RNG bridging).

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (n_pair, 2, dim)
        """
        parents_batch = np.asarray(parents_batch, dtype=float)
        n_pair, _n_parents, dim = parents_batch.shape
        offspring = np.empty((n_pair, 2, dim), dtype=float)
        if n_pair == 0:
            return offspring
        with seeded_global_random(rng):
            for i in range(n_pair):
                ind1 = parents_batch[i, 0].tolist()
                ind2 = parents_batch[i, 1].tolist()
                result = self.operator(ind1, ind2)
                child1, child2 = self._validate_result(result, dim)
                offspring[i, 0] = child1
                offspring[i, 1] = child2
        return offspring

    def _validate_result(
        self, result: object, dim: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate the wrapped operator's return value and coerce to float arrays."""
        if not isinstance(result, tuple) or len(result) != 2:
            length = len(result) if hasattr(result, "__len__") else "unknown"
            raise ValidationError(
                "DeapCrossover: wrapped operator must return a 2-tuple of "
                f"(child1, child2); got {type(result).__name__} of length "
                f"{length}."
            )
        children = []
        for k, child in enumerate(result):
            try:
                arr = np.asarray(child, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "DeapCrossover: wrapped operator's returned child "
                    f"{k} could not be interpreted as a float array: {exc}"
                ) from exc
            if arr.shape != (dim,):
                raise ValidationError(
                    f"DeapCrossover: wrapped operator's returned child {k} "
                    f"has shape {arr.shape}, expected ({dim},)."
                )
            children.append(arr)
        return children[0], children[1]
