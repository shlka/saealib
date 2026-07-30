"""Adapter exposing a registered DEAP mutation callable as saealib's Mutation."""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Protocol

import numpy as np

from saealib.exceptions import ValidationError
from saealib.operators._deap_rng import seeded_global_random
from saealib.operators.mutation import Mutation


class _DeapMutationLike(Protocol):
    """Structural interface of a toolbox-registered DEAP mutation callable.

    Matches the calling convention of DEAP's built-in mutation operators
    (e.g. ``deap.tools.mutPolynomialBounded``, ``deap.tools.mutGaussian``)
    once registered with all of their own hyperparameters (``eta``,
    ``low``, ``up``, ``indpb``, ``sigma``, ...) already bound -- e.g. via
    ``toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0,
    low=lb, up=ub, indpb=0.1)`` (pass ``toolbox.mutate`` as ``operator``),
    or an equivalent ``functools.partial``. This mirrors an
    already-constructed pymoo ``Mutation`` instance in
    :class:`PymooMutation`: the operator itself, not this adapter, owns
    those hyperparameters.
    """

    def __call__(self, individual: MutableSequence[float]) -> tuple[object]:
        """Mutate one individual in place and return it as a 1-tuple."""
        ...


class DeapMutation(Mutation):
    """
    Adapter wrapping a registered DEAP mutation callable as a saealib ``Mutation``.

    Lets researchers who already have a DEAP mutation operator (e.g.
    ``tools.mutPolynomialBounded``, ``tools.mutGaussian``) registered on a
    ``Toolbox`` reuse it unchanged inside saealib's ``GA``.

    Parameters
    ----------
    operator : callable
        An already-registered DEAP mutation callable (see
        ``_DeapMutationLike``), e.g. ``toolbox.mutate`` after
        ``toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0,
        low=lb, up=ub, indpb=0.1)``.
    prob : float, optional
        Individual-level mutation probability. Like ``PymooMutation``, this
        class applies the gate itself inside ``mutate_batch()``. Defaults
        to 1.0.

    Notes
    -----
    **Scope.** This adapter targets fixed-length numeric mutable sequences
    only -- the ``individual`` argument DEAP's mutation operators mutate in
    place. DEAP also supports GP trees, variable-length genomes, permutation
    genomes, and custom ``creator`` classes; none of that is in scope here.

    **``prob`` vs. ``indpb``: not conflated.** DEAP's own per-gene mutation
    rate -- ``mutPolynomialBounded``'s ``indpb`` parameter -- is a
    per-variable rate, distinct from saealib's individual-level ``prob``
    gate. ``self.prob_var`` (saealib's own per-variable-probability
    attribute) is deliberately **not** mirrored from, or forwarded to, the
    wrapped operator: it stays ``None`` here, exactly as
    ``PymooMutation.prob_var`` does, so that any saealib code reading it
    (e.g. mixed-variable routing in ``GA``) falls back to its own default
    rather than seeing a foreign value. ``indpb`` (and any other per-gene
    rate the wrapped operator exposes) must be bound directly on the
    operator at registration time, e.g. ``toolbox.register("mutate",
    tools.mutPolynomialBounded, ..., indpb=0.1)`` -- it is never merged
    with, or derived from, ``self.prob``.

    **Bounds are not forwarded.** ``mutate_batch()``'s own ``mutate_range``
    parameter is accepted for interface uniformity with the ``Mutation``
    ABC, but is **unused**: the wrapped DEAP callable must already have its
    own ``low``/``up`` (or equivalent) bound at registration time, and it is
    the caller's responsibility to keep those in sync with the problem's
    actual bounds.

    When binding ``low``/``up`` at registration, pass plain Python
    sequences (e.g. ``lb.tolist()``), not ``np.ndarray``: DEAP's own bounded
    operators branch on ``isinstance(low, collections.abc.Sequence)``,
    which a raw ``np.ndarray`` does not satisfy, silently changing how the
    bound is broadcast per dimension.

    **In-place mutation is copied away.** DEAP's mutation operators mutate
    their ``individual`` argument in place and return it (as a 1-tuple).
    Each gated row of ``candidates_batch`` is copied into a fresh Python
    list (via ``.tolist()``) before the call, so the caller's own array is
    never mutated by the wrapped operator. The returned tuple's arity and
    the mutated individual's shape/dtype are validated after the call; a
    malformed return raises :class:`~saealib.exceptions.ValidationError`.

    **RNG bridging.** DEAP's named operators call Python's global ``random``
    module directly -- there is no injectable generator to pass in.
    ``mutate_batch()`` first draws the individual-level ``prob`` gate from
    ``rng`` (mirroring ``PymooMutation.mutate_batch``'s draw order, so a
    parallel identically-seeded generator still predicts the gate), and
    only then, if any row is gated, seeds a fresh ``random`` state (derived
    from ``rng``, consuming one further ``rng`` draw) via
    :func:`saealib.operators._deap_rng.seeded_global_random` for the
    duration of the gated rows' DEAP calls, restoring the previous global
    ``random`` state in a ``finally`` block -- even if the wrapped operator
    raises. If no row is gated, ``random``'s global state is never touched
    and the wrapped operator is not called at all. This swap is
    process-global and **not safe under concurrent/multi-threaded use**.

    For more than one gated row, a loop of separate single-row
    ``mutate_batch`` calls (equivalently, separate :meth:`mutate` calls)
    does not reproduce the same per-row results as one batched call with the
    same seeded ``rng``: each separate call draws its own fresh seed and
    starts a new ``random`` stream, whereas one batched call draws a single
    seed and lets DEAP's per-row calls consume one continuous stream across
    all gated rows.
    """

    def __init__(self, operator: _DeapMutationLike, *, prob: float = 1.0) -> None:
        super().__init__()
        self.operator = operator
        self.prob = prob
        self.prob_var = None

    def mutate_batch(
        self,
        candidates_batch: np.ndarray,
        mutate_range: tuple,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute mutation on a batch of candidates via the wrapped DEAP operator.

        The wrapped operator is called once per gated row, under a single
        seeded ``random`` state.

        Parameters
        ----------
        candidates_batch : np.ndarray
            Batch of candidate individuals. shape = (n, dim)
        mutate_range : tuple
            Accepted for interface uniformity with the ``Mutation`` ABC,
            but unused -- see the class Notes.
        rng : np.random.Generator, optional
            Used to draw the ``prob`` gate and to derive the seed for the
            wrapped operator's calls (see the class Notes on RNG bridging).

        Returns
        -------
        np.ndarray
            Mutated individuals. shape = (n, dim)
        """
        candidates_batch = np.asarray(candidates_batch, dtype=float)
        n, dim = candidates_batch.shape
        gate = rng.random(n) < self.prob
        result = candidates_batch.copy()
        if not np.any(gate):
            return result
        with seeded_global_random(rng):
            for i in np.flatnonzero(gate):
                individual = candidates_batch[i].tolist()
                mutated = self.operator(individual)
                result[i] = self._validate_result(mutated, dim)
        return result

    def _validate_result(self, result: object, dim: int) -> np.ndarray:
        """Validate the wrapped operator's return value and coerce to a float array."""
        if not isinstance(result, tuple) or len(result) != 1:
            length = len(result) if hasattr(result, "__len__") else "unknown"
            raise ValidationError(
                "DeapMutation: wrapped operator must return a 1-tuple of "
                f"(mutated_individual,); got {type(result).__name__} of "
                f"length {length}."
            )
        try:
            arr = np.asarray(result[0], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "DeapMutation: wrapped operator's returned individual could "
                f"not be interpreted as a float array: {exc}"
            ) from exc
        if arr.shape != (dim,):
            raise ValidationError(
                "DeapMutation: wrapped operator's returned individual has "
                f"shape {arr.shape}, expected ({dim},)."
            )
        return arr
