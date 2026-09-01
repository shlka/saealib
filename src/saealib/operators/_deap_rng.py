"""Shared RNG-bridging helpers for saealib's DEAP adapters.

DEAP's toolbox-registered operators (e.g. ``tools.cxSimulatedBinaryBounded``,
``tools.mutPolynomialBounded``) call Python's global ``random`` module
directly -- there is no injectable generator to pass in. :func:`seeded_global_random`
seeds a fresh, saealib-``rng``-derived state into ``random`` for the duration of a
single adapter call and restores the previous global state afterward, so a
wrapped DEAP operator's randomness is still driven by saealib's own seeded
``rng`` end to end. This helper is used by the ``Crossover``/``Mutation``
adapters in this package.

``deap.cma.Strategy.generate()`` (used by
:class:`saealib.algorithms.deap_algorithm.DeapGenerateUpdateAlgorithm`) draws
from **numpy's** global RNG (``numpy.random.standard_normal``) instead of
Python's ``random`` module, so :func:`seeded_global_numpy_random` provides the
same bridging pattern for that case. Both helpers live in this one module
(despite the module living under ``operators/``) since they share the same
snapshot/seed/restore pattern and DEAP-specific rationale; the algorithm
adapter imports the numpy variant from here rather than duplicating it.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np


@contextmanager
def seeded_global_random(rng: np.random.Generator) -> Iterator[None]:
    """Temporarily seed Python's global ``random`` module from ``rng``.

    Parameters
    ----------
    rng : np.random.Generator
        saealib's own random number generator. A single integer seed is
        drawn from it (consuming exactly one draw from ``rng``) to seed a
        fresh ``random.Random`` state.

    Notes
    -----
    This swaps ``random``'s state -- which is process-global, not
    thread-local -- for the duration of the ``with`` block, and restores the
    previous state in a ``finally`` clause so restoration happens even if
    the wrapped call raises. **This is a process-global mutation and is not
    safe under concurrent or multi-threaded use**: two threads racing
    through this context manager can observe, or clobber, each other's
    seeded state.
    """
    saved_state = random.getstate()
    seed = int(rng.integers(0, 2**32))
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(saved_state)


@contextmanager
def seeded_global_numpy_random(rng: np.random.Generator) -> Iterator[None]:
    """Temporarily seed numpy's legacy global RNG (``numpy.random``) from ``rng``.

    Parameters
    ----------
    rng : np.random.Generator
        saealib's own random number generator. A single integer seed is
        drawn from it (consuming exactly one draw from ``rng``) to seed
        numpy's legacy global RNG singleton via ``numpy.random.seed``.

    Notes
    -----
    ``deap.cma.Strategy.generate()`` calls ``numpy.random.standard_normal``
    directly -- i.e. numpy's legacy global RNG singleton, not an injectable
    ``Generator`` and not Python's ``random`` module (see
    :func:`seeded_global_random` for that case). This swaps
    ``numpy.random``'s global state for the duration of the ``with`` block via
    ``numpy.random.get_state()``/``set_state()``, and restores the previous
    state in a ``finally`` clause so restoration happens even if the wrapped
    call raises. **This is a process-global mutation and is not safe under
    concurrent or multi-threaded use**: two threads racing through this
    context manager can observe, or clobber, each other's seeded state.
    """
    # Deliberate use of numpy's legacy global RNG (not np.random.Generator):
    # this bridges DEAP's own use of that same legacy singleton, which is the
    # whole point of this helper -- see the Notes above.
    saved_state = np.random.get_state()  # noqa: NPY002
    seed = int(rng.integers(0, 2**32))
    np.random.seed(seed)  # noqa: NPY002
    try:
        yield
    finally:
        np.random.set_state(saved_state)  # noqa: NPY002
