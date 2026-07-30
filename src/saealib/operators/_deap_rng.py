"""Shared RNG-bridging helper for the DEAP crossover/mutation adapters.

DEAP's toolbox-registered operators (e.g. ``tools.cxSimulatedBinaryBounded``,
``tools.mutPolynomialBounded``) call Python's global ``random`` module
directly -- there is no injectable generator to pass in. This module seeds a
fresh, saealib-``rng``-derived state into ``random`` for the duration of a
single adapter call and restores the previous global state afterward, so a
wrapped DEAP operator's randomness is still driven by saealib's own seeded
``rng`` end to end.
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
