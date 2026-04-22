"""
Event classes for the optimization lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from saealib.context import OptimizationContext
    from saealib.optimizer import ComponentProvider
    from saealib.population import Population
    from saealib.surrogate.base import Surrogate


@dataclass
class Event:
    """
    Base class for all callback events.

    Attributes
    ----------
    ctx : OptimizationContext
        The current optimization context.
    provider : ComponentProvider
        The component provider (e.g. Optimizer) that fired this event.
    """

    ctx: OptimizationContext
    provider: ComponentProvider


# --- Optimizer.run events ---


@dataclass
class RunStartEvent(Event):
    """Fired once when the optimization run starts."""


@dataclass
class RunEndEvent(Event):
    """Fired once when the optimization run ends."""


@dataclass
class GenerationStartEvent(Event):
    """Fired at the beginning of each generation."""


@dataclass
class GenerationEndEvent(Event):
    """Fired at the end of each generation, before yielding the context."""


# --- Surrogate events ---


@dataclass
class SurrogateStartEvent(Event):
    """Fired before surrogate-based candidate scoring."""

    offspring: Population | None = None


@dataclass
class SurrogateEndEvent(Event):
    """Fired after surrogate-based candidate scoring."""

    offspring: Population | None = None


# --- Algorithm.ask events ---


@dataclass
class PostCrossoverEvent(Event):
    """
    Fired after crossover and repair.

    Handlers may replace ``candidates`` with a modified array.
    """

    candidates: np.ndarray | None = None


@dataclass
class PostMutationEvent(Event):
    """
    Fired after mutation and repair.

    Handlers may replace ``candidates`` with a modified array.
    """

    candidates: np.ndarray | None = None


@dataclass
class PostAskEvent(Event):
    """
    Fired after the full ask step (post crossover and mutation).

    Handlers may replace ``candidates`` with a modified array.
    """

    candidates: np.ndarray | None = None


# --- Model events ---


@dataclass
class PostSurrogateFitEvent(Event):
    """Fired after the surrogate model is fitted.

    Attributes
    ----------
    surrogate : Surrogate or None
        The surrogate model that was just fitted.
    train_x : np.ndarray or None
        Design variable matrix used for fitting, shape (n_train, dim).
    train_f : np.ndarray or None
        Objective value matrix used for fitting, shape (n_train, n_obj).
    """

    surrogate: Surrogate | None = None
    train_x: np.ndarray | None = None
    train_f: np.ndarray | None = None


@dataclass
class PostEvaluationEvent(Event):
    """Fired after true evaluation of selected candidates.

    Attributes
    ----------
    offspring : Population or None
        The candidates that were evaluated with the true objective function.
        All individuals in this population have true objective values assigned.
    """

    offspring: Population | None = None
