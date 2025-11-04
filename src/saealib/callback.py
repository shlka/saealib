"""
Callback module.

This module contains the implementation of callback events and manager.
"""
from __future__ import annotations

import logging
from enum import Enum, auto
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saealib.optimizer import Optimizer


logger = logging.getLogger(__name__)


class CallbackEvent(Enum):
    """
    Enum class for callback events.

    Attributes
    ----------
    RUN_START
        Triggered when the optimization run starts.
    RUN_END
        Triggered when the optimization run ends.
    GENERATION_START
        Triggered when a new generation starts.
    GENERATION_END
        Triggered when a generation ends.
    SURROGATE_START
        Triggered when surrogate model training starts.
    SURROGATE_END
        Triggered when surrogate model training ends.
    POST_CROSSOVER
        Triggered after crossover operation.
    POST_MUTATION
        Triggered after mutation operation.
    POST_SURROGATE_FIT
        Triggered after surrogate model fitting.
    """
    # Optimizer.run events
    RUN_START = auto()
    RUN_END = auto()
    GENERATION_START = auto()
    GENERATION_END = auto()
    SURROGATE_START = auto()
    SURROGATE_END = auto()
    # Algorithm.ask events
    POST_CROSSOVER = auto()
    POST_MUTATION = auto()
    # ModelManager.run events (commented out for future use)
    POST_SURROGATE_FIT = auto()
    # POST_SURROGATE_PREDICT = auto()


class CallbackManager:
    """
    Manages callback events and their handlers.
    """
    def __init__(self):
        """
        Initialize CallbackManager.

        Attributes
        ----------
        handlers : dict

        """
        self.handlers = defaultdict(list)

    def register(self, event: CallbackEvent, func: callable):
        self.handlers[event].append(func)

    def dispatch(self, event: CallbackEvent, data, **kwargs):
        cur_data = data
        for handler in self.handlers[event]:
            cur_data = handler(data=cur_data, **kwargs)
        return cur_data


def logging_generation(data, **kwargs):
    optimizer: Optimizer = kwargs.get("optimizer", None)
    logger.info(f"Generation {optimizer.gen} started. fe: {optimizer.fe}. Best f: {optimizer.archive.get('y').min()}")
