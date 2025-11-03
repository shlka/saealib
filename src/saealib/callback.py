import logging
from enum import Enum, auto
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .optimizer import Optimizer


class CallbackEvent(Enum):
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
    def __init__(self):
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
    logging.info(f"Generation {optimizer.gen} started. fe: {optimizer.fe}. Best f: {optimizer.archive.get('y').min()}")
