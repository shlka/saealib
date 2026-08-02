"""Independent optimizer groups following the island-model pattern."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.execution.runner import Runner

if TYPE_CHECKING:
    from saealib.optimizer import Optimizer


__all__ = ["IslandModel"]


class IslandModel:
    """Group independent optimizers and run each one to completion."""

    def __init__(self, optimizers: Iterable[Optimizer] = ()) -> None:
        self.optimizers = tuple(optimizers)
        if any(not hasattr(optimizer, "strategy") for optimizer in self.optimizers):
            raise ValidationError("optimizers must be configured Optimizer instances")

    def run(self) -> tuple[OptimizationState, ...]:
        """Run each configured island and return its final state."""
        return tuple(Runner(optimizer).run() for optimizer in self.optimizers)
