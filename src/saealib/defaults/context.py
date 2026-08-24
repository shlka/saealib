"""DefaultContext for default resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from saealib.problem import Problem


@dataclass(frozen=True)
class DefaultContext:
    """Context for default resolution.

    Provides problem information and component references to hint providers.
    """

    problem: Problem
    seed: int | None = None
    components: dict[str, Any] | None = None

    @property
    def dim(self) -> int:
        """Problem dimension."""
        return self.problem.dim

    @property
    def n_obj(self) -> int:
        """Number of objectives."""
        return self.problem.n_obj

    @property
    def comparator(self) -> Any:
        """Problem comparator."""
        return self.problem.comparator
