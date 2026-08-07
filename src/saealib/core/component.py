from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from saealib.core.contracts.contract import ComponentContract

__all__ = ["Component"]


class Component(Protocol):
    """Declare the contract supplied by a component."""

    def contract(self) -> ComponentContract:
        """Return the component's pure contract."""
