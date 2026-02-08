"""
Evolutionary Algorithm Module.

This module contains the implementation of evolutionary algorithms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from saealib.context import OptimizationContext
from saealib.population import Archive, Population, PopulationAttribute
from saealib.problem import Problem

if TYPE_CHECKING:
    from saealib.optimizer import ComponentProvider


class Algorithm(ABC):
    """Base class for evolutionary algorithms."""

    @abstractmethod
    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Return the list of attributes required for Population."""
        pass

    @property
    @abstractmethod
    def population_class(self) -> type[Population]:
        """Return the population class."""
        pass

    @property
    @abstractmethod
    def archive_class(self) -> type[Archive]:
        """Return the archive class."""
        pass

    @abstractmethod
    def ask(self, ctx: OptimizationContext, provider: ComponentProvider) -> Population:
        """
        Generate offspring solutions.

        Parameters
        ----------
        ctx : OptimizationContext
            Context instance.
        provider : ComponentProvider
            Provider instance.

        Returns
        -------
        Population
            Generated offspring solutions.
        """
        pass

    @abstractmethod
    def tell(
        self,
        ctx: OptimizationContext,
        provider: ComponentProvider,
        offspring: Population,
    ) -> None:
        """
        Update the population with offspring solutions.

        Parameters
        ----------
        ctx : OptimizationContext
            Context instance.
        provider : ComponentProvider
            Provider instance.
        offspring : Population
            Offspring solutions.
        """
        pass
