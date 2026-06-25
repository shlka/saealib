"""Abstract base class for evolutionary algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from saealib.context import OptimizationState
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem

if TYPE_CHECKING:
    from saealib.optimizer import Dispatchable


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

    @property
    def pareto_archive_class(self) -> type[ParetoArchive]:
        """Return the Pareto archive class."""
        return ParetoArchive

    def create_pareto_archive(
        self,
        attrs: list[PopulationAttribute],
        init_capacity: int,
        problem: Problem,
    ) -> ParetoArchive:
        """Create a ParetoArchive with the correct direction for the problem."""
        return self.pareto_archive_class(
            attrs=attrs,
            init_capacity=init_capacity,
            direction=problem.direction,
        )

    @property
    def ask_notation(self) -> list[str] | None:
        r"""LaTeX notation lines describing the internal steps of :meth:`ask`.

        Returns ``None`` by default; :class:`~saealib.stages.AskStage` then
        renders a single collapsed ``\\State`` line.  Override to return a list
        of LaTeX math strings (one per logical step) so that
        ``AskStage.to_pseudocode(expand=True)`` expands them inside a
        ``\\Comment`` block.

        Example (GA)::

            return [
                r"$I_m \\leftarrow \\mathrm{select}(P,\\, n_{pair})$",
                r"$\\mathcal{Q} \\leftarrow \\mathrm{crossover}(P[I_m])$",
                r"$\\mathcal{Q} \\leftarrow \\mathrm{mutate}(\\mathcal{Q})$",
            ]
        """
        return None

    @abstractmethod
    def ask(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        n_offspring: int | None = None,
    ) -> Population:
        """
        Generate offspring solutions.

        Parameters
        ----------
        ctx : OptimizationState
            Context instance.
        provider : Dispatchable
            Provider instance.
        n_offspring : int or None, optional
            Number of offspring to generate. If ``None``, the algorithm
            determines the count (typically equal to the population size).

        Returns
        -------
        Population
            Generated offspring solutions.
        """
        pass

    @property
    def tell_notation(self) -> list[str] | None:
        r"""LaTeX notation lines describing the internal steps of :meth:`tell`.

        Returns ``None`` by default; :class:`~saealib.stages.TellStage` then
        renders a single collapsed ``\State`` line.  Override to return a list
        of LaTeX math strings so that ``TellStage.to_pseudocode(expand=True)``
        expands them inside a ``\Comment`` block.
        """
        return None

    @abstractmethod
    def tell(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        offspring: Population,
    ) -> None:
        """
        Update the population with offspring solutions.

        Parameters
        ----------
        ctx : OptimizationState
            Context instance.
        provider : Dispatchable
            Provider instance.
        offspring : Population
            Offspring solutions.
        """
        pass
