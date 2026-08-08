"""Abstract base class for evolutionary algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

import numpy as np

from saealib.context import OptimizationState
from saealib.core.contracts import (
    MANY,
    TRUE,
    ComponentContract,
    DataSpec,
    FeedbackBatch,
    FeedbackContract,
    FeedbackRequirement,
    LifecycleContract,
    PortContract,
    PortDirection,
    PortSpec,
    ProposalBatch,
    ProposalRelations,
    ServiceRequirement,
    StateContract,
)
from saealib.core.state import (
    POPULATIONS_MAIN,
    RUNTIME_RNG,
    LegacyAlgorithmStateView,
    StatePatch,
    StateView,
)
from saealib.exceptions import ValidationError
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem

if TYPE_CHECKING:
    from saealib.optimizer import Dispatchable


class _NoOpDispatchProvider:
    """Fallback event provider for legacy algorithms used without callbacks."""

    def dispatch(self, event: object) -> None:
        return None


class Algorithm(ABC):
    """Base class for evolutionary algorithms."""

    def contract(self) -> ComponentContract:
        """Return the evolutionary-algorithm family contract."""
        return ComponentContract(
            ports={
                "proposer": PortContract(
                    outputs=(
                        PortSpec(
                            name="genomes",
                            direction=PortDirection.OUTPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                            required_services=(
                                ServiceRequirement(name="BoundsService"),
                            ),
                        ),
                    ),
                ),
                "feedback_consumer": PortContract(
                    inputs=(
                        PortSpec(
                            name="offspring",
                            direction=PortDirection.INPUT,
                            data=DataSpec(kind="Population"),
                            cardinality=MANY,
                        ),
                    ),
                ),
            },
            state=StateContract(
                reads=(POPULATIONS_MAIN, RUNTIME_RNG),
                writes=(POPULATIONS_MAIN, RUNTIME_RNG),
            ),
            lifecycle=LifecycleContract(
                feedback=FeedbackContract(accepted_channels=frozenset({TRUE}))
            ),
        )

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


@dataclass(frozen=True, kw_only=True)
class ProposalRequest:
    """The request-specific part of the proposer interface."""

    n_offspring: int | None = None

    def __post_init__(self) -> None:
        """Validate the one request parameter carried across the new boundary."""
        if self.n_offspring is None:
            return
        if isinstance(self.n_offspring, (bool, np.bool_)) or not isinstance(
            self.n_offspring, (int, np.integer)
        ):
            raise ValidationError("n_offspring must be a non-negative integer or None")
        n_offspring = int(self.n_offspring)
        if n_offspring < 0:
            raise ValidationError("n_offspring must be a non-negative integer or None")
        object.__setattr__(self, "n_offspring", n_offspring)


@runtime_checkable
class Proposer(Protocol):
    """Generate proposals without owning the feedback lifecycle."""

    def ask(self, request: ProposalRequest, state: StateView) -> ProposalBatch:
        """Return a batch of candidates and its feedback requirement."""
        ...


@runtime_checkable
class FeedbackConsumer(Protocol):
    """Consume feedback independently from proposal generation."""

    def tell(self, feedback: FeedbackBatch, state: StateView) -> StatePatch:
        """Return state changes caused by a feedback batch."""
        ...


@dataclass(frozen=True, slots=True)
class _LegacyAdapterDerived:
    """Immutable contract-derived values shared by legacy adapter instances."""

    contract: ComponentContract
    requirements: FeedbackRequirement


def _derive_legacy_adapter_values(
    algorithm: Any,
    requirements: FeedbackRequirement | None = None,
) -> _LegacyAdapterDerived:
    """Build immutable values needed to initialize a legacy adapter."""
    contract = getattr(algorithm, "contract", None)
    legacy_contract = (
        contract() if callable(contract) else Algorithm.contract(algorithm)
    )
    feedback_contract = legacy_contract.lifecycle.feedback
    if feedback_contract is None:
        raise ValidationError("legacy algorithm must declare a feedback contract")
    if requirements is None:
        requirements = FeedbackRequirement(
            quantities=(), completion=feedback_contract.completion
        )
    elif not feedback_contract.contains_requirement(requirements):
        raise ValidationError(
            "legacy algorithm requirements are wider than its feedback contract"
        )
    return _LegacyAdapterDerived(
        contract=legacy_contract,
        requirements=requirements,
    )


class LegacyPopulationAlgorithmAdapter:
    """Adapt one legacy :class:`Algorithm` to the proposer-only interface.

    This is a Phase 4-10 migration adapter and is removed in Phase 11 together
    with :class:`LegacyAlgorithmStateView`.
    """

    def __init__(
        self,
        algorithm: Any,
        provider: Dispatchable | None = None,
        requirements: FeedbackRequirement | None = None,
        *,
        _derived: _LegacyAdapterDerived | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.provider = provider if provider is not None else _NoOpDispatchProvider()
        derived = (
            _derived
            if _derived is not None
            else _derive_legacy_adapter_values(algorithm, requirements)
        )
        self.requirements = derived.requirements

    @classmethod
    def for_stage(
        cls,
        algorithm: Any,
        provider: Dispatchable,
    ) -> LegacyPopulationAlgorithmAdapter:
        """Build an adapter from cached immutable values while pipelines rebuild.

        Strategy ``step()`` methods intentionally rebuild ``AskStage`` every
        generation.  Cache only the contract-derived values; each stage gets a
        fresh adapter so its provider remains local to that stage.
        """
        try:
            derived = _LEGACY_ADAPTER_DERIVED_CACHE.get(algorithm)
        except TypeError:
            return cls(algorithm, provider)
        if derived is None:
            derived = _derive_legacy_adapter_values(algorithm)
            _LEGACY_ADAPTER_DERIVED_CACHE[algorithm] = derived
        return cls(algorithm, provider, _derived=derived)

    @property
    def ask_notation(self) -> list[str] | None:
        """Preserve the wrapped algorithm's pseudocode notation."""
        return self.algorithm.ask_notation

    def ask(self, request: ProposalRequest, state: StateView) -> ProposalBatch:
        """Call old ask once, using the named legacy OptimizationState seam."""
        if not isinstance(request, ProposalRequest):
            raise ValidationError("legacy proposer requires a ProposalRequest")
        if not isinstance(state, LegacyAlgorithmStateView):
            raise ValidationError("legacy proposer requires a LegacyAlgorithmStateView")
        context = state.legacy_optimization_state
        candidates = self.algorithm.ask(context, self.provider, request.n_offspring)
        if not isinstance(candidates, Population):
            raise ValidationError("legacy algorithm ask() must return a Population")
        return ProposalBatch.from_allocator(
            context.request_id_allocator,
            candidates=candidates,
            relations=ProposalRelations({}, row_count=len(candidates)),
            requirements=self.requirements,
        )


_LEGACY_ADAPTER_DERIVED_CACHE: WeakKeyDictionary[Any, _LegacyAdapterDerived] = (
    WeakKeyDictionary()
)
