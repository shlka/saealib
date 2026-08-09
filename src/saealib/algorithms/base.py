"""Abstract base class for evolutionary algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

import numpy as np

from saealib.context import OptimizationState
from saealib.core.contracts import (
    CONSTRAINT,
    CV,
    MANY,
    OBJECTIVE,
    TRUE,
    ComponentContract,
    DataSpec,
    FeedbackBatch,
    FeedbackContract,
    FeedbackRequirement,
    LifecycleContract,
    ObservationBatch,
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
        self._state_reads = derived.contract.state
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
        return getattr(self.algorithm, "ask_notation", None)

    @property
    def tell_notation(self) -> list[str] | None:
        """Preserve the wrapped algorithm's tell pseudocode notation."""
        return getattr(self.algorithm, "tell_notation", None)

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
            context.proposal_id_allocator,
            candidates=candidates,
            relations=ProposalRelations({}, row_count=len(candidates)),
            requirements=self.requirements,
        )

    def tell(self, feedback: FeedbackBatch, state: StateView) -> StatePatch:
        """Call old ``tell`` and deliberately return an empty state patch.

        The old ``tell(ctx, provider, offspring)`` mutates ``ctx.population``
        in place.  Returning an empty patch is intentional: routing that
        mutation through ``StatePatch`` would change the established behavior.
        Whether state threading becomes patch-only is the Phase 6 question
        recorded as ADR-index D-6; the migration adapter must not decide it.
        """
        if not isinstance(feedback, FeedbackBatch):
            raise ValidationError("legacy consumer requires a FeedbackBatch")
        if not isinstance(state, LegacyAlgorithmStateView):
            raise ValidationError("legacy consumer requires a LegacyAlgorithmStateView")
        context = state.legacy_optimization_state
        offspring = _legacy_population_from_observations(context, feedback.observations)
        self.algorithm.tell(context, self.provider, offspring)
        return StatePatch(writes={})


_LEGACY_ADAPTER_DERIVED_CACHE: WeakKeyDictionary[Any, _LegacyAdapterDerived] = (
    WeakKeyDictionary()
)


def _legacy_population_from_observations(
    context: OptimizationState,
    observations: ObservationBatch,
) -> Population:
    """Build the old Population input from columnar observations."""
    owner = context.offspring
    if owner is None:
        raise ValidationError("legacy consumer requires an offspring population")
    records = observations.records
    payload = np.asarray(records.column("subject_payload"))
    if payload.ndim == 1:
        record_ids = np.asarray(payload, dtype=np.int64)
    elif payload.ndim == 2 and payload.shape[1] == 1:
        record_ids = np.asarray(payload[:, 0], dtype=np.int64)
    else:
        raise ValidationError(
            "legacy consumer requires one candidate ID per observation subject"
        )
    if len(record_ids):
        _, first = np.unique(record_ids, return_index=True)
        candidate_ids = record_ids[np.sort(first)]
    else:
        candidate_ids = np.empty(0, dtype=np.int64)
    quantity_kinds = np.asarray(records.column("quantity_kind"))
    quantity_indices = np.asarray(records.column("quantity_index"), dtype=np.intp)
    values = np.asarray(records.column("value"), dtype=np.float64)
    if len(values) != len(record_ids):
        raise ValidationError("observation columns are not row-aligned")
    if len(candidate_ids):
        order = np.argsort(candidate_ids, kind="stable")
        sorted_ids = candidate_ids[order]
        positions = np.searchsorted(sorted_ids, record_ids)
        valid = (positions < len(sorted_ids)) & (
            sorted_ids[np.minimum(positions, len(sorted_ids) - 1)] == record_ids
        )
        if not np.all(valid):
            raise ValidationError("observation subject IDs are inconsistent")
        record_positions = order[positions]
    else:
        record_positions = np.empty(0, dtype=np.intp)

    objective_count = len(observations.schema.indices(OBJECTIVE))
    constraint_count = len(observations.schema.indices(CONSTRAINT))
    f = np.full((len(candidate_ids), objective_count), np.nan, dtype=np.float64)
    g = np.full((len(candidate_ids), constraint_count), np.nan, dtype=np.float64)
    cv_mask = quantity_kinds == CV
    cv = None if not np.any(cv_mask) else np.full(len(candidate_ids), np.nan)

    def fill(kind: str, width: int, target: np.ndarray) -> None:
        mask = quantity_kinds == kind
        if not np.any(mask):
            return
        indices = quantity_indices[mask]
        if np.any(indices >= width):
            raise ValidationError(f"observation {kind} index exceeds its schema")
        keys = record_positions[mask] * max(width, 1) + indices
        if len(np.unique(keys)) != len(keys):
            raise ValidationError(f"observation {kind} has duplicate values")
        target[record_positions[mask], indices] = values[mask]

    fill(OBJECTIVE, objective_count, f)
    fill(CONSTRAINT, constraint_count, g)
    if cv is not None:
        indices = quantity_indices[cv_mask]
        if np.any(indices != 0) or len(indices) != len(candidate_ids):
            raise ValidationError("observation cv must have one value per candidate")
        if len(np.unique(record_positions[cv_mask])) != len(candidate_ids):
            raise ValidationError("observation cv has duplicate values")
        cv[record_positions[cv_mask]] = values[cv_mask]

    if "id" in owner.schema:
        owner_ids = np.asarray(owner.get_array("id"), dtype=np.int64)
        owner_order = np.argsort(owner_ids, kind="stable")
        sorted_owner_ids = owner_ids[owner_order]
        owner_positions = np.searchsorted(sorted_owner_ids, candidate_ids)
        valid = np.zeros(len(candidate_ids), dtype=bool)
        if len(sorted_owner_ids):
            valid = (owner_positions < len(sorted_owner_ids)) & (
                sorted_owner_ids[np.minimum(owner_positions, len(sorted_owner_ids) - 1)]
                == candidate_ids
            )
        if not np.all(valid):
            raise ValidationError("feedback candidate ID is not in offspring")
        rows = owner_order[owner_positions[valid]]
    else:
        rows = np.asarray(candidate_ids, dtype=np.intp)
        if np.any(rows < 0) or np.any(rows >= len(owner)):
            raise ValidationError("feedback candidate row is not in offspring")
    result = owner.extract(rows)
    updates: dict[str, np.ndarray] = {"f": f}
    if "g" in result.schema and g.shape[1] == result.get_array("g").shape[1]:
        updates["g"] = g
    if cv is not None and "cv" in result.schema:
        updates["cv"] = cv
    if len(result):
        result.update_rows(np.arange(len(result), dtype=np.intp), updates)
    return result
