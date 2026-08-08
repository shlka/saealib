from __future__ import annotations

from saealib.core.state.keys import (
    ARCHIVES_MAIN,
    ARCHIVES_PARETO,
    EVALUATIONS_COUNT,
    EVALUATIONS_OWNERS,
    EVALUATIONS_PENDING,
    EVALUATIONS_PLAN,
    EVALUATIONS_PLAN_STATE,
    EVALUATIONS_PLAN_UPDATES,
    FEEDBACK_RESULT,
    PENDING_EVALUATIONS,
    POPULATIONS_MAIN,
    PROPOSALS_CURRENT,
    PROPOSALS_OFFSPRING,
    RUNTIME_ASYNC_FATAL,
    RUNTIME_CANDIDATE_ID_ALLOCATOR,
    RUNTIME_GENERATION,
    RUNTIME_REQUEST_ID_ALLOCATOR,
    RUNTIME_RNG,
    STATE_NAMESPACES,
    SURROGATES_DEFAULT,
    SURROGATES_PREDICTIONS,
    USER_DATA,
    StateKey,
)
from saealib.core.state.migration import (
    STATE_MIGRATORS,
    Migrator,
    StateMigrationRegistry,
    _evaluation_entry_v1_to_v2,
    _population_entry_v1_to_v2,
)
from saealib.core.state.patch import PopulationRowUpdate, StatePatch, StateUpdate
from saealib.core.state.store import LegacyAlgorithmStateView, StateStore, StateView

# Population entries moved to schema v2 when v3 checkpoints began carrying
# genomes through the space-provided GenomeCodec.  ``main`` is the one
# statically known named population, so its migrator is registered once while
# this module is imported.  Additional names are resolved from checkpoint
# entries by the v3 loader, not by constructing a StateKey.
if ("populations", "main", 1) not in STATE_MIGRATORS.registered():
    STATE_MIGRATORS.register("populations", "main", 1, _population_entry_v1_to_v2)
for _evaluation_name in ("plan", "pending"):
    if ("evaluations", _evaluation_name, 1) not in STATE_MIGRATORS.registered():
        STATE_MIGRATORS.register(
            "evaluations", _evaluation_name, 1, _evaluation_entry_v1_to_v2
        )

__all__ = [
    "ARCHIVES_MAIN",
    "ARCHIVES_PARETO",
    "EVALUATIONS_COUNT",
    "EVALUATIONS_OWNERS",
    "EVALUATIONS_PENDING",
    "EVALUATIONS_PLAN",
    "EVALUATIONS_PLAN_STATE",
    "EVALUATIONS_PLAN_UPDATES",
    "FEEDBACK_RESULT",
    "PENDING_EVALUATIONS",
    "POPULATIONS_MAIN",
    "PROPOSALS_CURRENT",
    "PROPOSALS_OFFSPRING",
    "RUNTIME_ASYNC_FATAL",
    "RUNTIME_CANDIDATE_ID_ALLOCATOR",
    "RUNTIME_GENERATION",
    "RUNTIME_REQUEST_ID_ALLOCATOR",
    "RUNTIME_RNG",
    "STATE_MIGRATORS",
    "STATE_NAMESPACES",
    "SURROGATES_DEFAULT",
    "SURROGATES_PREDICTIONS",
    "USER_DATA",
    "LegacyAlgorithmStateView",
    "Migrator",
    "PopulationRowUpdate",
    "StateKey",
    "StateMigrationRegistry",
    "StatePatch",
    "StateStore",
    "StateUpdate",
    "StateView",
]
