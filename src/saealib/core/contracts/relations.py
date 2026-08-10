"""Relation-kind descriptor shape and the core candidate-ID registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np

from saealib.core.contracts.observation import ColumnSpec, StateCodec
from saealib.core.contracts.vocabulary import Vocabulary, VocabularyDescriptor
from saealib.exceptions import ValidationError
from saealib.identity import PopulationAttribute
from saealib.space.services import GenomeCodec

__all__ = [
    "CANDIDATE_ID",
    "GROUP",
    "MANY",
    "ONE",
    "OPAQUE",
    "RELATION_KINDS",
    "SUBPROBLEM",
    "RelationArity",
    "RelationKind",
    "RelationPayload",
    "RelationTarget",
]

RelationTarget: TypeAlias = str
RelationArity: TypeAlias = str
RelationPayload: TypeAlias = np.ndarray

CANDIDATE_ID: RelationTarget = "CANDIDATE_ID"
SUBPROBLEM: RelationTarget = "SUBPROBLEM"
GROUP: RelationTarget = "GROUP"
OPAQUE: RelationTarget = "OPAQUE"

ONE: RelationArity = "ONE"
MANY: RelationArity = "MANY"


@dataclass(frozen=True, kw_only=True)
class RelationKind(VocabularyDescriptor):
    """Operational descriptor for a registered proposal relation."""

    target: RelationTarget
    arity: RelationArity
    codec: StateCodec
    columns: tuple[ColumnSpec, ...]
    reindex: Callable[[RelationPayload, np.ndarray], RelationPayload]

    def __post_init__(self) -> None:
        """Validate relation metadata with the currently fixed leaf shapes."""
        super().__post_init__()
        if self.target not in {CANDIDATE_ID, SUBPROBLEM, GROUP, OPAQUE}:
            raise ValidationError(
                "Relation target must be 'CANDIDATE_ID', 'SUBPROBLEM', 'GROUP', "
                "or 'OPAQUE'"
            )
        if self.arity not in {ONE, MANY}:
            raise ValidationError("Relation arity must be 'ONE' or 'MANY'")
        if not isinstance(self.columns, tuple):
            raise ValidationError("Relation columns must be a tuple")
        if not callable(self.reindex):
            raise ValidationError("Relation reindex must be callable")


RELATION_KINDS: Vocabulary[RelationKind] = Vocabulary()


def _identity_reindex(payload: RelationPayload, indices: np.ndarray) -> RelationPayload:
    """Relations carry IDs, so row reindexing does not alter their payload."""
    return payload


for _name in ("parent_ids", "target_ids"):
    RELATION_KINDS.register(
        _name,
        RelationKind(
            name=_name,
            description=f"Candidate IDs for the {_name} relation.",
            target=CANDIDATE_ID,
            arity=MANY,
            codec=cast(StateCodec, GenomeCodec),
            columns=(PopulationAttribute(name=_name, dtype=np.int64, default=-1),),
            reindex=_identity_reindex,
        ),
    )

RELATION_KINDS.register(
    "subproblem_ids",
    RelationKind(
        name="subproblem_ids",
        description="Subproblem IDs associated with proposal rows.",
        target=SUBPROBLEM,
        arity=MANY,
        codec=cast(StateCodec, GenomeCodec),
        columns=(
            PopulationAttribute(name="subproblem_ids", dtype=np.int64, default=-1),
        ),
        reindex=_identity_reindex,
    ),
)
