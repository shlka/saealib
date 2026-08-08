"""Vocabularies and descriptors for observation records.

The registry contains the stable observation leaves needed by the core
contracts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np

from saealib.core.contracts.vocabulary import Vocabulary, VocabularyDescriptor
from saealib.exceptions import ValidationError
from saealib.population.population import PopulationAttribute
from saealib.space.services import GenomeCodec
from saealib.space.space import ValidationResult

__all__ = [
    "BEHAVIOR",
    "CANCELLED",
    "CONSTRAINT",
    "COST",
    "CV",
    "FAILED",
    "FEATURE",
    "HUMAN",
    "IMPUTED",
    "OBJECTIVE",
    "OBSERVATION_QUANTITY_KINDS",
    "OBSERVATION_SOURCES",
    "OBSERVATION_STATUSES",
    "OBSERVATION_SUBJECT_KINDS",
    "OBSERVATION_SUBJECT_MANY",
    "OBSERVATION_SUBJECT_ONE",
    "OK",
    "PORTABLE_CODECS",
    "QUANTITY_KINDS",
    "SIMULATOR",
    "SURROGATE",
    "TIMEOUT",
    "TRUE",
    "ColumnSpec",
    "ObservationSource",
    "ObservationStatus",
    "ObservationSubjectArity",
    "ObservationSubjectKind",
    "ObservationSubjectPayload",
    "ObservationValue",
    "PortableValue",
    "QuantityKind",
    "StateCodec",
]

ObservationSource: TypeAlias = str
ObservationStatus: TypeAlias = str
QuantityKind: TypeAlias = str
ObservationSubjectArity: TypeAlias = str
ObservationSubjectPayload: TypeAlias = np.ndarray
StateCodec: TypeAlias = GenomeCodec
ColumnSpec: TypeAlias = PopulationAttribute

OBSERVATION_SUBJECT_ONE: ObservationSubjectArity = "ONE"
OBSERVATION_SUBJECT_MANY: ObservationSubjectArity = "MANY"

TRUE: ObservationSource = "true"
SURROGATE: ObservationSource = "surrogate"
HUMAN: ObservationSource = "human"
SIMULATOR: ObservationSource = "simulator"
IMPUTED: ObservationSource = "imputed"
OK: ObservationStatus = "ok"
FAILED: ObservationStatus = "failed"
CANCELLED: ObservationStatus = "cancelled"
TIMEOUT: ObservationStatus = "timeout"

OBJECTIVE: QuantityKind = "objective"
CONSTRAINT: QuantityKind = "constraint"
CV: QuantityKind = "cv"
FEATURE: QuantityKind = "feature"
BEHAVIOR: QuantityKind = "behavior"
COST: QuantityKind = "cost"

# Portable values are recursively restricted to these shapes.  The annotation
# is deliberately readable rather than a fully recursive union; validators
# enforcing checkpoint portability must apply the rule to nested containers.
PortableValue: TypeAlias = (
    float | int | bool | str | np.ndarray | Mapping[str, object] | Sequence[object]
)
ObservationValue: TypeAlias = PortableValue


@dataclass(frozen=True, kw_only=True)
class ObservationSubjectKind(VocabularyDescriptor):
    """Operational descriptor for one kind of observation subject."""

    arity: ObservationSubjectArity
    ordered: bool
    codec: StateCodec
    candidate_ids: Callable[[ObservationSubjectPayload], ObservationSubjectPayload]
    validate: Callable[[ObservationSubjectPayload], ValidationResult]
    columns: tuple[ColumnSpec, ...]

    def __post_init__(self) -> None:
        """Validate the descriptor fields whose shape is known to core."""
        super().__post_init__()
        if self.arity not in {OBSERVATION_SUBJECT_ONE, OBSERVATION_SUBJECT_MANY}:
            raise ValidationError("Observation subject arity must be 'ONE' or 'MANY'")
        if not isinstance(self.ordered, bool):
            raise ValidationError("Observation subject ordered must be a bool")
        if not callable(self.candidate_ids):
            raise ValidationError("Observation subject candidate_ids must be callable")
        if not callable(self.validate):
            raise ValidationError("Observation subject validate must be callable")
        if not isinstance(self.columns, tuple):
            raise ValidationError("Observation subject columns must be a tuple")


OBSERVATION_SOURCES: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _name, _description in (
    (TRUE, "The value came from a true evaluation."),
    (SURROGATE, "The value came from a surrogate prediction."),
    (HUMAN, "The value came from a human assessment."),
    (SIMULATOR, "The value came from a simulator."),
    (IMPUTED, "The value was imputed because no usable observation was available."),
):
    OBSERVATION_SOURCES.register(
        _name, VocabularyDescriptor(name=_name, description=_description)
    )

OBSERVATION_STATUSES: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _name, _description in (
    (OK, "The observation completed successfully."),
    (FAILED, "The observation failed."),
    (CANCELLED, "The observation was cancelled."),
    (TIMEOUT, "The observation exceeded its time limit."),
):
    OBSERVATION_STATUSES.register(
        _name, VocabularyDescriptor(name=_name, description=_description)
    )

OBSERVATION_QUANTITY_KINDS: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _name, _description in (
    (OBJECTIVE, "An objective quantity."),
    (CONSTRAINT, "A constraint quantity."),
    (CV, "A constraint-violation quantity."),
    (FEATURE, "A feature quantity."),
    (BEHAVIOR, "A behavior-descriptor quantity."),
    (COST, "An evaluation cost quantity."),
):
    OBSERVATION_QUANTITY_KINDS.register(
        _name, VocabularyDescriptor(name=_name, description=_description)
    )

QUANTITY_KINDS = OBSERVATION_QUANTITY_KINDS


PORTABLE_CODECS: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _name, _description in (
    ("float", "A portable Python float."),
    ("int", "A portable Python int."),
    ("bool", "A portable Python bool."),
    ("str", "A portable Python str."),
    ("ndarray", "A portable NumPy array."),
):
    PORTABLE_CODECS.register(
        _name, VocabularyDescriptor(name=_name, description=_description)
    )

OBSERVATION_SUBJECT_KINDS: Vocabulary[ObservationSubjectKind] = Vocabulary()


def _identity_candidate_ids(
    payload: ObservationSubjectPayload,
) -> ObservationSubjectPayload:
    """Return the subject payload unchanged, regardless of subject kind."""
    return payload


def _validate_candidate_ids(payload: ObservationSubjectPayload) -> ValidationResult:
    """Validate the one non-negative ID carried by a candidate subject."""
    if not isinstance(payload, np.ndarray):
        return ValidationResult(errors=("candidate ids must be a NumPy array",))
    if payload.dtype != np.dtype(np.int64) or payload.ndim != 1:
        return ValidationResult(errors=("candidate ids must be a 1D int64 array",))
    if payload.size != 1:
        return ValidationResult(errors=("candidate subject must contain one ID",))
    if np.any(payload < 0):
        return ValidationResult(errors=("candidate IDs must be non-negative",))
    return ValidationResult(valid_mask=(True,))


OBSERVATION_SUBJECT_KINDS.register(
    "candidate",
    ObservationSubjectKind(
        name="candidate",
        description="A candidate identified by candidate IDs.",
        arity=OBSERVATION_SUBJECT_ONE,
        ordered=False,
        codec=cast(StateCodec, GenomeCodec),
        candidate_ids=_identity_candidate_ids,
        validate=_validate_candidate_ids,
        columns=(PopulationAttribute(name="id", dtype=np.int64, default=-1),),
    ),
)
