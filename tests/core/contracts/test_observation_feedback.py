from __future__ import annotations

import numpy as np
import pytest

import saealib.core.contracts.observation as observation_contracts
from saealib.core.contracts import CandidateIds as CoreCandidateIds
from saealib.core.contracts.feedback import (
    BY_CANDIDATE,
    BY_PROPOSAL,
    COMPLETE_BATCH,
    COMPLETION_MODES,
    DEFAULT_COMPLETION_MODE,
    DEFAULT_FEEDBACK_GROUPING,
    DEFAULT_MULTIPLICITY_MODE,
    DEFAULT_ORDERING_MODE,
    FEEDBACK_CHANNELS,
    FEEDBACK_GROUPINGS,
    IN_ORDER,
    MULTIPLICITY_MODES,
    ORDERING_MODES,
    OUT_OF_ORDER_ALLOWED,
    PARTIAL_ALLOWED,
    REPEATED_ALLOWED,
    SINGLE,
)
from saealib.core.contracts.observation import (
    BEHAVIOR,
    CANCELLED,
    CONSTRAINT,
    COST,
    CV,
    FAILED,
    FEATURE,
    HUMAN,
    IMPUTED,
    OBJECTIVE,
    OBSERVATION_QUANTITY_KINDS,
    OBSERVATION_SOURCES,
    OBSERVATION_STATUSES,
    OBSERVATION_SUBJECT_KINDS,
    OK,
    PORTABLE_CODECS,
    QUANTITY_KINDS,
    SIMULATOR,
    SURROGATE,
    TIMEOUT,
    TRUE,
    ColumnSpec,
    ObservationSubjectKind,
    ObservationSubjectPayload,
    PortableValue,
    StateCodec,
)
from saealib.core.contracts.relations import (
    CANDIDATE_ID,
    MANY,
    ONE,
    RELATION_KINDS,
    RelationKind,
    RelationPayload,
)
from saealib.exceptions import ValidationError
from saealib.population import CandidateIds as PopulationCandidateIds
from saealib.population.population import PopulationAttribute
from saealib.space.services import GenomeCodec
from saealib.space.space import ValidationResult


def test_observation_value_vocabularies_are_exact() -> None:
    assert OBSERVATION_SOURCES.names() == (
        TRUE,
        SURROGATE,
        HUMAN,
        SIMULATOR,
        IMPUTED,
    )
    assert OBSERVATION_STATUSES.names() == (OK, FAILED, CANCELLED, TIMEOUT)


def test_feedback_channels_are_separate_from_observation_sources() -> None:
    assert FEEDBACK_CHANNELS is not OBSERVATION_SOURCES
    assert FEEDBACK_CHANNELS.names() == (TRUE, SURROGATE, HUMAN, SIMULATOR)
    assert IMPUTED in OBSERVATION_SOURCES
    assert IMPUTED not in FEEDBACK_CHANNELS


def test_feedback_policy_vocabularies_are_exact() -> None:
    assert COMPLETION_MODES.names() == (COMPLETE_BATCH, PARTIAL_ALLOWED)
    assert ORDERING_MODES.names() == (IN_ORDER, OUT_OF_ORDER_ALLOWED)
    assert MULTIPLICITY_MODES.names() == (SINGLE, REPEATED_ALLOWED)
    assert FEEDBACK_GROUPINGS.names() == (BY_PROPOSAL, BY_CANDIDATE)


def test_feedback_defaults_are_the_strongest_values() -> None:
    assert DEFAULT_COMPLETION_MODE == COMPLETE_BATCH
    assert DEFAULT_ORDERING_MODE == IN_ORDER
    assert DEFAULT_MULTIPLICITY_MODE == SINGLE
    assert DEFAULT_FEEDBACK_GROUPING == BY_PROPOSAL


def test_quantity_kinds_are_exact() -> None:
    assert OBSERVATION_QUANTITY_KINDS is QUANTITY_KINDS
    assert QUANTITY_KINDS.names() == (
        OBJECTIVE,
        CONSTRAINT,
        CV,
        FEATURE,
        BEHAVIOR,
        COST,
    )


def test_descriptor_registries_are_exact() -> None:
    assert OBSERVATION_SUBJECT_KINDS.names() == ("candidate",)
    assert RELATION_KINDS.names() == ("parent_ids", "target_ids", "subproblem_ids")
    assert PORTABLE_CODECS.names() == ("float", "int", "bool", "str", "ndarray")
    assert not hasattr(observation_contracts, "VALUE_CODECS")


def test_candidate_ids_alias_is_shared_across_layers() -> None:
    assert CoreCandidateIds is PopulationCandidateIds


def test_payload_aliases_are_independent_named_numpy_aliases() -> None:
    assert ObservationSubjectPayload is np.ndarray
    assert RelationPayload is np.ndarray
    assert ObservationSubjectPayload is RelationPayload
    assert "ObservationSubjectPayload" in str(
        ObservationSubjectKind.__annotations__["candidate_ids"]
    )
    assert "RelationPayload" in str(RelationKind.__annotations__["reindex"])
    assert "CandidateIds" not in str(
        ObservationSubjectKind.__annotations__["candidate_ids"]
    )
    assert "CandidateIds" not in str(RelationKind.__annotations__["reindex"])


def test_observation_descriptor_aliases_keep_established_shapes() -> None:
    assert StateCodec is GenomeCodec
    assert ColumnSpec is PopulationAttribute
    subject = OBSERVATION_SUBJECT_KINDS.get("candidate")
    assert subject is not None
    assert len(subject.columns) == 1
    assert isinstance(subject.columns[0], PopulationAttribute)
    assert subject.columns[0].name == "id"
    assert subject.columns[0].dtype is np.int64


def test_candidate_subject_is_kind_agnostic_and_validates_ids() -> None:
    subject = OBSERVATION_SUBJECT_KINDS.get("candidate")
    assert subject is not None
    payload = np.array([4], dtype=np.int64)
    assert subject.candidate_ids(payload) is payload
    other = object()
    assert subject.candidate_ids(other) is other  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    assert subject.codec is GenomeCodec
    assert subject.validate(payload) == ValidationResult(valid_mask=(True,))
    assert not subject.validate(None).valid  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    assert not subject.validate(np.array([4, 9], dtype=np.int64)).valid
    assert not subject.validate(np.array([-1], dtype=np.int64)).valid


def test_candidate_relations_reindex_by_identity() -> None:
    indices = np.array([1, 0], dtype=np.int64)
    for name in ("parent_ids", "target_ids"):
        relation = RELATION_KINDS.get(name)
        assert relation is not None
        payload = np.array([10, 20], dtype=np.int64)
        assert relation.target == CANDIDATE_ID
        assert relation.arity == MANY
        assert relation.codec is GenomeCodec
        assert relation.reindex(payload, indices) is payload


def test_subject_descriptor_requires_all_seven_fields() -> None:
    with pytest.raises(TypeError):
        ObservationSubjectKind(  # ty: ignore[missing-argument]
            name="candidate",
            description="one candidate",
            arity="ONE",
            ordered=False,
            codec=object(),  # ty: ignore[invalid-argument-type]
            candidate_ids=lambda payload: payload,
            validate=lambda payload: payload,  # ty: ignore[invalid-argument-type]
        )


def test_relation_descriptor_requires_all_six_fields() -> None:
    with pytest.raises(TypeError):
        RelationKind(  # ty: ignore[missing-argument]
            name="parent_ids",
            description="parent candidate ids",
            target=CANDIDATE_ID,
            arity=ONE,
            codec=object(),  # ty: ignore[invalid-argument-type]
            columns=(),
        )


def test_descriptor_shape_validation_rejects_invalid_fixed_fields() -> None:
    with pytest.raises(ValidationError):
        ObservationSubjectKind(
            name="candidate",
            description="one candidate",
            arity="one",
            ordered=False,
            codec=object(),  # ty: ignore[invalid-argument-type]
            candidate_ids=lambda payload: payload,
            validate=lambda payload: payload,  # ty: ignore[invalid-argument-type]
            columns=(),
        )
    with pytest.raises(ValidationError):
        RelationKind(
            name="parent_ids",
            description="parent candidate ids",
            target="UNKNOWN",
            arity=ONE,
            codec=object(),  # ty: ignore[invalid-argument-type]
            columns=(),
            reindex=lambda payload, indices: payload,
        )
    with pytest.raises(ValidationError):
        RelationKind(
            name="parent_ids",
            description="parent candidate ids",
            target=CANDIDATE_ID,
            arity=MANY,
            codec=object(),  # ty: ignore[invalid-argument-type]
            columns=[],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            reindex=lambda payload, indices: payload,
        )


def test_portable_value_annotation_is_loose_but_documented_as_recursive() -> None:
    assert PortableValue is not object
    assert "Mapping" in repr(PortableValue)
    assert "Sequence" in repr(PortableValue)
