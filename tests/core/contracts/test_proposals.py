"""Proposal contract tests."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from saealib.core.contracts.observations import QuantityRef
from saealib.core.contracts.proposals import (
    FeedbackRequirement,
    ProposalBatch,
    ProposalRelations,
    QuantityRequirement,
)
from saealib.core.contracts.relations import (
    ONE,
    OPAQUE,
    RELATION_KINDS,
    RelationKind,
)
from saealib.exceptions import ValidationError
from saealib.identity import IDAllocator
from saealib.population.population import Population, PopulationAttribute
from saealib.space.services import GenomeCodec


def _population(size: int) -> Population:
    population = Population([PopulationAttribute("id", np.int64, default=-1)])
    for _ in range(size):
        population.append()
    return population


def test_requirements_keep_sources_and_fidelity_per_quantity() -> None:
    """Each quantity keeps its own source set and fidelity floor.

    Mutation note: moving these fields to ``FeedbackRequirement`` makes the two
    independent values collapse and fails this assertion.
    """
    requirement = FeedbackRequirement(
        quantities=(
            QuantityRequirement(
                quantity=QuantityRef(kind="objective", index=0),
                sources=frozenset({"true", "surrogate"}),
                fidelity=0,
            ),
            QuantityRequirement(
                quantity=QuantityRef(kind="constraint", index=0), fidelity=2
            ),
        )
    )
    assert requirement.quantities[0].sources == frozenset({"true", "surrogate"})
    assert requirement.quantities[0].fidelity == 0
    assert requirement.quantities[1].sources == frozenset({"true"})
    assert requirement.quantities[1].fidelity == 2


def test_relations_reject_unknown_kinds_and_keep_empty_row_count() -> None:
    """Only registered kinds are accepted, including the empty relation case.

    Mutation note: accepting arbitrary relation names or omitting ``row_count``
    on an empty mapping breaks the ProposalBatch alignment invariant.
    """
    with pytest.raises(ValidationError, match="unknown relation kind"):
        ProposalRelations({"unregistered": np.array([1], dtype=np.int64)})
    assert ProposalRelations({}).row_count == 0


def test_relations_follow_declared_column_dtype_and_are_read_only() -> None:
    """Core relation columns stay native and cannot be mutated through a view."""
    relations = ProposalRelations(
        {"parent_ids": np.arange(6, dtype=np.int64).reshape(3, 2)}
    )

    column = cast(np.ndarray, relations["parent_ids"])
    assert column.dtype == np.dtype(np.int64)
    assert column.shape == (3, 2)
    assert not column.flags.writeable
    with pytest.raises(ValueError):
        column[0, 0] = 99


def test_variable_many_relation_outer_store_is_read_only() -> None:
    """Ragged MANY values use a read-only object outer column."""
    relations = ProposalRelations({"parent_ids": [[1, 2], [3]]})
    column = cast(np.ndarray, relations["parent_ids"])

    assert column.dtype == np.dtype(object)
    assert not column.flags.writeable


def test_take_calls_reindex_and_preserves_candidate_relation_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Taking rows invokes the descriptor and renumbers row-index payloads.

    Mutation note: removing the reindex call, or taking candidates and
    relations with different indices, leaves this callback/alignment check
    failing.
    """
    calls: list[np.ndarray] = []

    def reindex(payload: np.ndarray, indices: np.ndarray) -> np.ndarray:
        calls.append(indices.copy())
        mapping = {int(old): new for new, old in enumerate(indices)}
        return np.asarray([mapping[int(value)] for value in payload], dtype=np.int64)

    kind = RelationKind(
        name="test:row_indexes",
        description="test relation",
        target=OPAQUE,
        arity=ONE,
        codec=cast(GenomeCodec, GenomeCodec),
        columns=(PopulationAttribute(name="row", dtype=np.int64, default=-1),),
        reindex=reindex,
    )
    monkeypatch.setitem(RELATION_KINDS._entries, kind.name, kind)
    relations = ProposalRelations({kind.name: np.array([0, 1, 2], dtype=np.int64)})
    batch = ProposalBatch(
        proposal_id=4,
        candidates=_population(3),
        relations=relations,
        requirements=FeedbackRequirement(quantities=()),
    )

    selected = batch.take(np.array([2, 0], dtype=np.intp))
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [2, 0])
    np.testing.assert_array_equal(selected.relations[kind.name], [0, 1])
    assert len(selected.candidates) == selected.relations.row_count == 2


def test_allocator_path_assigns_proposal_id_and_metadata_is_frozen() -> None:
    """Proposal IDs come from the supplied allocator and metadata is copied.

    Mutation note: adding an independent proposal counter or retaining the
    caller's mutable metadata makes one of these assertions fail.
    """
    metadata = {"tag": "kept"}
    batch = ProposalBatch.from_allocator(
        IDAllocator(start=40),
        candidates=_population(0),
        relations=ProposalRelations({}),
        requirements=FeedbackRequirement(quantities=()),
        metadata=metadata,
    )
    metadata["tag"] = "mutated"
    assert batch.proposal_id == 40
    assert batch.metadata["tag"] == "kept"
    with pytest.raises(TypeError):
        cast(dict[str, object], batch.metadata)["new"] = "value"
