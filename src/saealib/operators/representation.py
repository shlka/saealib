"""Operators for permutation and variable-length sequence representations."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import replace
from typing import cast

import numpy as np

from saealib.core.contracts import ComponentContract, Fixed
from saealib.exceptions import ValidationError
from saealib.operators.crossover import Crossover
from saealib.operators.mutation import Mutation
from saealib.population.genome import (
    GenomeBatch,
    PermutationBatch,
    VariableLengthBatch,
)
from saealib.registry import register

__all__ = ["OrderCrossover", "SequenceMutation", "SwapMutation"]


def _fixed_representation(
    base: ComponentContract, value: str, *, requires_bounds: bool = True
) -> ComponentContract:
    """Bind both sides of a representation operator to one representation kind."""
    role = next(iter(base.ports.values()))
    inputs = tuple(
        replace(
            port,
            required_services=() if not requires_bounds else port.required_services,
            data=replace(
                port.data,
                bindings={**port.data.bindings, "representation": Fixed(value=value)},
            ),
        )
        for port in role.inputs
    )
    outputs = tuple(
        replace(
            port,
            data=replace(
                port.data,
                bindings={**port.data.bindings, "representation": Fixed(value=value)},
            ),
        )
        for port in role.outputs
    )
    return replace(
        base,
        ports={next(iter(base.ports)): replace(role, inputs=inputs, outputs=outputs)},
    )


@register()
class OrderCrossover(Crossover):
    """Order crossover (OX) for fixed-length permutations.

    ``crossover_batch`` accepts the usual NumPy shape ``(n, 2, length)`` and
    returns that shape.  It also accepts a ``PermutationBatch`` containing
    parent rows in consecutive pairs and returns a ``PermutationBatch``.
    """

    def __init__(self, prob: float = 1.0) -> None:
        self.prob = prob

    def contract(self) -> ComponentContract:
        """Return the contract bound to permutation representation."""
        return _fixed_representation(
            super().contract(), "permutation", requires_bounds=False
        )

    def crossover_batch(self, parents_batch, bounds=None, rng=np.random.default_rng()):
        """Apply order crossover to a batch of parent permutations."""
        as_batch = isinstance(parents_batch, PermutationBatch)
        if as_batch:
            parents = parents_batch.array
            if len(parents) % 2:
                raise ValidationError(
                    "OrderCrossover requires an even number of parent rows"
                )
            parents = parents.reshape(-1, 2, parents.shape[1])
        else:
            parents = np.asarray(parents_batch)
            if parents.ndim != 3 or parents.shape[1] != 2:
                raise ValidationError(
                    "OrderCrossover parents must have shape (n, 2, length)"
                )
        n_pairs, _, length = parents.shape
        children = np.empty((n_pairs, 2, length), dtype=np.int64)
        for pair in range(n_pairs):
            left, right = sorted(rng.choice(length + 1, size=2, replace=False))
            p1, p2 = parents[pair]
            c1, c2 = p2.copy(), p1.copy()
            c1[left:right] = p1[left:right]
            c2[left:right] = p2[left:right]
            used1 = set(c1[left:right].tolist())
            used2 = set(c2[left:right].tolist())
            positions = list(range(right, length)) + list(range(0, left))
            values1 = [value for value in p2 if value not in used1]
            values2 = [value for value in p1 if value not in used2]
            for position, value1, value2 in zip(positions, values1, values2):
                c1[position] = value1
                c2[position] = value2
            children[pair] = (c1, c2)
        if as_batch:
            return PermutationBatch(children.reshape(-1, length), length=length)
        return children

    def apply(
        self, parents: GenomeBatch, rng=np.random.default_rng()
    ) -> PermutationBatch:
        """Apply OX to consecutive parent pairs and return a permutation batch."""
        return self.crossover_batch(parents, rng=rng)


@register()
class SwapMutation(Mutation):
    """Mutation that swaps two positions in selected permutation rows."""

    def __init__(self, prob: float = 1.0) -> None:
        self.prob = prob
        self.prob_var = None

    def contract(self) -> ComponentContract:
        """Return the contract bound to permutation representation."""
        return _fixed_representation(
            super().contract(), "permutation", requires_bounds=False
        )

    def mutate_batch(
        self, candidates_batch, mutate_range=None, rng=np.random.default_rng()
    ):
        """Mutate a batch of permutations by swapping positions."""
        as_batch = isinstance(candidates_batch, PermutationBatch)
        values = (
            candidates_batch.array.copy()
            if as_batch
            else np.asarray(candidates_batch, dtype=np.int64).copy()
        )
        if values.ndim != 2:
            raise ValidationError("SwapMutation candidates must have shape (n, length)")
        for row in range(len(values)):
            if rng.random() < self.prob and values.shape[1] >= 2:
                first, second = rng.choice(values.shape[1], size=2, replace=False)
                values[row, first], values[row, second] = (
                    values[row, second],
                    values[row, first],
                )
        return (
            PermutationBatch(values, length=candidates_batch.length)
            if as_batch
            else values
        )

    def apply(
        self, candidates: GenomeBatch, rng=np.random.default_rng()
    ) -> PermutationBatch:
        """Mutate candidate genomes and return permutation sequences."""
        return self.mutate_batch(candidates, rng=rng)


@register()
class SequenceMutation(Mutation):
    """Point, insertion, deletion, or swap mutation for finite sequences.

    ``alphabet`` enables point and insertion mutations.  Without it, the
    operator still provides a useful order-preserving swap mutation.
    """

    def __init__(
        self,
        prob: float = 1.0,
        *,
        alphabet: Sequence[Hashable] | None = None,
        min_length: int = 0,
        max_length: int | None = None,
    ) -> None:
        self.prob = prob
        self.prob_var = None
        self.alphabet = tuple(alphabet) if alphabet is not None else None
        self.min_length = min_length
        self.max_length = max_length

    def contract(self) -> ComponentContract:
        """Return the contract bound to sequence representation."""
        return _fixed_representation(
            super().contract(), "sequence", requires_bounds=False
        )

    def mutate_batch(
        self, candidates_batch, mutate_range=None, rng=np.random.default_rng()
    ):
        """Apply a random sequence mutation to each selected row."""
        as_batch = isinstance(candidates_batch, VariableLengthBatch)
        if not as_batch:
            raise ValidationError("SequenceMutation requires VariableLengthBatch")
        rows = [list(row) for row in candidates_batch.sequences]
        max_length = self.max_length
        alphabet = cast(tuple[Hashable, ...], self.alphabet)
        for row in rows:
            if rng.random() >= self.prob:
                continue
            can_delete = len(row) > self.min_length
            can_insert = alphabet is not None and (
                max_length is None or len(row) < max_length
            )
            operations = ["swap"] if len(row) >= 2 else []
            if alphabet is not None and row:
                operations.append("replace")
            if can_insert:
                operations.append("insert")
            if can_delete:
                operations.append("delete")
            if not operations:
                continue
            operation = operations[int(rng.integers(len(operations)))]
            if operation == "swap":
                i, j = rng.choice(len(row), size=2, replace=False)
                row[i], row[j] = row[j], row[i]
            elif operation == "replace":
                row[int(rng.integers(len(row)))] = alphabet[
                    int(rng.integers(len(alphabet)))
                ]
            elif operation == "insert":
                row.insert(
                    int(rng.integers(len(row) + 1)),
                    alphabet[int(rng.integers(len(alphabet)))],
                )
            else:
                del row[int(rng.integers(len(row)))]
        return VariableLengthBatch(rows)

    def apply(
        self, candidates: GenomeBatch, rng=np.random.default_rng()
    ) -> VariableLengthBatch:
        """Mutate candidate genomes and return variable-length sequences."""
        return self.mutate_batch(candidates, rng=rng)
