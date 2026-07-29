"""Tests for the internal MRO-aware batch-dispatch consistency helper.

Covers Issue #224's follow-up fix: ``type(obj).batch_method is not
Base.batch_method`` has a blind spot when a user subclasses a batch-capable
built-in operator and overrides only the scalar method(s) --
``batch_override_is_consistent`` closes that gap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pytest

from saealib._dispatch import batch_override_is_consistent
from saealib.operators.crossover import (
    CrossoverBLXAlpha,
    CrossoverCategorical,
    CrossoverIntegerSBX,
    CrossoverOnePoint,
    CrossoverSBX,
    CrossoverTwoPoint,
    CrossoverUniform,
)
from saealib.operators.mutation import (
    MutationCategorical,
    MutationGaussian,
    MutationIntegerUniform,
    MutationPolynomial,
    MutationUniform,
)

# ---------------------------------------------------------------------------
# Generic scenarios, using a minimal same-file (batch, scalar) pair rather
# than any real saealib class, so the test is independent of any future
# change to those classes.
# ---------------------------------------------------------------------------


class _Base(ABC):
    """Mirrors the real ``Crossover``/``Mutation`` shape:
    ``scalar`` is abstract (every concrete subclass MUST override it), while
    ``batch`` has a concrete default and is optional to override."""

    @abstractmethod
    def scalar(self):
        pass

    def batch(self):
        return "base-batch"


class _SameClass(_Base):
    """Overrides both scalar and batch together -- the common built-in case."""

    def scalar(self):
        return "same-scalar"

    def batch(self):
        return "same-batch"


class _ScalarOnly(_SameClass):
    """Overrides only the scalar method -- the CustomSBX-style bug scenario."""

    def scalar(self):
        return "scalar-only"


class _BatchOnly(_SameClass):
    """Overrides only the batch method, leaving the scalar inherited."""

    def batch(self):
        return "batch-only"


class _Neither(_Base):
    """Fresh subclass of the raw base: implements only the abstract scalar
    method (as required to be instantiable) and never touches batch --
    mirrors "a fresh Crossover subclass that implements only the abstract
    crossover"."""

    def scalar(self):
        return "neither-scalar"


class TestBatchOverrideIsConsistent:
    def test_true_for_matched_same_class_definitions(self):
        assert batch_override_is_consistent(_SameClass(), "batch", "scalar") is True

    def test_false_when_subclass_overrides_only_scalar(self):
        assert batch_override_is_consistent(_ScalarOnly(), "batch", "scalar") is False

    def test_true_when_subclass_overrides_both_together(self):
        obj = _ScalarOnly()

        class _Both(_ScalarOnly):
            def batch(self):
                return "both-batch"

        assert batch_override_is_consistent(_Both(), "batch", "scalar") is True
        # sanity: obj (scalar-only) is untouched by defining _Both.
        assert batch_override_is_consistent(obj, "batch", "scalar") is False

    def test_true_when_subclass_overrides_only_batch(self):
        assert batch_override_is_consistent(_BatchOnly(), "batch", "scalar") is True

    def test_false_for_abc_default_never_overridden(self):
        assert batch_override_is_consistent(_Neither(), "batch", "scalar") is False

    def test_multiple_scalar_methods_all_must_be_covered(self):
        class _Multi(_Base):
            def scalar(self):
                return "multi-scalar"

            def scalar2(self):
                return "multi-scalar2"

            def batch(self):
                return "multi-batch"

        assert (
            batch_override_is_consistent(_Multi(), "batch", "scalar", "scalar2") is True
        )

        class _MultiStale(_Multi):
            def scalar2(self):
                return "stale-scalar2"

        assert (
            batch_override_is_consistent(_MultiStale(), "batch", "scalar", "scalar2")
            is False
        )

    def test_missing_method_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            batch_override_is_consistent(_SameClass(), "batch", "nonexistent")


# ---------------------------------------------------------------------------
# Built-in Crossover/Mutation regression: none of commits 7-12's
# batch overrides should regress -- every built-in operator's plain,
# unsubclassed instance must still be found "consistent".
# ---------------------------------------------------------------------------


class TestBuiltinOperatorsRemainConsistent:
    @pytest.mark.parametrize(
        "op",
        [
            CrossoverSBX(prob=0.9, eta=2.0),
            CrossoverUniform(prob=0.8),
            CrossoverCategorical(prob=1.0),
            CrossoverIntegerSBX(prob=1.0, eta=2.0),
            CrossoverBLXAlpha(prob=0.9, alpha=0.4),
            CrossoverOnePoint(prob=0.9),
            CrossoverTwoPoint(prob=0.9),
        ],
    )
    def test_builtin_crossover_consistent(self, op):
        assert batch_override_is_consistent(op, "crossover_batch", "crossover") is True

    @pytest.mark.parametrize(
        "op",
        [
            MutationUniform(prob_var=0.5),
            MutationPolynomial(eta=20.0, prob_var=0.5),
            MutationGaussian(sigma=1.0, prob_var=0.5),
            MutationIntegerUniform(prob_var=0.5),
            MutationCategorical(prob_var=0.5),
        ],
    )
    def test_builtin_mutation_consistent(self, op):
        assert batch_override_is_consistent(op, "mutate_batch", "mutate") is True


# ---------------------------------------------------------------------------
# The concrete CustomSBX / custom-Mutation scenarios from the bug report.
# ---------------------------------------------------------------------------


class _CustomSBX(CrossoverSBX):
    """Overrides only the scalar crossover(); crossover_batch is inherited
    from CrossoverSBX and would silently ignore this override if dispatched
    to directly."""

    def crossover(self, parent, bounds=None, rng=np.random.default_rng()):
        return parent[:2].copy()


class _CustomMutation(MutationUniform):
    """Overrides only the scalar mutate(); mutate_batch is inherited from
    MutationUniform and would silently ignore this override if dispatched
    to directly."""

    def mutate(self, p, mutate_range, rng=np.random.default_rng()):
        return p.copy()


class TestCustomSubclassScenarios:
    def test_custom_sbx_scalar_only_override_is_inconsistent(self):
        op = _CustomSBX(prob=1.0, eta=15.0)
        assert batch_override_is_consistent(op, "crossover_batch", "crossover") is False

    def test_custom_mutation_scalar_only_override_is_inconsistent(self):
        op = _CustomMutation(prob=1.0, prob_var=0.5)
        assert batch_override_is_consistent(op, "mutate_batch", "mutate") is False
