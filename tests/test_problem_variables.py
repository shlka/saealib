"""Tests for Problem.variables, masks, and repair()."""

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.problem import Problem
from saealib.variables import CategoricalVariable, ContinuousVariable, IntegerVariable


def _base_kwargs(**overrides):
    defaults = dict(
        func=lambda x: np.sum(x),
        dim=3,
        n_obj=1,
        direction=np.array([-1.0]),
    )
    defaults.update(overrides)
    return defaults


def _mixed_problem():
    """3-dim problem: continuous, integer, categorical."""
    return Problem(
        **_base_kwargs(),
        variables=[
            ContinuousVariable(0.0, 1.0),
            IntegerVariable(1, 5),
            CategoricalVariable(["a", "b", "c"]),
        ],
    )


class TestBackwardCompat:
    def test_lb_ub_still_works(self):
        p = Problem(**_base_kwargs(), lb=[-1.0, -2.0, -3.0], ub=[1.0, 2.0, 3.0])
        np.testing.assert_array_equal(p.lb, [-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(p.ub, [1.0, 2.0, 3.0])

    def test_no_lb_ub_no_variables_raises(self):
        with pytest.raises(ValidationError):
            Problem(**_base_kwargs())

    def test_only_lb_raises(self):
        with pytest.raises(ValidationError):
            Problem(**_base_kwargs(), lb=[0.0, 0.0, 0.0])

    def test_variables_synthesized_as_continuous(self):
        p = Problem(**_base_kwargs(), lb=[0.0, 0.0, 0.0], ub=[1.0, 1.0, 1.0])
        assert all(isinstance(v, ContinuousVariable) for v in p.variables)
        assert len(p.variables) == 3

    def test_lb_ub_from_variables(self):
        p = _mixed_problem()
        np.testing.assert_array_equal(p.lb, [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(p.ub, [1.0, 5.0, 2.0])

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValidationError):
            Problem(
                **_base_kwargs(dim=2),
                variables=[ContinuousVariable(0.0, 1.0)] * 3,
            )


class TestMasks:
    def test_continuous_mask(self):
        p = _mixed_problem()
        np.testing.assert_array_equal(p.continuous_mask, [True, False, False])

    def test_integer_mask(self):
        p = _mixed_problem()
        np.testing.assert_array_equal(p.integer_mask, [False, True, False])

    def test_categorical_mask(self):
        p = _mixed_problem()
        np.testing.assert_array_equal(p.categorical_mask, [False, False, True])

    def test_masks_are_mutually_exclusive(self):
        p = _mixed_problem()
        overlap = p.continuous_mask & p.integer_mask
        assert not overlap.any()
        overlap = p.continuous_mask & p.categorical_mask
        assert not overlap.any()
        overlap = p.integer_mask & p.categorical_mask
        assert not overlap.any()

    def test_masks_cover_all_dims(self):
        p = _mixed_problem()
        union = p.continuous_mask | p.integer_mask | p.categorical_mask
        assert union.all()

    def test_all_continuous_masks(self):
        p = Problem(**_base_kwargs(), lb=[0.0] * 3, ub=[1.0] * 3)
        assert p.continuous_mask.all()
        assert not p.integer_mask.any()
        assert not p.categorical_mask.any()

    def test_n_categories(self):
        p = _mixed_problem()
        np.testing.assert_array_equal(p.n_categories, [0, 0, 3])

    def test_n_categories_all_continuous(self):
        p = Problem(**_base_kwargs(), lb=[0.0] * 3, ub=[1.0] * 3)
        np.testing.assert_array_equal(p.n_categories, [0, 0, 0])


class TestRepair:
    def test_repair_1d_continuous(self):
        p = Problem(**_base_kwargs(), lb=[0.0, 0.0, 0.0], ub=[1.0, 1.0, 1.0])
        x = np.array([-0.5, 0.5, 1.5])
        np.testing.assert_array_equal(p.repair(x), [0.0, 0.5, 1.0])

    def test_repair_1d_integer(self):
        p = _mixed_problem()
        x = np.array([0.5, 2.4, 1.0])
        result = p.repair(x)
        assert result[1] == 2.0  # rounded
        assert result[2] == 1.0  # categorical index, no change needed

    def test_repair_1d_categorical_clamps(self):
        p = _mixed_problem()
        x = np.array([0.5, 3.0, -0.5])
        result = p.repair(x)
        assert result[2] == 0.0  # clamped from -0.5

    def test_repair_2d(self):
        p = _mixed_problem()
        x = np.array(
            [
                [0.5, 2.4, 1.6],
                [-0.1, 5.8, 3.0],
            ]
        )
        result = p.repair(x)
        assert result.shape == (2, 3)
        # row 0: continuous clipped, integer rounded, categorical rounded
        assert result[0, 0] == 0.5
        assert result[0, 1] == 2.0
        assert result[0, 2] == 2.0
        # row 1: continuous clipped to lb, integer clamped, categorical clamped
        assert result[1, 0] == 0.0
        assert result[1, 1] == 5.0
        assert result[1, 2] == 2.0

    def test_repair_preserves_shape_1d(self):
        p = _mixed_problem()
        x = np.array([0.5, 3.0, 1.0])
        assert p.repair(x).shape == (3,)

    def test_repair_preserves_shape_2d(self):
        p = _mixed_problem()
        x = np.ones((5, 3))
        assert p.repair(x).shape == (5, 3)

    def test_repair_valid_input_unchanged(self):
        p = _mixed_problem()
        x = np.array([0.5, 3.0, 1.0])
        np.testing.assert_array_equal(p.repair(x), x)

    def test_repair_all_continuous_is_clipping(self):
        p = Problem(**_base_kwargs(), lb=[-1.0, -1.0, -1.0], ub=[1.0, 1.0, 1.0])
        x = np.array([-2.0, 0.0, 2.0])
        np.testing.assert_array_equal(p.repair(x), [-1.0, 0.0, 1.0])
