"""Tests for saealib.variables: Variable hierarchy."""

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.variables import (
    CategoricalVariable,
    ContinuousVariable,
    IntegerVariable,
    Variable,
)


class TestContinuousVariable:
    def test_is_variable(self):
        assert issubclass(ContinuousVariable, Variable)

    def test_bounds(self):
        v = ContinuousVariable(-1.0, 5.0)
        assert v.lb == -1.0
        assert v.ub == 5.0

    def test_lb_gt_ub_raises(self):
        with pytest.raises(ValidationError):
            ContinuousVariable(5.0, -1.0)

    def test_lb_eq_ub_ok(self):
        v = ContinuousVariable(3.0, 3.0)
        assert v.lb == v.ub == 3.0

    def test_repair_scalar(self):
        v = ContinuousVariable(0.0, 1.0)
        assert v.repair(np.float64(-0.5)) == 0.0
        assert v.repair(np.float64(0.5)) == 0.5
        assert v.repair(np.float64(1.5)) == 1.0

    def test_repair_array(self):
        v = ContinuousVariable(0.0, 1.0)
        x = np.array([-1.0, 0.5, 2.0])
        np.testing.assert_array_equal(v.repair(x), [0.0, 0.5, 1.0])

    def test_repair_preserves_interior(self):
        v = ContinuousVariable(-5.0, 5.0)
        x = np.array([-3.0, 0.0, 3.0])
        np.testing.assert_array_equal(v.repair(x), x)


class TestIntegerVariable:
    def test_is_variable(self):
        assert issubclass(IntegerVariable, Variable)

    def test_bounds_as_float(self):
        v = IntegerVariable(1, 10)
        assert v.lb == 1.0
        assert v.ub == 10.0

    def test_lb_gt_ub_raises(self):
        with pytest.raises(ValidationError):
            IntegerVariable(10, 1)

    def test_repair_rounds(self):
        v = IntegerVariable(0, 5)
        result = v.repair(np.array([0.4, 1.6, 2.5]))
        np.testing.assert_array_equal(result, [0.0, 2.0, 2.0])

    def test_repair_clamps(self):
        v = IntegerVariable(0, 5)
        np.testing.assert_array_equal(v.repair(np.array([-1.0, 6.0])), [0.0, 5.0])

    def test_repair_scalar(self):
        v = IntegerVariable(0, 10)
        assert v.repair(np.float64(3.7)) == 4.0

    def test_repair_integer_values_unchanged(self):
        v = IntegerVariable(0, 5)
        x = np.array([0.0, 2.0, 5.0])
        np.testing.assert_array_equal(v.repair(x), x)


class TestCategoricalVariable:
    def test_is_variable(self):
        assert issubclass(CategoricalVariable, Variable)

    def test_empty_raises(self):
        with pytest.raises(ValidationError):
            CategoricalVariable([])

    def test_bounds(self):
        v = CategoricalVariable(["a", "b", "c"])
        assert v.lb == 0.0
        assert v.ub == 2.0

    def test_n_categories(self):
        v = CategoricalVariable(["x", "y"])
        assert v.n_categories == 2

    def test_categories_copy(self):
        cats = ["a", "b"]
        v = CategoricalVariable(cats)
        cats.append("c")
        assert v.n_categories == 2  # original unchanged

    def test_repair_rounds_and_clamps(self):
        v = CategoricalVariable(["a", "b", "c"])
        x = np.array([-0.5, 0.4, 1.6, 2.5, 3.0])
        np.testing.assert_array_equal(v.repair(x), [0.0, 0.0, 2.0, 2.0, 2.0])

    def test_repair_scalar(self):
        v = CategoricalVariable(["a", "b", "c"])
        assert v.repair(np.float64(1.6)) == 2.0

    def test_encode(self):
        v = CategoricalVariable(["red", "green", "blue"])
        assert v.encode("red") == 0
        assert v.encode("green") == 1
        assert v.encode("blue") == 2

    def test_encode_unknown_raises(self):
        v = CategoricalVariable(["a", "b"])
        with pytest.raises(ValidationError):
            v.encode("c")

    def test_decode(self):
        v = CategoricalVariable(["red", "green", "blue"])
        assert v.decode(0) == "red"
        assert v.decode(1) == "green"
        assert v.decode(2) == "blue"

    def test_decode_float_rounds(self):
        v = CategoricalVariable(["a", "b", "c"])
        assert v.decode(1.4) == "b"
        assert v.decode(1.6) == "c"

    def test_decode_out_of_range_raises(self):
        v = CategoricalVariable(["a", "b"])
        with pytest.raises(ValidationError):
            v.decode(2)
        with pytest.raises(ValidationError):
            v.decode(-1)

    def test_encode_decode_roundtrip(self):
        cats = ["red", "green", "blue"]
        v = CategoricalVariable(cats)
        for cat in cats:
            assert v.decode(v.encode(cat)) == cat

    def test_single_category(self):
        v = CategoricalVariable(["only"])
        assert v.n_categories == 1
        assert v.lb == 0.0
        assert v.ub == 0.0
        assert v.repair(np.array([0.0])) == 0.0


class TestTopLevelExport:
    def test_importable_from_saealib(self):
        import saealib

        assert hasattr(saealib, "Variable")
        assert hasattr(saealib, "ContinuousVariable")
        assert hasattr(saealib, "IntegerVariable")
        assert hasattr(saealib, "CategoricalVariable")

    def test_in_all(self):
        import saealib

        names = (
            "Variable",
            "ContinuousVariable",
            "IntegerVariable",
            "CategoricalVariable",
        )
        for name in names:
            assert name in saealib.__all__, f"{name} not in __all__"
