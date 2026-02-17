"""
Tests for the termination module.

Tests cover:
- Termination: init validation, is_terminated with single/multiple conditions
- max_fe: factory function for function evaluation termination
- max_gen: factory function for generation termination
- Custom callable conditions
"""

from types import SimpleNamespace

import pytest

from saealib.termination import Termination, max_fe, max_gen


def _make_ctx(fe: int = 0, gen: int = 0) -> SimpleNamespace:
    """Create a minimal context-like object with fe and gen attributes."""
    return SimpleNamespace(fe=fe, gen=gen)


# ===========================================================================
# Termination.__init__ Tests
# ===========================================================================
class TestTerminationInit:
    """Tests for Termination initialization and validation."""

    def test_single_condition(self) -> None:
        t = Termination(max_fe(100))
        assert len(t.conditions) == 1

    def test_multiple_conditions(self) -> None:
        t = Termination(max_fe(100), max_gen(50))
        assert len(t.conditions) == 2

    def test_lambda_condition(self) -> None:
        t = Termination(lambda ctx: ctx.fe >= 10)
        assert len(t.conditions) == 1

    def test_no_conditions_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            Termination()

    def test_non_callable_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be callable"):
            Termination(42)

    def test_mixed_callable_and_non_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="must be callable"):
            Termination(max_fe(100), "not_callable")

    def test_conditions_stored_as_tuple(self) -> None:
        t = Termination(max_fe(100))
        assert isinstance(t.conditions, tuple)


# ===========================================================================
# Termination.is_terminated Tests
# ===========================================================================
class TestIsTerminated:
    """Tests for Termination.is_terminated logic."""

    def test_not_terminated_below_threshold(self) -> None:
        t = Termination(max_fe(100))
        ctx = _make_ctx(fe=50)
        assert t.is_terminated(ctx) is False

    def test_terminated_at_threshold(self) -> None:
        t = Termination(max_fe(100))
        ctx = _make_ctx(fe=100)
        assert t.is_terminated(ctx) is True

    def test_terminated_above_threshold(self) -> None:
        t = Termination(max_fe(100))
        ctx = _make_ctx(fe=150)
        assert t.is_terminated(ctx) is True

    def test_multiple_conditions_or_logic_first_met(self) -> None:
        """Terminates when ANY condition is met (OR logic)."""
        t = Termination(max_fe(100), max_gen(50))
        ctx = _make_ctx(fe=100, gen=10)
        assert t.is_terminated(ctx) is True

    def test_multiple_conditions_or_logic_second_met(self) -> None:
        t = Termination(max_fe(100), max_gen(50))
        ctx = _make_ctx(fe=10, gen=50)
        assert t.is_terminated(ctx) is True

    def test_multiple_conditions_or_logic_both_met(self) -> None:
        t = Termination(max_fe(100), max_gen(50))
        ctx = _make_ctx(fe=100, gen=50)
        assert t.is_terminated(ctx) is True

    def test_multiple_conditions_none_met(self) -> None:
        t = Termination(max_fe(100), max_gen(50))
        ctx = _make_ctx(fe=10, gen=10)
        assert t.is_terminated(ctx) is False

    def test_custom_callable_condition(self) -> None:
        def custom(ctx):
            return ctx.fe > 0 and ctx.gen > 0

        t = Termination(custom)
        assert t.is_terminated(_make_ctx(fe=0, gen=0)) is False
        assert t.is_terminated(_make_ctx(fe=1, gen=1)) is True

    def test_lambda_condition(self) -> None:
        t = Termination(lambda ctx: ctx.fe >= 5)
        assert t.is_terminated(_make_ctx(fe=4)) is False
        assert t.is_terminated(_make_ctx(fe=5)) is True


# ===========================================================================
# max_fe Tests
# ===========================================================================
class TestMaxFe:
    """Tests for the max_fe factory function."""

    def test_returns_callable(self) -> None:
        cond = max_fe(100)
        assert callable(cond)

    def test_below_threshold(self) -> None:
        cond = max_fe(100)
        assert cond(_make_ctx(fe=99)) is False

    def test_at_threshold(self) -> None:
        cond = max_fe(100)
        assert cond(_make_ctx(fe=100)) is True

    def test_above_threshold(self) -> None:
        cond = max_fe(100)
        assert cond(_make_ctx(fe=200)) is True

    def test_zero_threshold(self) -> None:
        cond = max_fe(0)
        assert cond(_make_ctx(fe=0)) is True

    def test_qualname(self) -> None:
        cond = max_fe(500)
        assert cond.__qualname__ == "max_fe(500)"

    def test_doc(self) -> None:
        cond = max_fe(500)
        assert "500" in cond.__doc__


# ===========================================================================
# max_gen Tests
# ===========================================================================
class TestMaxGen:
    """Tests for the max_gen factory function."""

    def test_returns_callable(self) -> None:
        cond = max_gen(50)
        assert callable(cond)

    def test_below_threshold(self) -> None:
        cond = max_gen(50)
        assert cond(_make_ctx(gen=49)) is False

    def test_at_threshold(self) -> None:
        cond = max_gen(50)
        assert cond(_make_ctx(gen=50)) is True

    def test_above_threshold(self) -> None:
        cond = max_gen(50)
        assert cond(_make_ctx(gen=100)) is True

    def test_zero_threshold(self) -> None:
        cond = max_gen(0)
        assert cond(_make_ctx(gen=0)) is True

    def test_qualname(self) -> None:
        cond = max_gen(200)
        assert cond.__qualname__ == "max_gen(200)"

    def test_doc(self) -> None:
        cond = max_gen(200)
        assert "200" in cond.__doc__


# ===========================================================================
# Edge Case / Combination Tests
# ===========================================================================
class TestTerminationEdgeCases:
    """Tests for edge cases and advanced usage patterns."""

    def test_three_conditions(self) -> None:
        """Three conditions: terminates when any one is met."""
        t = Termination(
            max_fe(100),
            max_gen(50),
            lambda ctx: ctx.fe + ctx.gen > 80,
        )
        # Only third condition is met (fe=50 + gen=40 = 90 > 80)
        assert t.is_terminated(_make_ctx(fe=50, gen=40)) is True
        # No condition met
        assert t.is_terminated(_make_ctx(fe=10, gen=10)) is False

    def test_condition_with_class_method(self) -> None:
        """A class with __call__ can be used as a condition."""

        class StagnationCheck:
            def __call__(self, ctx):
                return ctx.gen >= 10

        t = Termination(StagnationCheck())
        assert t.is_terminated(_make_ctx(gen=9)) is False
        assert t.is_terminated(_make_ctx(gen=10)) is True

    def test_state_changes_over_iterations(self) -> None:
        """Simulate iteration: is_terminated returns False then True."""
        t = Termination(max_fe(5))
        ctx = _make_ctx(fe=0)
        for i in range(10):
            ctx.fe = i
            if i < 5:
                assert t.is_terminated(ctx) is False
            else:
                assert t.is_terminated(ctx) is True
