import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.utils import (
    gd,
    gd_plus,
    hypervolume,
    igd,
    igd_plus,
    spacing,
    spread,
)


def test_distance_indicators_have_analytic_values() -> None:
    solutions = np.array([[0.0, 1.0], [1.0, 0.0]])
    reference = np.array([[0.0, 0.0], [1.0, 1.0]])

    np.testing.assert_allclose(gd(solutions, reference), 1.0)
    np.testing.assert_allclose(igd(solutions, reference), 1.0)
    np.testing.assert_allclose(gd_plus(solutions, reference), 0.0)
    np.testing.assert_allclose(igd_plus(solutions, reference), 0.5)


def test_plus_indicators_use_the_expected_direction() -> None:
    worse_solution = np.array([[1.0, 0.0]])
    better_solution = np.array([[0.0, 0.0]])
    reference = np.array([[0.0, 0.0]])
    worse_reference = np.array([[1.0, 0.0]])

    np.testing.assert_allclose(igd_plus(worse_solution, reference), 1.0)
    np.testing.assert_allclose(igd_plus(better_solution, worse_reference), 0.0)
    np.testing.assert_allclose(gd_plus(worse_solution, reference), 1.0)
    np.testing.assert_allclose(gd_plus(better_solution, worse_reference), 0.0)


def test_spacing_has_analytic_values() -> None:
    np.testing.assert_allclose(
        spacing(np.array([[0.0], [1.0], [3.0]])), np.sqrt(1.0 / 3.0)
    )
    np.testing.assert_allclose(
        spacing(np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 1.0]])), 0.0
    )


def test_spacing_squared_returns_schott_variance() -> None:
    f = np.array([[0.0], [1.0], [3.0]])

    np.testing.assert_allclose(spacing(f, squared=True), 1.0 / 3.0)
    np.testing.assert_allclose(spacing(f, squared=True), spacing(f) ** 2)
    assert np.isnan(spacing(np.array([[1.0]]), squared=True))


def test_spread_perfect_reference_sample_is_zero() -> None:
    reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

    np.testing.assert_allclose(spread(reference, reference), 0.0)


def test_spread_matches_generalized_definition() -> None:
    reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    nonuniform = np.array([[0.0, 1.0], [0.1, 0.9], [1.0, 0.0]])
    np.testing.assert_allclose(spread(nonuniform, reference), 4.0 / 3.0)

    missing_extreme = np.array([[0.5, 0.5], [1.0, 0.0]])
    np.testing.assert_allclose(spread(missing_extreme, reference), 7.0 / 6.0)


def test_spread_uses_one_extreme_per_objective() -> None:
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [3.0, 1.0, 2.0],
            [2.0, 3.0, 1.0],
        ]
    )
    obtained = reference[1:]

    np.testing.assert_allclose(spread(obtained, reference), 3.0 / 2.0)


def test_empty_solution_sets_return_nan() -> None:
    empty = np.empty((0, 2))
    reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

    for indicator in (
        gd,
        igd,
        gd_plus,
        igd_plus,
        spread,
    ):
        assert np.isnan(indicator(empty, reference))
    assert np.isnan(spacing(empty))


def test_empty_reference_front_raises_validation_error() -> None:
    empty_reference = np.empty((0, 2))
    point = np.array([[0.5, 0.5]])
    for indicator in (gd, igd, gd_plus, igd_plus, spread):
        with pytest.raises(ValidationError):
            indicator(point, empty_reference)


def test_spacing_singleton_returns_nan() -> None:
    assert np.isnan(spacing(np.array([[0.5, 0.5]])))


def test_hypervolume_empty_solution_set_returns_zero() -> None:
    assert hypervolume(np.empty((0, 2)), np.array([1.0, 1.0])) == 0.0


def test_spread_singleton_keeps_generalized_value() -> None:
    reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    point = np.array([[0.5, 0.5]])

    np.testing.assert_allclose(spread(point, reference), 5.0 / 6.0)


def test_spread_duplicate_only_front_matches_singleton_value() -> None:
    reference = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    duplicate_only = np.array([[0.5, 0.5], [0.5, 0.5]])

    np.testing.assert_allclose(spread(duplicate_only, reference), 5.0 / 6.0)


def test_duplicate_points_are_supported() -> None:
    duplicate = np.array([[0.0, 1.0], [0.0, 1.0]])
    reference = np.array([[0.0, 1.0]])

    assert spacing(duplicate) == 0.0
    assert spread(duplicate, reference) == 0.0
    assert gd(duplicate, reference) == 0.0
    assert igd(duplicate, reference) == 0.0


def test_dimension_mismatch_is_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        gd(np.zeros((2, 2)), np.zeros((2, 3)))
