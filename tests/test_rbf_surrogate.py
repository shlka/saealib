from functools import partial

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.surrogate.rbf import (
    RBFSurrogate,
    cubic_kernel,
    gaussian_kernel,
    linear_kernel,
    matern_kernel,
    multiquadric_kernel,
    thin_plate_spline_kernel,
)

TRAIN_X = np.array([[-1.0], [0.0], [1.0]])
TRAIN_Y = np.array([1.0, 2.0, -0.5])


def test_default_degree_and_solver_interpolate_training_points():
    surrogate = RBFSurrogate(kernel=gaussian_kernel, dim=1)
    surrogate.fit(TRAIN_X, TRAIN_Y)

    np.testing.assert_allclose(
        surrogate.predict(TRAIN_X).value[:, 0],
        TRAIN_Y,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("polynomial_degree", [0, 1])
def test_polynomial_terms_interpolate_training_points(polynomial_degree):
    surrogate = RBFSurrogate(
        kernel=gaussian_kernel,
        dim=1,
        polynomial_degree=polynomial_degree,
    )
    surrogate.fit(TRAIN_X, TRAIN_Y)

    np.testing.assert_allclose(
        surrogate.predict(TRAIN_X).value[:, 0],
        TRAIN_Y,
        rtol=1e-10,
        atol=1e-10,
    )


def test_degree_one_reproduces_linear_extrapolation_but_default_does_not():
    coefficient = 2.0
    intercept = 3.0
    train_y = coefficient * TRAIN_X[:, 0] + intercept
    outside_x = np.array([[3.0]])
    expected = coefficient * outside_x[:, 0] + intercept

    polynomial_surrogate = RBFSurrogate(
        kernel=gaussian_kernel,
        dim=1,
        polynomial_degree=1,
    )
    polynomial_surrogate.fit(TRAIN_X, train_y)
    polynomial_prediction = polynomial_surrogate.predict(outside_x).value[:, 0]

    default_surrogate = RBFSurrogate(kernel=gaussian_kernel, dim=1)
    default_surrogate.fit(TRAIN_X, train_y)
    default_prediction = default_surrogate.predict(outside_x).value[:, 0]

    np.testing.assert_allclose(
        polynomial_prediction,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.max(np.abs(default_prediction - expected)) > 1e-3


def test_thin_plate_spline_kernel_with_degree_one_interpolates_training_points():
    surrogate = RBFSurrogate(
        kernel=thin_plate_spline_kernel,
        dim=1,
        polynomial_degree=1,
    )
    surrogate.fit(TRAIN_X, TRAIN_Y)

    np.testing.assert_allclose(
        surrogate.predict(TRAIN_X).value[:, 0],
        TRAIN_Y,
        rtol=1e-10,
        atol=1e-10,
    )


def test_tikhonov_residual_increases_with_alpha_and_approaches_solve():
    solve_surrogate = RBFSurrogate(kernel=gaussian_kernel, dim=1)
    solve_surrogate.fit(TRAIN_X, TRAIN_Y)
    solve_prediction = solve_surrogate.predict(TRAIN_X).value[:, 0]

    predictions = []
    residuals = []
    for alpha in (1e-6, 1e-3, 1.0):
        surrogate = RBFSurrogate(
            kernel=gaussian_kernel,
            dim=1,
            solver="tikhonov",
            alpha=alpha,
        )
        surrogate.fit(TRAIN_X, TRAIN_Y)
        prediction = surrogate.predict(TRAIN_X).value[:, 0]
        predictions.append(prediction)
        residuals.append(np.linalg.norm(prediction - TRAIN_Y))

    assert residuals[0] > 1e-10
    assert residuals[0] < residuals[1] < residuals[2]
    np.testing.assert_allclose(
        predictions[0],
        solve_prediction,
        rtol=1e-5,
        atol=1e-5,
    )


def test_lstsq_returns_finite_values_for_rank_deficient_training_data():
    train_x = np.array([[-1.0], [0.0], [0.0], [1.0]])
    train_y = np.array([1.0, 2.0, 2.0, 3.0])
    test_x = np.array([[-1.0], [0.0], [1.0], [2.0]])
    surrogate = RBFSurrogate(
        kernel=gaussian_kernel,
        dim=1,
        polynomial_degree=1,
        solver="lstsq",
    )

    surrogate.fit(train_x, train_y)

    assert np.isfinite(surrogate.predict(test_x).value).all()


@pytest.mark.parametrize("polynomial_degree", [-2, 2, 0.5])
def test_invalid_polynomial_degree_raises_validation_error(polynomial_degree):
    with pytest.raises(ValidationError):
        RBFSurrogate(
            kernel=gaussian_kernel,
            dim=1,
            polynomial_degree=polynomial_degree,
        )


@pytest.mark.parametrize("solver", ["invalid", "qr"])
def test_invalid_solver_raises_validation_error(solver):
    with pytest.raises(ValidationError):
        RBFSurrogate(kernel=gaussian_kernel, dim=1, solver=solver)


@pytest.mark.parametrize("solver", ["solve", "lstsq", "tikhonov"])
@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_nonpositive_alpha_raises_validation_error(solver, alpha):
    with pytest.raises(ValidationError):
        RBFSurrogate(kernel=gaussian_kernel, dim=1, solver=solver, alpha=alpha)


@pytest.mark.parametrize(
    ("kernel", "polynomial_degree"),
    [
        (linear_kernel, 0),
        (cubic_kernel, 1),
        (multiquadric_kernel, 0),
    ],
)
def test_additional_cpd_kernels_interpolate_training_points(kernel, polynomial_degree):
    surrogate = RBFSurrogate(
        kernel=kernel,
        dim=1,
        polynomial_degree=polynomial_degree,
    )
    surrogate.fit(TRAIN_X, TRAIN_Y)

    np.testing.assert_allclose(
        surrogate.predict(TRAIN_X).value[:, 0],
        TRAIN_Y,
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_matern_kernels_interpolate_training_points_without_polynomial_term(nu):
    surrogate = RBFSurrogate(kernel=partial(matern_kernel, nu=nu), dim=1)
    surrogate.fit(TRAIN_X, TRAIN_Y)

    np.testing.assert_allclose(
        surrogate.predict(TRAIN_X).value[:, 0],
        TRAIN_Y,
        rtol=1e-10,
        atol=1e-10,
    )


def test_matern_kernel_rejects_invalid_nu_at_fit_time():
    surrogate = RBFSurrogate(kernel=partial(matern_kernel, nu=1.0), dim=1)

    with pytest.raises(ValidationError, match=r"nu must be 0\.5, 1\.5, or 2\.5"):
        surrogate.fit(TRAIN_X, TRAIN_Y)


def test_multiquadric_without_required_constant_term_has_worse_interpolation():
    train_x = np.arange(40, dtype=float).reshape(-1, 1)
    train_y = train_x[:, 0] ** 3

    valid_surrogate = RBFSurrogate(
        kernel=multiquadric_kernel,
        dim=1,
        polynomial_degree=0,
    )
    invalid_surrogate = RBFSurrogate(
        kernel=multiquadric_kernel,
        dim=1,
        polynomial_degree=-1,
    )
    valid_surrogate.fit(train_x, train_y)
    invalid_surrogate.fit(train_x, train_y)

    valid_prediction = valid_surrogate.predict(train_x).value[:, 0]
    invalid_prediction = invalid_surrogate.predict(train_x).value[:, 0]
    valid_residual = np.max(np.abs(valid_prediction - train_y))

    assert np.isfinite(valid_prediction).all()
    if np.isfinite(invalid_prediction).all():
        invalid_residual = np.max(np.abs(invalid_prediction - train_y))
        assert invalid_residual > max(1e-3, 10 * valid_residual)
