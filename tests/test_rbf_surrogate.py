import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.surrogate.rbf import RBFSurrogate
from saealib.surrogate.rbf_kernels import (
    CubicKernel,
    GaussianKernel,
    LinearKernel,
    MaternKernel,
    MultiquadricKernel,
    RBFKernel,
    ThinPlateSplineKernel,
)

TRAIN_X = np.array([[-1.0], [0.0], [1.0]])
TRAIN_Y = np.array([1.0, 2.0, -0.5])


def test_default_degree_and_solver_interpolate_training_points():
    surrogate = RBFSurrogate(kernel=GaussianKernel(), dim=1)
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
        kernel=GaussianKernel(),
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
        kernel=GaussianKernel(),
        dim=1,
        polynomial_degree=1,
    )
    polynomial_surrogate.fit(TRAIN_X, train_y)
    polynomial_prediction = polynomial_surrogate.predict(outside_x).value[:, 0]

    default_surrogate = RBFSurrogate(kernel=GaussianKernel(), dim=1)
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
        kernel=ThinPlateSplineKernel(),
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
    solve_surrogate = RBFSurrogate(kernel=GaussianKernel(), dim=1)
    solve_surrogate.fit(TRAIN_X, TRAIN_Y)
    solve_prediction = solve_surrogate.predict(TRAIN_X).value[:, 0]

    predictions = []
    residuals = []
    for alpha in (1e-6, 1e-3, 1.0):
        surrogate = RBFSurrogate(
            kernel=GaussianKernel(),
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
        kernel=GaussianKernel(),
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
            kernel=GaussianKernel(),
            dim=1,
            polynomial_degree=polynomial_degree,
        )


@pytest.mark.parametrize("solver", ["invalid", "qr"])
def test_invalid_solver_raises_validation_error(solver):
    with pytest.raises(ValidationError):
        RBFSurrogate(kernel=GaussianKernel(), dim=1, solver=solver)


@pytest.mark.parametrize("solver", ["solve", "lstsq", "tikhonov"])
@pytest.mark.parametrize("alpha", [0.0, -1.0])
def test_nonpositive_alpha_raises_validation_error(solver, alpha):
    with pytest.raises(ValidationError):
        RBFSurrogate(kernel=GaussianKernel(), dim=1, solver=solver, alpha=alpha)


@pytest.mark.parametrize(
    ("kernel", "polynomial_degree"),
    [
        (LinearKernel(), 0),
        (CubicKernel(), 1),
        (MultiquadricKernel(), 0),
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
    surrogate = RBFSurrogate(kernel=MaternKernel(nu=nu), dim=1)
    surrogate.fit(TRAIN_X, TRAIN_Y)

    np.testing.assert_allclose(
        surrogate.predict(TRAIN_X).value[:, 0],
        TRAIN_Y,
        rtol=1e-10,
        atol=1e-10,
    )


def test_matern_kernel_rejects_invalid_nu_at_construction_time():
    with pytest.raises(ValidationError, match=r"nu must be 0\.5, 1\.5, or 2\.5"):
        MaternKernel(nu=1.0)


def test_multiquadric_without_required_constant_term_rejects_construction():
    with pytest.raises(
        ValidationError,
        match=r"MultiquadricKernel requires polynomial_degree >= 0, got -1",
    ):
        RBFSurrogate(
            kernel=MultiquadricKernel(),
            dim=1,
            polynomial_degree=-1,
        )


@pytest.mark.parametrize(
    ("kernel", "expected_degree"),
    [
        (GaussianKernel(), -1),
        (LinearKernel(), 0),
        (CubicKernel(), 1),
        (ThinPlateSplineKernel(), 1),
        (MultiquadricKernel(), 0),
    ],
)
def test_auto_polynomial_degree_resolution(kernel, expected_degree):
    surrogate = RBFSurrogate(kernel=kernel, dim=1)

    assert surrogate.polynomial_degree == expected_degree


@pytest.mark.parametrize(
    ("kernel", "polynomial_degree"),
    [
        (CubicKernel(), 0),
        (ThinPlateSplineKernel(), 0),
        (ThinPlateSplineKernel(), -1),
        (LinearKernel(), -1),
        (MultiquadricKernel(), -1),
    ],
)
def test_invalid_polynomial_degree_kernel_combinations_raise_validation_error(
    kernel, polynomial_degree
):
    with pytest.raises(ValidationError):
        RBFSurrogate(
            kernel=kernel,
            dim=1,
            polynomial_degree=polynomial_degree,
        )


def test_linear_kernel_with_degree_one_is_allowed():
    surrogate = RBFSurrogate(
        kernel=LinearKernel(),
        dim=1,
        polynomial_degree=1,
    )

    surrogate.fit(TRAIN_X, TRAIN_Y)
    surrogate.predict(TRAIN_X)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("kernel", GaussianKernel(length_scale=0.5)),
        ("polynomial_degree", 0),
        ("solver", "lstsq"),
        ("alpha", 1e-3),
    ],
)
def test_configuration_replacement_invalidates_fitted_state(attribute, value):
    surrogate = RBFSurrogate(kernel=GaussianKernel(), dim=1)
    surrogate.fit(TRAIN_X, TRAIN_Y)
    setattr(surrogate, attribute, value)

    with pytest.raises(RuntimeError):
        surrogate.predict(TRAIN_X)

    surrogate.fit(TRAIN_X, TRAIN_Y)
    surrogate.predict(TRAIN_X)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("polynomial_degree", 2), ("solver", "bad"), ("alpha", 0.0)],
)
def test_invalid_configuration_replacement_preserves_fitted_state(attribute, value):
    surrogate = RBFSurrogate(kernel=GaussianKernel(), dim=1)
    surrogate.fit(TRAIN_X, TRAIN_Y)

    with pytest.raises(ValidationError):
        setattr(surrogate, attribute, value)

    surrogate.predict(TRAIN_X)


def test_failed_fit_clears_previous_fitted_state():
    surrogate = RBFSurrogate(
        kernel=GaussianKernel(length_scale=1.0),
        dim=1,
        polynomial_degree=1,
    )
    surrogate.fit(TRAIN_X, TRAIN_Y)

    degenerate_x = np.array([[1.0], [1.0], [1.0]])
    with pytest.raises(ValidationError, match=r"rank|degenerate"):
        surrogate.fit(degenerate_x, np.ones(3))

    with pytest.raises(RuntimeError):
        surrogate.predict(TRAIN_X)


@pytest.mark.parametrize(
    ("train_x", "train_y"),
    [
        (TRAIN_X[:, 0], TRAIN_Y),
        (np.empty((0, 1)), np.empty(0)),
        (np.array([[np.nan], [0.0], [1.0]]), TRAIN_Y),
        (np.array([[np.inf], [0.0], [1.0]]), TRAIN_Y),
        (TRAIN_X, np.array([1.0, np.nan, 3.0])),
        (TRAIN_X, np.array([1.0, np.inf, 3.0])),
        (TRAIN_X, np.ones(2)),
    ],
    ids=[
        "train_x_1d",
        "empty_training_set",
        "train_x_nan",
        "train_x_inf",
        "train_y_nan",
        "train_y_inf",
        "sample_count_mismatch",
    ],
)
def test_fit_rejects_invalid_training_inputs(train_x, train_y):
    surrogate = RBFSurrogate(kernel=GaussianKernel(length_scale=1.0), dim=1)

    with pytest.raises(ValidationError):
        surrogate.fit(train_x, train_y)


def test_fit_rejects_non_1d_or_2d_training_targets():
    surrogate = RBFSurrogate(kernel=GaussianKernel(length_scale=1.0), dim=1)

    with pytest.raises(ValidationError):
        surrogate.fit(TRAIN_X, np.ones((3, 1, 1)))


def test_predict_rejects_nonfinite_inputs_and_feature_count_mismatch():
    surrogate = RBFSurrogate(kernel=GaussianKernel(), dim=1)
    surrogate.fit(TRAIN_X, TRAIN_Y)

    for test_x in (np.array([[np.nan]]), np.array([[np.inf]])):
        with pytest.raises(ValidationError):
            surrogate.predict(test_x)

    with pytest.raises(ValidationError, match="features"):
        surrogate.predict(np.ones((1, 2)))


def test_predict_before_fit_raises_runtime_error():
    surrogate = RBFSurrogate(kernel=GaussianKernel(), dim=1)

    with pytest.raises(RuntimeError):
        surrogate.predict(TRAIN_X)


class _CustomKernel(RBFKernel):
    def evaluate(self, r: np.ndarray) -> np.ndarray:
        return np.exp(-r)


def test_custom_rbf_kernel_subclass_works_with_surrogate():
    surrogate = RBFSurrogate(kernel=_CustomKernel(), dim=1)

    surrogate.fit(TRAIN_X, TRAIN_Y)
    prediction = surrogate.predict(TRAIN_X).value

    assert np.isfinite(prediction).all()


def test_fixed_length_scale_bypasses_auto_resolution():
    resolved = GaussianKernel(length_scale=0.5).resolve(TRAIN_X)

    assert resolved.length_scale == 0.5


@pytest.mark.parametrize(
    "train_x",
    [
        np.array([[0.0]]),
        np.array([[1.0], [1.0], [1.0]]),
    ],
)
def test_degenerate_auto_length_scale_raises_validation_error(train_x):
    surrogate = RBFSurrogate(kernel=GaussianKernel(), dim=1)

    with pytest.raises(ValidationError):
        surrogate.fit(train_x, np.ones(len(train_x)))
