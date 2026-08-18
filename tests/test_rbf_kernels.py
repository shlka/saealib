import dataclasses

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.surrogate.rbf_kernels import (
    CubicKernel,
    GaussianKernel,
    LinearKernel,
    MaternKernel,
    MultiquadricKernel,
    RBFKernel,
    ThinPlateSplineKernel,
)


class _ConstantKernel(RBFKernel):
    def evaluate(self, r: np.ndarray) -> np.ndarray:
        return np.ones_like(r)


def test_builtin_kernels_match_closed_formulas():
    r = np.array([[0.0, 1.0, 2.0, 5.0]])

    np.testing.assert_allclose(
        GaussianKernel(length_scale=2.0).evaluate(r),
        np.exp(-0.5 * (r / 2.0) ** 2),
        rtol=1e-10,
    )

    expected = np.zeros_like(r)
    positive = r > 0
    expected[positive] = r[positive] ** 2 * np.log(r[positive])
    np.testing.assert_allclose(
        ThinPlateSplineKernel().evaluate(r),
        expected,
        rtol=1e-10,
    )
    assert ThinPlateSplineKernel().evaluate(np.array([[0.0]]))[0, 0] == 0.0

    np.testing.assert_allclose(LinearKernel().evaluate(r), r, rtol=1e-10)
    np.testing.assert_allclose(CubicKernel().evaluate(r), r**3, rtol=1e-10)

    np.testing.assert_allclose(
        MultiquadricKernel(length_scale=2.0).evaluate(r),
        np.sqrt(r**2 + 2.0**2),
        rtol=1e-10,
    )

    np.testing.assert_allclose(
        MaternKernel(length_scale=2.0, nu=0.5).evaluate(r),
        np.exp(-r / 2.0),
        rtol=1e-10,
    )

    scaled = np.sqrt(3) * r / 2.0
    np.testing.assert_allclose(
        MaternKernel(length_scale=2.0, nu=1.5).evaluate(r),
        (1 + scaled) * np.exp(-scaled),
        rtol=1e-10,
    )

    scaled = np.sqrt(5) * r / 2.0
    np.testing.assert_allclose(
        MaternKernel(length_scale=2.0, nu=2.5).evaluate(r),
        (1 + scaled + scaled**2 / 3) * np.exp(-scaled),
        rtol=1e-10,
    )


def test_min_polynomial_degree_metadata():
    assert GaussianKernel.min_polynomial_degree is None
    assert ThinPlateSplineKernel.min_polynomial_degree == 1
    assert LinearKernel.min_polynomial_degree == 0
    assert CubicKernel.min_polynomial_degree == 1
    assert MultiquadricKernel.min_polynomial_degree == 0
    assert MaternKernel.min_polynomial_degree is None


@pytest.mark.parametrize(
    "kernel_cls", [GaussianKernel, MultiquadricKernel, MaternKernel]
)
@pytest.mark.parametrize("length_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_length_scale_raises_at_construction(kernel_cls, length_scale):
    with pytest.raises(ValidationError):
        kernel_cls(length_scale=length_scale)


@pytest.mark.parametrize("nu", [1.0, 0.0])
def test_invalid_matern_nu_raises_at_construction(nu):
    with pytest.raises(ValidationError):
        MaternKernel(nu=nu)


def test_valid_kernel_configurations_construct():
    GaussianKernel()
    GaussianKernel(length_scale=1.0)
    MaternKernel(nu=1.5)


@pytest.mark.parametrize(
    "kernel",
    [
        GaussianKernel(length_scale=1.0),
        MaternKernel(length_scale=1.0),
        MultiquadricKernel(length_scale=1.0),
    ],
)
def test_length_scale_is_immutable(kernel):
    with pytest.raises(dataclasses.FrozenInstanceError):
        kernel.length_scale = 0.5


def test_resolve_auto_length_scale_does_not_mutate_original():
    configured = GaussianKernel()
    train_x = np.array([[0.0], [1.0], [2.0], [3.0]])

    resolved = configured.resolve(train_x)

    assert configured.length_scale is None
    assert resolved.length_scale is not None
    assert resolved.length_scale > 0


def test_resolve_uses_median_pairwise_distance():
    train_x = np.array([[0.0], [1.0], [3.0]])

    resolved = GaussianKernel().resolve(train_x)

    assert resolved.length_scale == pytest.approx(2.0, rel=1e-10)


def test_resolve_returns_fixed_length_scale_kernel_unchanged():
    train_x = np.array([[0.0], [1.0], [3.0]])
    kernel = GaussianKernel(length_scale=0.7)

    resolved = kernel.resolve(train_x)

    assert resolved is kernel
    assert resolved.length_scale == 0.7


def test_resolve_returns_kernels_without_length_scale_unchanged():
    train_x = np.array([[0.0], [1.0], [3.0]])
    r = np.array([[0.0, 1.0, 2.0]])
    kernel = ThinPlateSplineKernel()

    resolved = kernel.resolve(train_x)

    assert resolved is kernel
    np.testing.assert_allclose(
        resolved.evaluate(r),
        kernel.evaluate(r),
        rtol=1e-10,
    )


class _EagerlyInvalidKernel(RBFKernel):
    """Custom subclass that skips dataclass ``__post_init__``, so
    ``validate()`` is never called at construction — only ``resolve()`` (the
    default implementation) can catch it."""

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        return r

    def validate(self) -> None:
        raise ValidationError("always invalid")


def test_default_resolve_calls_validate_for_custom_subclass():
    kernel = _EagerlyInvalidKernel()
    train_x = np.array([[0.0], [1.0]])

    with pytest.raises(ValidationError, match="always invalid"):
        kernel.resolve(train_x)


@pytest.mark.parametrize(
    "kernel_cls", [GaussianKernel, MultiquadricKernel, MaternKernel]
)
@pytest.mark.parametrize(
    "train_x",
    [
        np.array([[0.0]]),
        np.array([[1.0], [1.0], [1.0]]),
    ],
)
def test_resolve_rejects_degenerate_training_data(kernel_cls, train_x):
    with pytest.raises(ValidationError):
        kernel_cls().resolve(train_x)


@pytest.mark.parametrize(
    "kernel", [GaussianKernel(), MultiquadricKernel(), MaternKernel()]
)
def test_unresolved_length_scale_evaluate_raises_validation_error(kernel):
    with pytest.raises(ValidationError, match="length_scale is unresolved"):
        kernel.evaluate(np.array([[0.0, 1.0]]))


@pytest.mark.parametrize(
    "kernel", [GaussianKernel(), MultiquadricKernel(), MaternKernel()]
)
def test_unresolved_length_scale_pairwise_raises_validation_error(kernel):
    x = np.array([[0.0], [1.0]])

    with pytest.raises(ValidationError, match="length_scale is unresolved"):
        kernel.pairwise(x, x)


@pytest.mark.parametrize(
    "name",
    [
        "gaussian_kernel",
        "thin_plate_spline_kernel",
        "linear_kernel",
        "cubic_kernel",
        "multiquadric_kernel",
        "matern_kernel",
    ],
)
def test_old_kernel_functions_are_not_importable(name):
    import saealib
    import saealib.surrogate
    import saealib.surrogate.rbf

    assert not hasattr(saealib, name)
    assert not hasattr(saealib.surrogate, name)
    assert not hasattr(saealib.surrogate.rbf, name)


def test_pairwise_returns_cross_kernel_matrix_shape():
    x1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x2 = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]],
    )

    result = GaussianKernel(length_scale=1.0).pairwise(x1, x2)

    assert result.shape == (3, 4)


def test_pairwise_self_kernel_is_symmetric_with_unit_diagonal():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])

    result = GaussianKernel(length_scale=1.0).pairwise(x, x)

    np.testing.assert_allclose(np.diag(result), 1.0, rtol=1e-10)
    np.testing.assert_allclose(result, result.T, rtol=1e-10)


def test_custom_kernel_uses_shared_pairwise_helper():
    kernel = _ConstantKernel()

    result = kernel.pairwise(
        np.array([[0.0], [1.0]]),
        np.array([[0.0], [1.0], [2.0]]),
    )

    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, 1.0, rtol=1e-10)
