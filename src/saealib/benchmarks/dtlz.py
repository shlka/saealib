"""DTLZ multi-objective benchmark suite.

Test problems DTLZ1-7 introduced in:

Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005).
Scalable test problems for evolutionary multiobjective optimization.
In A. Abraham, L. Jain, & R. Goldberg (Eds.),
Evolutionary Multiobjective Optimization (pp. 105-145). Springer.
https://doi.org/10.1007/1-84628-137-7_6

Section 6.7, Eq. (6.18)-(6.25).
"""

from __future__ import annotations

import numpy as np

from saealib.problem.problem import Problem

__all__ = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"]

_HALF_PI = 0.5 * np.pi


def dtlz1(n_obj: int = 3, k: int = 5) -> Problem:
    """DTLZ1 -- linear Pareto hyperplane; (11^k - 1) local Pareto fronts.

    Position variables x1,...,x_{M-1} in [0,1]; k distance variables in [0,1].
    Pareto front: g(x_M)=0 (all x_i=0.5), sum(f_i)=0.5.

    Parameters
    ----------
    n_obj : int
        Number of objectives M. Default: 3.
    k : int
        Number of distance variables; n_var = n_obj - 1 + k. Default: 5.

    References
    ----------
    :cite:`deb2005dtlz`: Deb, K., Thiele, L., Laumanns, M., & Zitzler,
    E. (2005). Scalable test problems for evolutionary multiobjective
    optimization. In *Evolutionary Multiobjective Optimization*
    (pp. 105-145). Springer. (Section 6.7, Eq. (6.18)-(6.19).)
    """
    n_var = n_obj - 1 + k

    def func(x: np.ndarray) -> np.ndarray:
        x_m = x[n_obj - 1 :]
        g = 100.0 * (k + np.sum((x_m - 0.5) ** 2 - np.cos(20.0 * np.pi * (x_m - 0.5))))
        half_g = 0.5 * (1.0 + g)
        f = np.empty(n_obj)
        for i in range(n_obj):
            if i == 0:
                f[i] = half_g * np.prod(x[: n_obj - 1])
            else:
                f[i] = half_g * np.prod(x[: n_obj - 1 - i]) * (1.0 - x[n_obj - 1 - i])
        return f

    return Problem(
        func=func,
        dim=n_var,
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0),
        lb=[0.0] * n_var,
        ub=[1.0] * n_var,
    )


def dtlz2(n_obj: int = 3, k: int = 10) -> Problem:
    """DTLZ2 -- spherical Pareto front; tests distribution on unit hypersphere.

    Position variables x1,...,x_{M-1} in [0,1]; k distance variables in [0,1].
    Pareto front: g(x_M)=0 (all x_i=0.5), sum(f_i^2)=1.

    Parameters
    ----------
    n_obj : int
        Number of objectives M. Default: 3.
    k : int
        Number of distance variables. Default: 10.

    References
    ----------
    :cite:`deb2005dtlz`: Deb, K., Thiele, L., Laumanns, M., & Zitzler,
    E. (2005). Scalable test problems for evolutionary multiobjective
    optimization. In *Evolutionary Multiobjective Optimization*
    (pp. 105-145). Springer. (Section 6.7, Eq. (6.20).)
    """
    n_var = n_obj - 1 + k

    def func(x: np.ndarray) -> np.ndarray:
        theta = x[: n_obj - 1] * _HALF_PI
        g = float(np.sum((x[n_obj - 1 :] - 0.5) ** 2))
        return _sphere_objectives(theta, g, n_obj)

    return Problem(
        func=func,
        dim=n_var,
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0),
        lb=[0.0] * n_var,
        ub=[1.0] * n_var,
    )


def dtlz3(n_obj: int = 3, k: int = 10) -> Problem:
    """DTLZ3 -- spherical Pareto front; 3^k local Pareto fronts.

    Same spherical objective shape as DTLZ2 with DTLZ1's multimodal g function.
    Pareto front: g(x_M)=0 (all x_i=0.5), sum(f_i^2)=1.

    Parameters
    ----------
    n_obj : int
        Number of objectives M. Default: 3.
    k : int
        Number of distance variables. Default: 10.

    References
    ----------
    :cite:`deb2005dtlz`: Deb, K., Thiele, L., Laumanns, M., & Zitzler,
    E. (2005). Scalable test problems for evolutionary multiobjective
    optimization. In *Evolutionary Multiobjective Optimization*
    (pp. 105-145). Springer. (Section 6.7, Eq. (6.21).)
    """
    n_var = n_obj - 1 + k

    def func(x: np.ndarray) -> np.ndarray:
        theta = x[: n_obj - 1] * _HALF_PI
        x_m = x[n_obj - 1 :]
        g = 100.0 * (k + np.sum((x_m - 0.5) ** 2 - np.cos(20.0 * np.pi * (x_m - 0.5))))
        return _sphere_objectives(theta, g, n_obj)

    return Problem(
        func=func,
        dim=n_var,
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0),
        lb=[0.0] * n_var,
        ub=[1.0] * n_var,
    )


def dtlz4(n_obj: int = 3, k: int = 10, alpha: float = 100.0) -> Problem:
    """DTLZ4 -- biased solution density on spherical Pareto front.

    Modifies DTLZ2 with x_i^alpha mapping; solutions cluster near the f_M-f_1 plane.
    Pareto front: g(x_M)=0 (all x_i=0.5), sum(f_i^2)=1.

    Parameters
    ----------
    n_obj : int
        Number of objectives M. Default: 3.
    k : int
        Number of distance variables. Default: 10.
    alpha : float
        Density bias exponent. Default: 100.

    References
    ----------
    :cite:`deb2005dtlz`: Deb, K., Thiele, L., Laumanns, M., & Zitzler,
    E. (2005). Scalable test problems for evolutionary multiobjective
    optimization. In *Evolutionary Multiobjective Optimization*
    (pp. 105-145). Springer. (Section 6.7, Eq. (6.22).)
    """
    n_var = n_obj - 1 + k

    def func(x: np.ndarray) -> np.ndarray:
        theta = (x[: n_obj - 1] ** alpha) * _HALF_PI
        g = float(np.sum((x[n_obj - 1 :] - 0.5) ** 2))
        return _sphere_objectives(theta, g, n_obj)

    return Problem(
        func=func,
        dim=n_var,
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0),
        lb=[0.0] * n_var,
        ub=[1.0] * n_var,
    )


def _sphere_objectives(theta: np.ndarray, g: float, n_obj: int) -> np.ndarray:
    """Compute spherical objectives shared by DTLZ2/3/4/5/6."""
    cos_theta = np.cos(theta)
    f = np.empty(n_obj)
    for i in range(n_obj):
        prod = (1.0 + g) * float(np.prod(cos_theta[: n_obj - 1 - i]))
        if i > 0:
            prod *= np.sin(theta[n_obj - 1 - i])
        f[i] = prod
    return f


def dtlz5(n_obj: int = 3, k: int = 10) -> Problem:
    """DTLZ5 -- degenerate 1-D Pareto curve on unit hypersphere.

    Replaces the uniform theta mapping of DTLZ2 with a g-dependent mapping
    (Eq. 6.10) that collapses the PF to a curve: only x_1 varies freely.
    Pareto front: g(x_M)=0 (all x_i=0.5), theta_i=pi/4 for i>=2, sum(f_i^2)=1.

    Parameters
    ----------
    n_obj : int
        Number of objectives M. Default: 3.
    k : int
        Number of distance variables. Default: 10.

    References
    ----------
    :cite:`deb2005dtlz`: Deb, K., Thiele, L., Laumanns, M., & Zitzler,
    E. (2005). Scalable test problems for evolutionary multiobjective
    optimization. In *Evolutionary Multiobjective Optimization*
    (pp. 105-145). Springer. (Section 6.7, Eq. (6.23).)
    """
    n_var = n_obj - 1 + k

    def func(x: np.ndarray) -> np.ndarray:
        g = float(np.sum((x[n_obj - 1 :] - 0.5) ** 2))
        theta = np.empty(n_obj - 1)
        theta[0] = x[0] * _HALF_PI
        for i in range(1, n_obj - 1):
            theta[i] = np.pi / (4.0 * (1.0 + g)) * (1.0 + 2.0 * g * x[i])
        return _sphere_objectives(theta, g, n_obj)

    return Problem(
        func=func,
        dim=n_var,
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0),
        lb=[0.0] * n_var,
        ub=[1.0] * n_var,
    )


def dtlz6(n_obj: int = 3, k: int = 10) -> Problem:
    """DTLZ6 -- harder variant of DTLZ5 with nonlinear g function.

    Same degenerate PF structure as DTLZ5 (1-D curve on unit sphere) but
    g = sum(x_i^0.1) makes convergence harder than DTLZ5.
    Pareto front: g(x_M)=0 (all x_i=0), sum(f_i^2)=1.

    Parameters
    ----------
    n_obj : int
        Number of objectives M. Default: 3.
    k : int
        Number of distance variables. Default: 10.

    References
    ----------
    :cite:`deb2005dtlz`: Deb, K., Thiele, L., Laumanns, M., & Zitzler,
    E. (2005). Scalable test problems for evolutionary multiobjective
    optimization. In *Evolutionary Multiobjective Optimization*
    (pp. 105-145). Springer. (Section 6.7, Eq. (6.24).)
    """
    n_var = n_obj - 1 + k

    def func(x: np.ndarray) -> np.ndarray:
        g = float(np.sum(x[n_obj - 1 :] ** 0.1))
        theta = np.empty(n_obj - 1)
        theta[0] = x[0] * _HALF_PI
        for i in range(1, n_obj - 1):
            theta[i] = np.pi / (4.0 * (1.0 + g)) * (1.0 + 2.0 * g * x[i])
        return _sphere_objectives(theta, g, n_obj)

    return Problem(
        func=func,
        dim=n_var,
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0),
        lb=[0.0] * n_var,
        ub=[1.0] * n_var,
    )


def dtlz7(n_obj: int = 3, k: int = 20) -> Problem:
    """DTLZ7 -- 2^{M-1} disconnected Pareto fronts.

    f_i = x_i for i=1,...,M-1; f_M couples through the h function.
    Distance variables x_M promote diversity in the last objective.

    Parameters
    ----------
    n_obj : int
        Number of objectives M. Default: 3.
    k : int
        Number of distance variables. Default: 20.

    References
    ----------
    :cite:`deb2005dtlz`: Deb, K., Thiele, L., Laumanns, M., & Zitzler,
    E. (2005). Scalable test problems for evolutionary multiobjective
    optimization. In *Evolutionary Multiobjective Optimization*
    (pp. 105-145). Springer. (Section 6.7, Eq. (6.25).)
    """
    n_var = n_obj - 1 + k

    def func(x: np.ndarray) -> np.ndarray:
        f = np.empty(n_obj)
        f[: n_obj - 1] = x[: n_obj - 1]
        g = 1.0 + (9.0 / k) * np.sum(x[n_obj - 1 :])
        fi = f[: n_obj - 1]
        h = n_obj - float(np.sum(fi / (1.0 + g) * (1.0 + np.sin(3.0 * np.pi * fi))))
        f[n_obj - 1] = (1.0 + g) * h
        return f

    return Problem(
        func=func,
        dim=n_var,
        n_obj=n_obj,
        direction=np.full(n_obj, -1.0),
        lb=[0.0] * n_var,
        ub=[1.0] * n_var,
    )
