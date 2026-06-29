"""ZDT multi-objective benchmark suite.

Six two-objective test functions (T1-T6) introduced in:

Zitzler, E., Deb, K., & Thiele, L. (2000).
Comparison of multiobjective evolutionary algorithms: Empirical results.
Evolutionary Computation, 8(2), 173-195.
https://doi.org/10.1162/106365600568202

Section 4, Definition 4, Eq. (7)-(12).
"""

from __future__ import annotations

import numpy as np

from saealib.problem.problem import Problem
from saealib.variables import IntegerVariable

__all__ = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt5", "zdt6"]

_DIR2 = np.array([-1.0, -1.0])


def zdt1(n_var: int = 30) -> Problem:
    """ZDT1 — convex Pareto front.

    Pareto front: f2 = 1 - sqrt(f1), f1 in [0, 1].

    Parameters
    ----------
    n_var : int
        Number of decision variables. Default: 30.

    References
    ----------
    Zitzler, Deb & Thiele (2000) EC 8(2), Section 4 Def. 4, Eq. (7).
    """

    def func(x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (n_var - 1)
        h = 1.0 - np.sqrt(f1 / g)
        return np.array([f1, g * h])

    return Problem(
        func=func,
        dim=n_var,
        n_obj=2,
        direction=_DIR2.copy(),
        lb=[0.0] * n_var,
        ub=[1.0] * n_var,
    )


def zdt2(n_var: int = 30) -> Problem:
    """ZDT2 — non-convex Pareto front.

    Pareto front: f2 = 1 - f1**2, f1 in [0, 1].

    Parameters
    ----------
    n_var : int
        Number of decision variables. Default: 30.

    References
    ----------
    Zitzler, Deb & Thiele (2000) EC 8(2), Section 4 Def. 4, Eq. (8).
    """

    def func(x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (n_var - 1)
        h = 1.0 - (f1 / g) ** 2
        return np.array([f1, g * h])

    return Problem(
        func=func,
        dim=n_var,
        n_obj=2,
        direction=_DIR2.copy(),
        lb=[0.0] * n_var,
        ub=[1.0] * n_var,
    )


def zdt3(n_var: int = 30) -> Problem:
    """ZDT3 — disconnected Pareto front.

    Pareto front: f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1), f1 in [0, 1].
    The front consists of several non-contiguous convex parts.

    Parameters
    ----------
    n_var : int
        Number of decision variables. Default: 30.

    References
    ----------
    Zitzler, Deb & Thiele (2000) EC 8(2), Section 4 Def. 4, Eq. (9).
    """

    def func(x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (n_var - 1)
        ratio = f1 / g
        h = 1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1)
        return np.array([f1, g * h])

    return Problem(
        func=func,
        dim=n_var,
        n_obj=2,
        direction=_DIR2.copy(),
        lb=[0.0] * n_var,
        ub=[1.0] * n_var,
    )


def zdt4(n_var: int = 10) -> Problem:
    """ZDT4 — multimodal; 21^(n_var-1) local Pareto fronts.

    x1 in [0, 1]; x2, ..., x_n in [-5, 5].
    Global Pareto front formed with g(x) = 1: f2 = 1 - sqrt(f1).

    Parameters
    ----------
    n_var : int
        Number of decision variables. Default: 10.

    References
    ----------
    Zitzler, Deb & Thiele (2000) EC 8(2), Section 4 Def. 4, Eq. (10).
    """

    def func(x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = (
            1.0
            + 10.0 * (n_var - 1)
            + np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:]))
        )
        h = 1.0 - np.sqrt(f1 / g)
        return np.array([f1, g * h])

    lb = [0.0] + [-5.0] * (n_var - 1)
    ub = [1.0] + [5.0] * (n_var - 1)

    return Problem(
        func=func,
        dim=n_var,
        n_obj=2,
        direction=_DIR2.copy(),
        lb=lb,
        ub=ub,
    )


def zdt5(
    n_bits_b1: int = 30,
    n_bits_rest: int = 5,
    n_rest: int = 10,
) -> Problem:
    """ZDT5 -- deceptive binary problem.

    x1 is a binary string of n_bits_b1 bits; x2,...,x_m each have n_bits_rest bits.
    Total variables: n_bits_b1 + n_rest * n_bits_rest (default: 80).
    Global Pareto front at g(x)=n_rest: f2 = n_rest/f1, f1 in {1,...,n_bits_b1+1}.

    Parameters
    ----------
    n_bits_b1 : int
        Length of the first binary substring. Default: 30.
    n_bits_rest : int
        Length of each remaining binary substring. Default: 5.
    n_rest : int
        Number of remaining substrings (m-1). Default: 10.

    References
    ----------
    Zitzler, Deb & Thiele (2000) EC 8(2), Section 4 Def. 4, Eq. (11).
    """
    n_var = n_bits_b1 + n_rest * n_bits_rest
    variables = [IntegerVariable(lb=0, ub=1) for _ in range(n_var)]

    def func(x: np.ndarray) -> np.ndarray:
        xi = np.round(x).astype(int)
        u1 = int(np.sum(xi[:n_bits_b1]))
        f1 = 1.0 + u1
        g = 0.0
        for i in range(n_rest):
            start = n_bits_b1 + i * n_bits_rest
            u_i = int(np.sum(xi[start : start + n_bits_rest]))
            g += 1.0 if u_i == n_bits_rest else 2.0 + u_i
        return np.array([f1, g / f1])

    return Problem(
        func=func,
        dim=n_var,
        n_obj=2,
        direction=_DIR2.copy(),
        variables=variables,
    )


def zdt6(n_var: int = 10) -> Problem:
    """ZDT6 — non-uniform distribution; non-convex Pareto front.

    x_i in [0, 1]. Pareto front formed with g(x) = 1.

    Parameters
    ----------
    n_var : int
        Number of decision variables. Default: 10.

    References
    ----------
    Zitzler, Deb & Thiele (2000) EC 8(2), Section 4 Def. 4, Eq. (12).
    """

    def func(x: np.ndarray) -> np.ndarray:
        f1 = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0]) ** 6)
        g = 1.0 + 9.0 * (np.sum(x[1:]) / (n_var - 1)) ** 0.25
        h = 1.0 - (f1 / g) ** 2
        return np.array([f1, g * h])

    return Problem(
        func=func,
        dim=n_var,
        n_obj=2,
        direction=_DIR2.copy(),
        lb=[0.0] * n_var,
        ub=[1.0] * n_var,
    )
