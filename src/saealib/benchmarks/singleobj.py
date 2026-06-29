"""Classical single-objective benchmark functions.

Functions and bounds from:

Jamil, M., & Yang, X.-S. (2013).
A literature survey of benchmark functions for global optimisation problems.
International Journal of Mathematical Modelling and Numerical Optimisation,
4(2), 150-194.
https://doi.org/10.1504/IJMMNO.2013.055204

Notes
-----
- Sphere: Jamil & Yang (2013) f137.
- Rosenbrock: Jamil & Yang (2013) f105.
- Ackley: Jamil & Yang (2013) f1; standard a=20, b=0.2, c=2*pi.
- Rastrigin: well-known scalable function; Jamil & Yang list it (f84) but
  that page is absent from the available PDF conversion.  The formula
  f(x) = A*D + sum(xi^2 - A*cos(2*pi*xi)) with A=10 is used universally.
"""

from __future__ import annotations

import numpy as np

from saealib.problem.problem import Problem

__all__ = ["ackley", "rastrigin", "rosenbrock", "sphere"]

_TWO_PI = 2.0 * np.pi


def sphere(n_var: int = 10) -> Problem:
    """Sphere function -- unimodal, separable, global min at origin.

    f(x) = sum(xi^2),  x* = (0,...,0),  f* = 0.

    Parameters
    ----------
    n_var : int
        Number of variables. Default: 10.

    References
    ----------
    Jamil & Yang (2013) IJMMNO 4(2), f137.
    """

    def func(x: np.ndarray) -> np.ndarray:
        return np.array([float(np.sum(x**2))])

    return Problem(
        func=func,
        dim=n_var,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.12] * n_var,
        ub=[5.12] * n_var,
    )


def rosenbrock(n_var: int = 10) -> Problem:
    """Rosenbrock (banana) function -- narrow curved valley, global min at (1,...,1).

    f(x) = sum_{i=1}^{D-1} [100*(x_{i+1}-xi^2)^2 + (xi-1)^2],
    x* = (1,...,1),  f* = 0.

    Parameters
    ----------
    n_var : int
        Number of variables. Default: 10.

    References
    ----------
    Jamil & Yang (2013) IJMMNO 4(2), f105.
    """

    def func(x: np.ndarray) -> np.ndarray:
        val = np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)
        return np.array([float(val)])

    return Problem(
        func=func,
        dim=n_var,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-30.0] * n_var,
        ub=[30.0] * n_var,
    )


def ackley(n_var: int = 10) -> Problem:
    """Ackley function -- nearly flat outer region, deep global minimum at origin.

    f(x) = -a*exp(-b*sqrt(1/D*sum(xi^2))) - exp(1/D*sum(cos(c*xi))) + a + exp(1),
    a=20, b=0.2, c=2*pi.  x* = (0,...,0),  f* = 0.

    Parameters
    ----------
    n_var : int
        Number of variables. Default: 10.

    References
    ----------
    Jamil & Yang (2013) IJMMNO 4(2), f1; standard a=20, b=0.2, c=2*pi.
    """

    def func(x: np.ndarray) -> np.ndarray:
        d = len(x)
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
        term2 = -np.exp(np.sum(np.cos(_TWO_PI * x)) / d)
        return np.array([float(term1 + term2 + 20.0 + np.e)])

    return Problem(
        func=func,
        dim=n_var,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-35.0] * n_var,
        ub=[35.0] * n_var,
    )


def rastrigin(n_var: int = 10, amplitude: float = 10.0) -> Problem:
    """Rastrigin function -- highly multimodal with regular local minima grid.

    f(x) = amplitude*D + sum(xi^2 - amplitude*cos(2*pi*xi)),  x* = (0,...,0),  f* = 0.

    Parameters
    ----------
    n_var : int
        Number of variables. Default: 10.
    amplitude : float
        Amplitude of cosine perturbation. Default: 10.

    References
    ----------
    Standard Rastrigin function; see Jamil & Yang (2013) IJMMNO 4(2) f84
    (page not captured in available PDF conversion).
    """

    def func(x: np.ndarray) -> np.ndarray:
        val = amplitude * len(x) + float(np.sum(x**2 - amplitude * np.cos(_TWO_PI * x)))
        return np.array([val])

    return Problem(
        func=func,
        dim=n_var,
        n_obj=1,
        direction=np.array([-1.0]),
        lb=[-5.12] * n_var,
        ub=[5.12] * n_var,
    )
