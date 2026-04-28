"""
Termination module.

Termination class defines criteria to stop the optimization process.
Users can specify arbitrary termination conditions as callable objects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saealib.context import OptimizationContext


#: Type alias for a termination condition function.
#: A callable that takes an OptimizationContext and returns True
#: when the optimization should stop.
TerminationCondition = Callable[["OptimizationContext"], bool]


class Termination:
    """
    Termination class to determine when to stop the optimization process.

    Accepts one or more callable conditions. Each condition receives
    an ``OptimizationContext`` and returns ``True`` when the process
    should terminate. The optimization stops when **any** condition
    evaluates to ``True``.

    Parameters
    ----------
    *conditions : TerminationCondition
        One or more callable conditions. Each must accept an
        ``OptimizationContext`` and return ``bool``.

    Raises
    ------
    ValueError
        If no conditions are provided.
    TypeError
        If any condition is not callable.

    Examples
    --------
    >>> termination = Termination(max_fe(2000))
    >>> termination = Termination(max_fe(2000), max_gen(100))
    >>> termination = Termination(
    ...     max_fe(2000),
    ...     lambda ctx: ctx.archive.get("f").min() < 1e-6,
    ... )
    """

    def __init__(self, *conditions: TerminationCondition):
        if not conditions:
            raise ValueError("At least one termination condition must be provided.")
        for cond in conditions:
            if not callable(cond):
                raise TypeError(
                    "Termination condition must be callable, "
                    f"got {type(cond).__name__}."
                )
        self.conditions: tuple[TerminationCondition, ...] = conditions

    def is_terminated(self, ctx: OptimizationContext) -> bool:
        """
        Check if any termination condition is met.

        Parameters
        ----------
        ctx : OptimizationContext
            The current optimization context.

        Returns
        -------
        bool
            True if any termination condition is met, False otherwise.
        """
        return any(cond(ctx) for cond in self.conditions)


# Built-in termination condition factories


def max_fe(value: int) -> TerminationCondition:
    """
    Create a termination condition based on maximum function evaluations.

    Parameters
    ----------
    value : int
        Maximum number of function evaluations.

    Returns
    -------
    TerminationCondition
        A callable that returns True when ``ctx.fe >= value``.

    Examples
    --------
    >>> termination = Termination(max_fe(2000))
    """

    def _condition(ctx: OptimizationContext) -> bool:
        return ctx.fe >= value

    _condition.__doc__ = f"Terminate when fe >= {value}."
    _condition.__qualname__ = f"max_fe({value})"
    return _condition


def max_gen(value: int) -> TerminationCondition:
    """
    Create a termination condition based on maximum generations.

    Parameters
    ----------
    value : int
        Maximum number of generations.

    Returns
    -------
    TerminationCondition
        A callable that returns True when ``ctx.gen >= value``.

    Examples
    --------
    >>> termination = Termination(max_gen(100))
    """

    def _condition(ctx: OptimizationContext) -> bool:
        return ctx.gen >= value

    _condition.__doc__ = f"Terminate when gen >= {value}."
    _condition.__qualname__ = f"max_gen({value})"
    return _condition
