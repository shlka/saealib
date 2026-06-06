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


#: Type alias for a plain termination condition function.
#: A callable that takes an ``OptimizationContext`` and returns ``True``
#: when the optimization should stop. Accepted everywhere a
#: :class:`TerminationCondition` is, and wrapped automatically when composed.
ConditionFunc = Callable[["OptimizationContext"], bool]


class TerminationCondition:
    """
    Composable termination condition.

    Wraps a callable ``ctx -> bool`` and adds logical operators so that
    conditions can be combined declaratively:

    - ``a | b`` : terminate when **either** ``a`` or ``b`` is met (OR)
    - ``a & b`` : terminate when **both** ``a`` and ``b`` are met (AND)
    - ``~a``    : terminate when ``a`` is **not** met (NOT)

    The other operand may be a plain ``Callable[[ctx], bool]``; it is
    wrapped automatically. Reflected operators (``__ror__`` / ``__rand__``)
    let a plain callable appear on the left-hand side as well.

    Parameters
    ----------
    func : Callable[[OptimizationContext], bool]
        The underlying condition callable.
    name : str, optional
        Human-readable name used for ``repr`` and ``__qualname__``.
        Defaults to ``func.__qualname__`` when available.
    doc : str, optional
        Docstring for the wrapped condition. Defaults to ``func.__doc__``.

    Raises
    ------
    TypeError
        If ``func`` is not callable.

    Examples
    --------
    >>> cond = max_fe(2000) & (lambda ctx: ctx.archive.get("f").min() < 1e-6)
    >>> cond = max_gen(100) | max_fe(2000)
    >>> cond = ~stalled(20)
    """

    def __init__(
        self,
        func: ConditionFunc,
        *,
        name: str | None = None,
        doc: str | None = None,
    ) -> None:
        if not callable(func):
            raise TypeError(
                "Termination condition must be callable, "
                f"got {type(func).__name__}."
            )
        self._func = func
        self.__qualname__ = (
            name if name is not None else getattr(func, "__qualname__", repr(func))
        )
        resolved_doc = doc if doc is not None else getattr(func, "__doc__", None)
        if resolved_doc is not None:
            self.__doc__ = resolved_doc

    @staticmethod
    def _coerce(cond: ConditionFunc | TerminationCondition) -> TerminationCondition:
        """Wrap a plain callable into a ``TerminationCondition`` if needed."""
        if isinstance(cond, TerminationCondition):
            return cond
        return TerminationCondition(cond)

    def __call__(self, ctx: OptimizationContext) -> bool:
        """Evaluate the wrapped condition against ``ctx``."""
        return bool(self._func(ctx))

    def __or__(
        self, other: ConditionFunc | TerminationCondition
    ) -> TerminationCondition:
        """Compose with OR: terminate when either condition is met."""
        other = self._coerce(other)
        return TerminationCondition(
            lambda ctx: self(ctx) or other(ctx),
            name=f"({self.__qualname__} | {other.__qualname__})",
        )

    def __ror__(
        self, other: ConditionFunc | TerminationCondition
    ) -> TerminationCondition:
        """Compose with OR when a plain callable is on the left-hand side."""
        return self._coerce(other) | self

    def __and__(
        self, other: ConditionFunc | TerminationCondition
    ) -> TerminationCondition:
        """Compose with AND: terminate when both conditions are met."""
        other = self._coerce(other)
        return TerminationCondition(
            lambda ctx: self(ctx) and other(ctx),
            name=f"({self.__qualname__} & {other.__qualname__})",
        )

    def __rand__(
        self, other: ConditionFunc | TerminationCondition
    ) -> TerminationCondition:
        """Compose with AND when a plain callable is on the left-hand side."""
        return self._coerce(other) & self

    def __invert__(self) -> TerminationCondition:
        """Negate the condition with NOT."""
        return TerminationCondition(
            lambda ctx: not self(ctx),
            name=f"~{self.__qualname__}",
        )

    def __repr__(self) -> str:
        """Return a readable representation including the condition name."""
        return f"TerminationCondition({self.__qualname__})"


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


def max_fe(value: int) -> ConditionFunc:
    """
    Create a termination condition based on maximum function evaluations.

    Parameters
    ----------
    value : int
        Maximum number of function evaluations.

    Returns
    -------
    ConditionFunc
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


def max_gen(value: int) -> ConditionFunc:
    """
    Create a termination condition based on maximum generations.

    Parameters
    ----------
    value : int
        Maximum number of generations.

    Returns
    -------
    ConditionFunc
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
