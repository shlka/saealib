"""Stage compatibility surface and the structural pipeline DSL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from saealib.core.compiler.regions import Condition
from saealib.core.contracts import ComponentContract, StateContract
from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.context import OptimizationState

__all__ = ["Branch", "Condition", "Loop", "Pipeline", "Repeat", "Stage"]


class Stage(ABC):
    """
    Abstract base class for a compatibility execution step.

    A Stage receives an :class:`~saealib.context.OptimizationState`, performs
    one well-defined operation, and returns a (possibly new) state.  Structured
    graph execution does not accept this boundary.

    Attributes
    ----------
    name : str
        Short machine-readable identifier used for lookup via
        ``pipeline["name"]``.
    label : str
        Human-readable description of what this stage does.
    notation : str
        LaTeX math expression used by :meth:`to_pseudocode`.
    stages : list[Stage]
        Sub-pipeline for composite stages.  Empty list for leaf stages.
    """

    name: str = ""
    label: str = ""
    notation: str = ""
    stages: list[Stage]
    _execution_mode = "optimization_state"
    _saealib_stage_boundary = True

    def __init__(
        self,
        *,
        name: str = "",
        label: str = "",
        notation: str = "",
    ) -> None:
        """
        Initialize base Stage attributes.

        Parameters
        ----------
        name : str, optional
            Override the class-level ``name``.
        label : str, optional
            Override the class-level ``label``.
        notation : str, optional
            Override the class-level ``notation``.
        """
        if name:
            self.name = name
        if label:
            self.label = label
        if notation:
            self.notation = notation
        self.stages: list[Stage] = []

    @abstractmethod
    def execute(self, state: OptimizationState) -> OptimizationState:
        """
        Execute this stage and return the updated state.

        Parameters
        ----------
        state : OptimizationState
            Current optimization state.

        Returns
        -------
        OptimizationState
            Updated state.
        """

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        r"""
        Render this stage as a LaTeX algorithmic line.

        Parameters
        ----------
        expand : bool
            If True and this stage has sub-stages, recursively expand them.
        indent : int
            Current indentation level (number of ``\State`` indent steps).

        Returns
        -------
        str
            LaTeX algorithmic fragment.
        """
        prefix = "  " * indent
        if expand and self.stages:
            inner = "\n".join(
                s.to_pseudocode(expand=True, indent=indent + 1) for s in self.stages
            )
            label = self.label or self.name or type(self).__name__
            return f"{prefix}\\Comment{{{label}}}\n{inner}"
        notation = self.notation or self.label or self.name or type(self).__name__
        return f"{prefix}\\State {notation}"

    def contract(self) -> ComponentContract:
        """Return this stage's structural and direct state-access contract."""
        return ComponentContract()


def _find_recursive(stages: Sequence[object], name: str) -> object | None:
    for stage in stages:
        if getattr(stage, "name", None) == name:
            return stage
        children = getattr(stage, "stages", None)
        if children is None and isinstance(stage, (Repeat, Loop)):
            children = (stage.body,)
        elif children is None and isinstance(stage, Branch):
            children = (stage.then,)
            if stage.else_ is not None:
                children += (stage.else_,)
        if children is not None and not isinstance(children, (str, bytes)):
            if not isinstance(children, (list, tuple)):
                children = (children,)
            result = _find_recursive(children, name)
            if result is not None:
                return result
    return None


def _validate_dsl_condition(condition: object) -> None:
    if not callable(getattr(condition, "contract", None)):
        raise ValidationError("DSL condition must provide contract()")
    if not callable(getattr(condition, "evaluate", None)):
        raise ValidationError("DSL condition must provide evaluate(context)")
    if not isinstance(condition.contract(), StateContract):
        raise ValidationError("DSL condition contract() must return StateContract")


class _ControlValue:
    """Common identity and pseudocode surface for structural DSL values."""

    name: str = ""
    label: str = ""
    notation: str = ""

    def __init__(self, *, name: str = "", label: str = "", notation: str = "") -> None:
        self.name = name or type(self).__name__.lower()
        self.label = label
        self.notation = notation

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this repeat as a lightweight pseudocode line."""
        prefix = "  " * indent
        return f"{prefix}\\State {self.notation or self.label or self.name}"


class Repeat(_ControlValue):
    """Repeat a structural body a fixed number of times."""

    _structured_kind = "repeat"

    def __init__(self, body: object, count: int, *, name: str = "") -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValidationError("Repeat count must be a non-negative integer")
        super().__init__(name=name, label=f"Repeat {count} times")
        self.body = body
        self.count = count

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this repeat as a lightweight pseudocode line."""
        prefix = "  " * indent
        return f"{prefix}\\State \\Repeat{{{self.count}}}"


class Loop(_ControlValue):
    """Repeat a structural body while a condition remains active."""

    _structured_kind = "loop"

    def __init__(self, body: object, *, until: Condition, name: str = "") -> None:
        _validate_dsl_condition(until)
        super().__init__(name=name, label="Loop")
        self.body = body
        self.until = until
        self.condition = until

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this loop as a lightweight pseudocode line."""
        prefix = "  " * indent
        return f"{prefix}\\State \\Loop{{{self.name}}}"


class Branch(_ControlValue):
    """Select one of two structural bodies using a condition."""

    _structured_kind = "branch"

    def __init__(
        self,
        condition: Condition,
        *,
        then: object,
        else_: object | None = None,
        name: str = "",
    ) -> None:
        _validate_dsl_condition(condition)
        super().__init__(name=name, label="Branch")
        self.condition = condition
        self.then = then
        self.else_ = else_

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this branch as a lightweight pseudocode line."""
        prefix = "  " * indent
        return f"{prefix}\\State \\If{{{self.name}}}"


class Pipeline:
    """
    An ordered structural DSL sequence lowered to a semantic graph.

    A Pipeline has no state execution path. Its entries may be graph-native
    components, nested pipelines, or structured control values such as
    :class:`Repeat`, :class:`Loop`, and :class:`Branch`.

    Parameters
    ----------
    stages : sequence of object, optional
        Ordered structural entries. ``steps`` is the keyword spelling for the
        same value.
    name : str, optional
        Machine-readable identifier for this pipeline.
    label : str, optional
        Human-readable description.
    notation : str, optional
        LaTeX notation for pseudocode generation.
    """

    def __init__(
        self,
        stages: Sequence[object] | None = None,
        name: str = "",
        label: str = "",
        notation: str = "",
        *,
        steps: Sequence[object] | None = None,
    ) -> None:
        if stages is not None and steps is not None:
            raise TypeError("Provide either positional stages or steps=, not both")
        self.name = name
        self.label = label
        self.notation = notation
        self.steps = list(steps if steps is not None else (stages or ()))
        self.stages = self.steps
        self._validate()

    def _validate(self) -> None:
        for stage in self.steps:
            if isinstance(stage, (Pipeline, Repeat, Loop, Branch, Stage)):
                continue
            if not callable(getattr(stage, "contract", None)):
                raise TypeError(
                    f"{stage!r} is not a Stage instance or graph component; "
                    "all elements of a Pipeline must be structural values"
                )

    def __getitem__(self, name: str) -> object:
        """Look up a stage by its ``name`` attribute.

        Parameters
        ----------
        name : str
            The ``name`` of the stage to find.

        Returns
        -------
        structural value

        Raises
        ------
        KeyError
            If no stage with the given name exists.
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def replace(self, name: str, stage: object) -> None:
        """Replace the named entry in the top-level structural sequence.

        Parameters
        ----------
        name : str
            The ``name`` of the stage to replace.
        stage : object
            Replacement structural value.

        Raises
        ------
        KeyError
            If no stage with the given name exists.
        TypeError
            If *stage* is not a structural value.
        """
        if not isinstance(
            stage, (Pipeline, Repeat, Loop, Branch, Stage)
        ) and not callable(getattr(stage, "contract", None)):
            raise TypeError(
                f"{stage!r} is not a Stage instance or graph component; "
                "replacement must be a structural value"
            )
        for i, s in enumerate(self.stages):
            if s.name == name:
                self.stages[i] = stage
                return
        raise KeyError(name)

    def find(self, name: str, *, recursive: bool = False) -> object:
        """Look up a named structural value, optionally recursively.

        Parameters
        ----------
        name : str
            The ``name`` of the stage to find.
        recursive : bool, optional
            If ``True``, descend into nested pipelines and control bodies.
            Defaults to ``False``.

        Returns
        -------
        structural value

        Raises
        ------
        KeyError
            If no stage with the given name exists.
        """
        if not recursive:
            return self[name]
        result = _find_recursive(self.stages, name)
        if result is not None:
            return result
        raise KeyError(name)

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self):
        return iter(self.stages)

    def __repr__(self) -> str:
        names = ", ".join(type(s).__name__ for s in self.stages)
        if self.name:
            return f"Pipeline(name={self.name!r}, stages=[{names}])"
        return f"Pipeline(stages=[{names}])"

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        """Render this pipeline as a LaTeX algorithmic block."""
        if expand and self.stages:
            prefix = "  " * indent
            label = self.label or self.name or "Pipeline"
            inner = "\n".join(
                _to_pseudocode(s, expand=True, indent=indent + 1) for s in self.stages
            )
            return f"{prefix}\\Comment{{{label}}}\n{inner}"
        prefix = "  " * indent
        notation = self.notation or self.label or self.name or "Pipeline"
        return f"{prefix}\\State {notation}"


def _to_pseudocode(value: object, *, expand: bool, indent: int) -> str:
    renderer = getattr(value, "to_pseudocode", None)
    if callable(renderer):
        return renderer(expand=expand, indent=indent)
    prefix = "  " * indent
    name = getattr(value, "name", type(value).__name__)
    return f"{prefix}\\State {name}"
