"""Stage ABC and Pipeline: building blocks for the optimization pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saealib.context import OptimizationState


class Stage(ABC):
    """
    Abstract base class for a single pipeline step.

    A Stage receives an :class:`~saealib.context.OptimizationState`, performs
    one well-defined operation, and returns a (possibly new) state.  Stages are
    composed into a :class:`Pipeline` via sequential ``reduce``.

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


def _find_recursive(stages: list[Stage], name: str) -> Stage | None:
    for stage in stages:
        if stage.name == name:
            return stage
        if stage.stages:
            result = _find_recursive(stage.stages, name)
            if result is not None:
                return result
    return None


class Pipeline(Stage):
    """
    A sequential composition of :class:`Stage` objects.

    Executes each stage in order using ``functools.reduce``.  Can be nested
    inside another ``Pipeline`` because ``Pipeline`` is itself a ``Stage``.

    Parameters
    ----------
    stages : list[Stage]
        Ordered list of stages to execute.
    name : str, optional
        Machine-readable identifier for this pipeline.
    label : str, optional
        Human-readable description.
    notation : str, optional
        LaTeX notation for pseudocode generation.
    """

    def __init__(
        self,
        stages: list[Stage],
        name: str = "",
        label: str = "",
        notation: str = "",
    ) -> None:
        super().__init__(name=name, label=label, notation=notation)
        self.stages = stages
        self._validate()

    def _validate(self) -> None:
        for stage in self.stages:
            if not isinstance(stage, Stage):
                raise TypeError(
                    f"{stage!r} is not a Stage instance; "
                    "all elements of a Pipeline must be Stage subclasses"
                )

    def execute(self, state: OptimizationState) -> OptimizationState:
        """Execute all stages sequentially, threading state through each."""
        return reduce(lambda s, stage: stage.execute(s), self.stages, state)

    def __getitem__(self, name: str) -> Stage:
        """Look up a stage by its ``name`` attribute.

        Parameters
        ----------
        name : str
            The ``name`` of the stage to find.

        Returns
        -------
        Stage

        Raises
        ------
        KeyError
            If no stage with the given name exists.
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def replace(self, name: str, stage: Stage) -> None:
        """Replace the stage named *name* in the top-level stages list.

        Parameters
        ----------
        name : str
            The ``name`` of the stage to replace.
        stage : Stage
            Replacement stage.

        Raises
        ------
        KeyError
            If no stage with the given name exists.
        TypeError
            If *stage* is not a ``Stage`` instance.
        """
        if not isinstance(stage, Stage):
            raise TypeError(
                f"{stage!r} is not a Stage instance; "
                "replacement must be a Stage subclass"
            )
        for i, s in enumerate(self.stages):
            if s.name == name:
                self.stages[i] = stage
                return
        raise KeyError(name)

    def find(self, name: str, *, recursive: bool = False) -> Stage:
        """Look up a stage by name, optionally descending into nested stages.

        Parameters
        ----------
        name : str
            The ``name`` of the stage to find.
        recursive : bool, optional
            If ``True``, descend into stages that expose a ``stages`` attribute
            (e.g., nested :class:`Pipeline` or
            :class:`~saealib.stages.SurrogateOnlyLoopStage`).
            Defaults to ``False``.

        Returns
        -------
        Stage

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
        """Return the number of top-level stages."""
        return len(self.stages)

    def __iter__(self):
        """Iterate over the top-level stages."""
        return iter(self.stages)

    def __repr__(self) -> str:
        """Return a concise developer-facing string for this pipeline."""
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
                s.to_pseudocode(expand=True, indent=indent + 1) for s in self.stages
            )
            return f"{prefix}\\Comment{{{label}}}\n{inner}"
        return super().to_pseudocode(expand=expand, indent=indent)
