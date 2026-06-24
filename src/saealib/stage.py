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
    stages : list[Stage] or None
        Sub-pipeline for composite stages.  ``None`` for leaf stages.
    """

    name: str = ""
    label: str = ""
    notation: str = ""
    stages: list[Stage] | None = None

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
        self.stages = stages
        self.name = name
        self.label = label
        self.notation = notation
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
