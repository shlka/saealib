"""Optimizer: assembles and runs the surrogate-assisted EA pipeline."""

from __future__ import annotations

import dataclasses
import pickle
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from typing_extensions import Self

from saealib.acquisition.mean import MeanPrediction
from saealib.callback import (
    CallbackManager,
    Event,
    GenerationStartEvent,
    logging_generation,
)
from saealib.context import OptimizationState
from saealib.exceptions import ConfigurationError
from saealib.execution.evaluator import Evaluator, SerialEvaluator
from saealib.execution.runner import Runner
from saealib.surrogate.manager import LocalSurrogateManager, SurrogateManager

if TYPE_CHECKING:
    from saealib.algorithms.base import Algorithm
    from saealib.execution.initializer import Initializer
    from saealib.problem import Problem
    from saealib.strategies.base import OptimizationStrategy
    from saealib.surrogate.base import Surrogate
    from saealib.termination import Termination


class Dispatchable(Protocol):
    """Minimal interface for objects that can dispatch callback events."""

    def dispatch(self, event: Event) -> None:
        """Dispatch a callback event."""
        ...


class ComponentProvider(Protocol):
    """The interface for components that can be used by the Optimizer."""

    @property
    def seed(self) -> int | None:
        """Return the master random seed."""
        ...

    @property
    def algorithm(self) -> Algorithm:
        """Return the algorithm instance."""
        ...

    @property
    def strategy(self) -> OptimizationStrategy:
        """Return the optimization strategy instance."""
        ...

    @property
    def surrogate_manager(self) -> SurrogateManager:
        """Return the surrogate manager instance."""
        ...

    @property
    def evaluator(self) -> Evaluator:
        """Return the evaluator instance."""
        ...

    @property
    def termination(self) -> Termination:
        """Return the termination condition."""
        ...

    @property
    def cbmanager(self) -> CallbackManager:
        """Return the callback manager."""
        ...

    def dispatch(self, event: Event) -> None:
        """Dispatch a callback event."""
        ...


# class Optimizer(ComponentProvider):
class Optimizer:
    """
    Optimizer class for evolutionary algorithms.

    Integrates problem definition, evolutionary algorithm, surrogate model,
    model manager, and termination condition, and manages the optimization process.

    Attributes
    ----------
    problem : Problem
        The optimization problem.
    algorithm : Algorithm
        The evolutionary algorithm.
    surrogate : Surrogate
        The surrogate model.
    strategy : OptimizationStrategy
        The optimization strategy.
    termination : Termination
        The termination condition.
    archive : Archive
        The archive of evaluated solutions.
    popsize : int
        The population size.
    seed : int or None
        The master random seed.  ``None`` means non-deterministic.
    fe : int
        The current number of function evaluations.
    gen : int
        The current generation number.
    cbmanager : CallbackManager
        The callback event manager.
    instance_name : str
        The name of the optimizer instance.
    """

    def __init__(self, problem: Problem, seed: int | None = None):
        """
        Initialize the Optimizer.

        Parameters
        ----------
        problem : Problem
            The optimization problem.
        seed : int or None, optional
            Master random seed propagated to the initializer.  ``None`` (default)
            means non-deterministic.
        """
        self.problem: Problem = problem
        self.seed: int | None = seed
        self.cbmanager: CallbackManager = CallbackManager()
        self.cbmanager.register(GenerationStartEvent, logging_generation)
        self.initializer: Initializer | None = None
        self.evaluator: Evaluator = SerialEvaluator()
        self.instance_name: str = ""

    # --- setters (all return self for chaining) ---

    def set_seed(self, seed: int | None) -> Self:
        """Set the master random seed. Returns self."""
        self.seed = seed
        return self

    def set_initializer(self, initializer: Initializer) -> Self:
        """Set the initializer. Returns self."""
        self.initializer = initializer
        return self

    def set_algorithm(self, algorithm: Algorithm) -> Self:
        """Set the evolutionary algorithm. Returns self."""
        self.algorithm = algorithm
        return self

    def set_surrogate_manager(self, manager: SurrogateManager) -> Self:
        """Set the surrogate manager. Returns self."""
        self.surrogate_manager = manager
        return self

    def set_surrogate(self, surrogate: Surrogate, n_neighbors: int = 50) -> Self:
        """
        Wrap a raw Surrogate in a LocalSurrogateManager. Returns self.

        Equivalent to ``set_surrogate_manager(LocalSurrogateManager(surrogate, ...))``.
        Provided for backward compatibility.
        """
        from saealib.surrogate.training_set import KNNObjectiveSet

        self.surrogate_manager = LocalSurrogateManager(
            surrogate,
            MeanPrediction(direction=self.problem.direction),
            training_set=KNNObjectiveSet(n_neighbors=n_neighbors),
        )
        return self

    def set_strategy(self, strategy: OptimizationStrategy) -> Self:
        """Set the optimization strategy. Returns self."""
        self.strategy = strategy
        return self

    def set_evaluator(self, evaluator: Evaluator) -> Self:
        """Set the evaluator. Returns self."""
        self.evaluator = evaluator
        return self

    def set_termination(self, termination: Termination) -> Self:
        """Set the termination condition. Returns self."""
        self.termination = termination
        return self

    def set_instance_name(self, name: str) -> Self:
        """Set the instance name. Returns self."""
        self.instance_name = name
        return self

    # --- callbacks ---

    def dispatch(self, event: Event) -> None:
        """Dispatch an event to the callback manager."""
        self.cbmanager.dispatch(event)

    # --- run ---

    def validate(self, *, require_initializer: bool = True) -> list[str]:
        """
        Check configuration consistency. Returns list of issues.

        Parameters
        ----------
        require_initializer : bool, optional
            When False, skip the initializer presence check.  Use this when
            resuming from a checkpoint where initialization is not needed.
        """
        issues: list[str] = []

        algorithm = getattr(self, "algorithm", None)
        strategy = getattr(self, "strategy", None)
        termination = getattr(self, "termination", None)
        surrogate_manager = getattr(self, "surrogate_manager", None)

        if algorithm is None:
            issues.append("algorithm is not set; call set_algorithm()")
        if strategy is None:
            issues.append("strategy is not set; call set_strategy()")
        if require_initializer and self.initializer is None:
            issues.append("initializer is not set; call set_initializer()")
        if termination is None:
            issues.append("termination is not set; call set_termination()")

        if (
            strategy is not None
            and getattr(strategy, "requires_surrogate", False)
            and surrogate_manager is None
        ):
            issues.append(
                f"{type(strategy).__name__} requires a surrogate_manager; "
                "call set_surrogate_manager() or set_surrogate()"
            )

        _dim = getattr(self.problem.comparator, "direction", None)
        if (
            _dim is not None
            and hasattr(_dim, "__len__")
            and len(_dim) > 0
            and len(_dim) != self.problem.n_obj
        ):
            issues.append(
                f"comparator direction length ({len(_dim)}) does not match "
                f"problem.n_obj ({self.problem.n_obj})"
            )

        if surrogate_manager is not None:
            acq = getattr(surrogate_manager, "acquisition", None)
            surr = getattr(surrogate_manager, "surrogate", None)
            if (
                acq is not None
                and surr is not None
                and getattr(acq, "requires_uncertainty", False)
                and not getattr(surr, "provides_uncertainty", False)
            ):
                issues.append(
                    f"{type(acq).__name__} requires uncertainty estimates but "
                    f"{type(surr).__name__} does not provide them "
                    "(provides_uncertainty=False)"
                )

        return issues

    def _register_checkpoint(
        self,
        path: str | Path,
        interval: int,
        format: str,
        delete_on_success: bool,
    ) -> None:
        from saealib.checkpoint import CheckpointCallback

        cb = CheckpointCallback(
            path=path,
            interval=interval,
            format=format,
            delete_on_success=delete_on_success,
            optimizer=self if format in ("pickle", "both") else None,
        )
        cb.register(self.cbmanager)

    def iterate(
        self,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 1,
        checkpoint_format: str = "npz",
        checkpoint_delete_on_success: bool = False,
    ) -> Generator[OptimizationState, None, None]:
        """
        Iterate the optimization process.

        Parameters
        ----------
        checkpoint_path : str, Path, or None, optional
            If provided, checkpoints are saved to this directory every
            *checkpoint_interval* generations.
        checkpoint_interval : int, optional
            Generations between checkpoints.  Default: 1.
        checkpoint_format : {'npz', 'pickle', 'both'}, optional
            Checkpoint format.  Default: ``'npz'``.
        checkpoint_delete_on_success : bool, optional
            Delete checkpoints on normal termination.  Default: False.

        Returns
        -------
        Generator[OptimizationState]
            Generator of OptimizationState.
        """
        issues = self.validate()
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        if checkpoint_path is not None:
            self._register_checkpoint(
                checkpoint_path,
                checkpoint_interval,
                checkpoint_format,
                checkpoint_delete_on_success,
            )
        return Runner(self).iterate()

    def run(
        self,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 1,
        checkpoint_format: str = "npz",
        checkpoint_delete_on_success: bool = False,
    ) -> OptimizationState:
        """
        Run the optimization process.

        Parameters
        ----------
        checkpoint_path : str, Path, or None, optional
            If provided, checkpoints are saved to this directory every
            *checkpoint_interval* generations.
        checkpoint_interval : int, optional
            Generations between checkpoints.  Default: 1.
        checkpoint_format : {'npz', 'pickle', 'both'}, optional
            Checkpoint format.  Default: ``'npz'``.
        checkpoint_delete_on_success : bool, optional
            Delete checkpoints on normal termination.  Default: False.

        Returns
        -------
        OptimizationState
            The optimization context.
        """
        issues = self.validate()
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        if checkpoint_path is not None:
            self._register_checkpoint(
                checkpoint_path,
                checkpoint_interval,
                checkpoint_format,
                checkpoint_delete_on_success,
            )
        return Runner(self).run()

    def iterate_from(
        self, ctx: OptimizationState
    ) -> Generator[OptimizationState, None, None]:
        """
        Resume iteration from an existing context (e.g. loaded from checkpoint).

        Does not call the initializer; all other components must be configured.

        Parameters
        ----------
        ctx : OptimizationState
            Context to resume from.

        Returns
        -------
        Generator[OptimizationState, None, None]
        """
        issues = self.validate(require_initializer=False)
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        return Runner(self).iterate_from(ctx)

    def run_from(self, ctx: OptimizationState) -> OptimizationState:
        """
        Resume and run to completion from an existing context.

        Parameters
        ----------
        ctx : OptimizationState
            Context to resume from.

        Returns
        -------
        OptimizationState
            The final optimization context.
        """
        issues = self.validate(require_initializer=False)
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        return Runner(self).run_from(ctx)

    # ------------------------------------------------------------------
    # Checkpoint: pickle (limited complete reproducibility)
    # ------------------------------------------------------------------

    _PICKLE_WARNING = (
        "Pickle checkpoints are version-sensitive. "
        "Reproducibility is only guaranteed within the same Python "
        "and library versions."
    )

    def save_pickle(self, ctx: OptimizationState, path: str | Path) -> None:
        """
        Save the optimizer and context together as a pickle checkpoint.

        This preserves fitted surrogate state and all component objects,
        offering complete reproducibility within the same environment.

        .. warning::
            Pickle files are tied to specific Python and library versions.
            Use :meth:`OptimizationState.save` for a more portable format.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        path : str or Path
            Destination file path.  The ``.pkl`` extension is added if absent.
        """
        warnings.warn(self._PICKLE_WARNING, UserWarning, stacklevel=2)
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".pkl")
        with open(p, "wb") as f:
            pickle.dump((self, ctx), f)

    @classmethod
    def load_pickle(cls, path: str | Path) -> tuple[Optimizer, OptimizationState]:
        """
        Load an optimizer and context from a pickle checkpoint.

        The returned context has ``resumed=True``.  Call
        :meth:`run_from` or :meth:`iterate_from` on the returned optimizer
        to continue the optimization.

        .. warning::
            Pickle files are tied to specific Python and library versions.

        Parameters
        ----------
        path : str or Path
            Path to the ``.pkl`` file.  The extension is added if absent.

        Returns
        -------
        tuple[Optimizer, OptimizationState]
        """
        warnings.warn(cls._PICKLE_WARNING, UserWarning, stacklevel=2)
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".pkl")
        with open(p, "rb") as f:
            optimizer, ctx = pickle.load(f)
        ctx = dataclasses.replace(ctx, data={**ctx.data, "resumed": True})
        return optimizer, ctx
