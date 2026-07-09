"""Optimizer: assembles and runs the surrogate-assisted EA pipeline."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
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
from saealib.exceptions import ConfigurationError, ValidationError
from saealib.execution.evaluator import Evaluator, SerialEvaluator
from saealib.execution.runner import Runner
from saealib.surrogate.manager import LocalSurrogateManager, SurrogateManager
from saealib.surrogate.rbf import gaussian_kernel
from saealib.termination import Termination
from saealib.termination import max_fe as max_fe_cond

if TYPE_CHECKING:
    from saealib.algorithms.base import Algorithm
    from saealib.execution.initializer import Initializer
    from saealib.problem import Problem
    from saealib.strategies.base import OptimizationStrategy
    from saealib.surrogate.base import Surrogate


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
        self._preset: dict | None = None

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

    def set_preset(self, preset: str | Path | dict) -> Self:
        """
        Set a user-defined preset. Returns self.

        The preset is only used to fill components not already configured
        via ``set_*()`` (see ``_resolve_defaults()``); explicitly set
        components always take precedence over the preset.
        """
        from saealib.defaults import load_preset

        self._preset = load_preset(preset)
        return self

    # --- preset export ---

    def save_preset(self, path: str | Path) -> Path:
        """
        Save the currently configured components as a reusable preset file.

        Serializes ``algorithm``, ``surrogate_manager``, ``strategy``, and
        ``termination`` (whichever are set via ``set_*()``) into a preset
        dict and writes it as YAML. Problem-owned parameters (``dim``,
        ``direction``) are stripped so the preset can be reused across
        problems of different dimensionality.

        Parameters
        ----------
        path : str or Path
            Destination file path. The ``.yaml`` extension is added if absent.

        Returns
        -------
        Path
            The path the preset was written to.

        Raises
        ------
        ValidationError
            If no components are configured, or a configured component
            cannot be serialized (e.g. holds a raw lambda).
        """
        from saealib.defaults import dump_preset
        from saealib.registry import _strip_params, to_spec

        preset: dict = {}
        for name in ("algorithm", "surrogate_manager", "strategy", "termination"):
            component = getattr(self, name, None)
            if component is None:
                continue
            try:
                spec = to_spec(component)
            except ValidationError as e:
                raise ValidationError(
                    f"Cannot save preset: {name} is not serializable: {e}"
                ) from e
            preset[name] = _strip_params(spec, "dim", "direction")

        if not preset:
            raise ValidationError(
                "Cannot save preset: no components are configured. Call "
                "set_algorithm()/set_strategy()/set_surrogate_manager()/"
                "set_termination() first."
            )
        return dump_preset(preset, path)

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

            for acq in surrogate_manager.iter_acquisitions():
                _adir = getattr(acq, "direction", None)
                if (
                    _adir is not None
                    and hasattr(_adir, "__len__")
                    and len(_adir) > 0
                    and len(_adir) != self.problem.n_obj
                ):
                    issues.append(
                        f"{type(acq).__name__} direction length ({len(_adir)}) does "
                        f"not match problem.n_obj ({self.problem.n_obj})"
                    )

        return issues

    def _resolve_defaults(self) -> None:
        """Fill unset components with library defaults (Registry + presets file).

        Components already set via ``set_*()`` are never overwritten. Gaps
        are filled first from a user preset (``set_preset()``), if any, then
        from a bundled preset selected by (1) the algorithm's registered
        name if ``algorithm`` is set, else (2) a Problem-shape rule, else
        (3) the universal fallback preset. ``initializer`` is computed
        directly from ``problem.dim`` and is not part of any preset;
        ``termination`` falls back to ``200 * problem.dim`` function
        evaluations only if neither ``set_*()`` nor a preset supplies one.
        """
        from saealib.defaults import load_defaults
        from saealib.registry import _inject_params, build

        algorithm = getattr(self, "algorithm", None)
        strategy = getattr(self, "strategy", None)
        surrogate_manager = getattr(self, "surrogate_manager", None)

        user_preset = getattr(self, "_preset", None)
        if user_preset is not None:
            dim = self.problem.dim
            direction = self.problem.direction
            if algorithm is None and "algorithm" in user_preset:
                algorithm = build(
                    _inject_params(
                        user_preset["algorithm"], dim=dim, direction=direction
                    )
                )
                self.algorithm = algorithm
            if surrogate_manager is None and "surrogate_manager" in user_preset:
                surrogate_manager = self._build_surrogate_manager(
                    user_preset["surrogate_manager"]
                )
                self.surrogate_manager = surrogate_manager
            if strategy is None and "strategy" in user_preset:
                strategy = build(
                    _inject_params(
                        user_preset["strategy"], dim=dim, direction=direction
                    )
                )
                self.strategy = strategy
            if (
                getattr(self, "termination", None) is None
                and "termination" in user_preset
            ):
                self.termination = build(
                    _inject_params(
                        user_preset["termination"], dim=dim, direction=direction
                    )
                )

        if algorithm is None or strategy is None or surrogate_manager is None:
            defaults = load_defaults()
            preset = defaults["presets"][self._select_preset_name(defaults, algorithm)]
            if algorithm is None and "algorithm" in preset:
                self.algorithm = build(preset["algorithm"])
            if surrogate_manager is None and "surrogate_manager" in preset:
                self.surrogate_manager = self._build_surrogate_manager(
                    preset["surrogate_manager"]
                )
            if strategy is None and "strategy" in preset:
                self.strategy = build(preset["strategy"])

        if self.initializer is None:
            from saealib.execution.initializer import LHSInitializer

            dim = self.problem.dim
            self.initializer = LHSInitializer(
                n_init_archive=5 * dim, n_init_population=4 * dim, seed=self.seed
            )

        if getattr(self, "termination", None) is None:
            self.termination = Termination(max_fe_cond(200 * self.problem.dim))

    def _inject_acquisition_directions(self) -> None:
        """Auto-inject ``problem.direction`` into unset acquisition directions.

        Mirrors the "inherit from problem unless explicitly set" contract used
        for ``NSGA3Comparator.rng`` and ``SingleObjectiveComparator.direction``:
        an acquisition function that already has an explicit ``direction`` (or
        opts out via ``direction_sensitive = False``) is left untouched.
        """
        surrogate_manager = getattr(self, "surrogate_manager", None)
        if surrogate_manager is None:
            return
        for acq in surrogate_manager.iter_acquisitions():
            if (
                getattr(acq, "direction_sensitive", True)
                and getattr(acq, "direction", None) is None
            ):
                acq.direction = self.problem.direction

    def _select_preset_name(self, defaults: dict, algorithm: Algorithm | None) -> str:
        if algorithm is not None:
            preset_name = defaults["by_algorithm"].get(type(algorithm).__name__)
            if preset_name is not None:
                return preset_name
        for rule in defaults["by_problem_shape"]:
            when = rule["when"]
            if all(
                getattr(self.problem, key, None) == value for key, value in when.items()
            ):
                return rule["preset"]
        return defaults["fallback"]

    def _build_surrogate_manager(self, spec: dict) -> SurrogateManager:
        return self._build_surrogate_manager_from_spec(
            spec, self.problem.dim, self.problem.direction
        )

    @staticmethod
    def _build_surrogate_manager_from_spec(
        spec: dict, dim: int, direction
    ) -> SurrogateManager:
        """Build a surrogate_manager preset spec, injecting dim/direction defaults.

        Shared by ``Optimizer._resolve_defaults()`` and ``saealib.api``'s
        ``'rbf'`` surrogate shorthand, so the injection logic is defined once.
        """
        from saealib.registry import _inject_params, build

        spec = copy.deepcopy(spec)
        params = spec.setdefault("params", {})
        params.setdefault(
            "surrogate",
            {"type": "RBFSurrogate", "params": {"kernel": gaussian_kernel, "dim": dim}},
        )
        params.setdefault(
            "acquisition",
            {"type": "MeanPrediction", "params": {"direction": direction}},
        )
        spec = _inject_params(spec, dim=dim, direction=direction)
        return build(spec)

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
        self._resolve_defaults()
        issues = self.validate()
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        self._inject_acquisition_directions()
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
        self._resolve_defaults()
        issues = self.validate()
        if issues:
            raise ConfigurationError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        self._inject_acquisition_directions()
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
        self._inject_acquisition_directions()
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
        self._inject_acquisition_directions()
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

    # ------------------------------------------------------------------
    # Construction: from a problem definition file
    # ------------------------------------------------------------------

    @classmethod
    def from_problem_file(
        cls,
        problem_path: str | Path,
        preset: str | Path | dict | None = None,
    ) -> Optimizer:
        """
        Build an Optimizer from a Python file defining a problem.

        The file at *problem_path* is executed as a standalone module (it is not
        registered in ``sys.modules``) and must define a top-level ``problem``
        variable holding a :class:`~saealib.problem.Problem` instance. If
        *preset* is given, it is applied first via :meth:`set_preset`; any of
        the top-level variables ``algorithm``, ``strategy``,
        ``surrogate_manager``, ``termination`` defined in the file are then
        applied via the corresponding ``set_*()`` method, so the file can
        override individual components of the preset. A top-level ``seed``
        variable, if present, is passed to the ``Optimizer`` constructor.

        Parameters
        ----------
        problem_path : str or Path
            Path to a ``.py`` file defining a top-level ``problem`` variable.
        preset : str, Path, dict, or None, optional
            Preset applied before the file's own component definitions, so
            that the file can override individual components. See
            :meth:`set_preset`.

        Returns
        -------
        Optimizer
            A configured, not-yet-run ``Optimizer`` instance. Call ``run()``
            or ``iterate()`` on it.

        Raises
        ------
        ValidationError
            If the file does not define a top-level ``problem`` variable, or
            it is not a ``Problem`` instance. Also raised if a top-level
            ``algorithm``, ``strategy``, ``surrogate_manager``,
            ``termination``, or ``seed`` variable is defined but is not an
            instance of its expected type.
        FileNotFoundError
            If *problem_path* does not exist.

        Examples
        --------
        >>> # problem.py:
        >>> #     problem = Problem(func=..., dim=2, n_obj=1, direction=[-1])
        >>> #     seed = 42
        >>> opt = Optimizer.from_problem_file("problem.py", preset="preset.yaml")
        >>> ctx = opt.run()  # doctest: +SKIP
        """
        from saealib.algorithms.base import Algorithm
        from saealib.problem import Problem
        from saealib.strategies.base import OptimizationStrategy
        from saealib.surrogate.manager import SurrogateManager
        from saealib.termination import Termination

        p = Path(problem_path)
        spec = importlib.util.spec_from_file_location(p.stem, p)
        if spec is None or spec.loader is None:
            raise ValidationError(f"Cannot load module from {p}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        problem = getattr(module, "problem", None)
        if not isinstance(problem, Problem):
            raise ValidationError(
                f"{p} must define a top-level `problem` variable holding a "
                "Problem instance"
            )

        seed = getattr(module, "seed", None)
        if seed is not None and not isinstance(seed, int):
            raise ValidationError(
                f"{p}: top-level 'seed' must be an int instance, "
                f"got {type(seed).__name__}"
            )
        opt = cls(problem, seed=seed) if seed is not None else cls(problem)

        if preset is not None:
            opt.set_preset(preset)

        for name, setter, expected in (
            ("algorithm", opt.set_algorithm, Algorithm),
            ("strategy", opt.set_strategy, OptimizationStrategy),
            ("surrogate_manager", opt.set_surrogate_manager, SurrogateManager),
            ("termination", opt.set_termination, Termination),
        ):
            component = getattr(module, name, None)
            if component is not None:
                if not isinstance(component, expected):
                    raise ValidationError(
                        f"{p}: top-level '{name}' must be a {expected.__name__} "
                        f"instance, got {type(component).__name__}"
                    )
                setter(component)

        return opt
