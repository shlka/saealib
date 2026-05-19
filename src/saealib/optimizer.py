"""Optimizer: assembles and runs the surrogate-assisted EA pipeline."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Protocol

from saealib.acquisition.mean import MeanPrediction
from saealib.callback import (
    CallbackManager,
    Event,
    GenerationStartEvent,
    logging_generation,
)
from saealib.context import OptimizationContext
from saealib.execution.runner import Runner
from saealib.surrogate.manager import LocalSurrogateManager, SurrogateManager

if TYPE_CHECKING:
    from saealib.algorithms.base import Algorithm
    from saealib.execution.initializer import Initializer
    from saealib.problem import Problem
    from saealib.strategies.base import OptimizationStrategy
    from saealib.surrogate.base import Surrogate
    from saealib.termination import Termination


class ComponentProvider(Protocol):
    """The interface for components that can be used by the Optimizer."""

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
    seed : int
        The random seed.
    rng : numpy.random.Generator
        The random number generator.
    fe : int
        The current number of function evaluations.
    gen : int
        The current generation number.
    cbmanager : CallbackManager
        The callback event manager.
    instance_name : str
        The name of the optimizer instance.
    """

    def __init__(self, problem: Problem):
        """
        Initialize the Optimizer.

        Parameters
        ----------
        problem : Problem
            The optimization problem.
        """
        self.problem: Problem = problem
        self.cbmanager: CallbackManager = CallbackManager()
        self.cbmanager.register(GenerationStartEvent, logging_generation)
        self.initializer: Initializer | None = None
        self.instance_name: str = ""

    # --- setters (all return self for chaining) ---

    def set_initializer(self, initializer: Initializer) -> Optimizer:
        """Set the initializer. Returns self."""
        self.initializer = initializer
        return self

    def set_algorithm(self, algorithm: Algorithm) -> Optimizer:
        """Set the evolutionary algorithm. Returns self."""
        self.algorithm = algorithm
        return self

    def set_surrogate_manager(self, manager: SurrogateManager) -> Optimizer:
        """Set the surrogate manager. Returns self."""
        self.surrogate_manager = manager
        return self

    def set_surrogate(self, surrogate: Surrogate, n_neighbors: int = 50) -> Optimizer:
        """
        Wrap a raw Surrogate in a LocalSurrogateManager. Returns self.

        Equivalent to ``set_surrogate_manager(LocalSurrogateManager(surrogate, ...))``.
        Provided for backward compatibility.
        """
        from saealib.surrogate.training_set import KNNObjectiveSet

        self.surrogate_manager = LocalSurrogateManager(
            surrogate,
            MeanPrediction(weights=self.problem.weight),
            training_set=KNNObjectiveSet(n_neighbors=n_neighbors),
        )
        return self

    def set_strategy(self, strategy: OptimizationStrategy) -> Optimizer:
        """Set the optimization strategy. Returns self."""
        self.strategy = strategy
        return self

    def set_termination(self, termination: Termination) -> Optimizer:
        """Set the termination condition. Returns self."""
        self.termination = termination
        return self

    def set_instance_name(self, name: str) -> Optimizer:
        """Set the instance name. Returns self."""
        self.instance_name = name
        return self

    # --- callbacks ---

    def dispatch(self, event: Event) -> None:
        """Dispatch an event to the callback manager."""
        self.cbmanager.dispatch(event)

    # --- run ---

    def validate(self) -> list[str]:
        """Check configuration consistency. Returns list of issues."""
        issues: list[str] = []

        algorithm         = getattr(self, "algorithm", None)
        strategy          = getattr(self, "strategy", None)
        termination       = getattr(self, "termination", None)
        surrogate_manager = getattr(self, "surrogate_manager", None)

        if algorithm is None:
            issues.append("algorithm is not set; call set_algorithm()")
        if strategy is None:
            issues.append("strategy is not set; call set_strategy()")
        if self.initializer is None:
            issues.append("initializer is not set; call set_initializer()")
        if termination is None:
            issues.append("termination is not set; call set_termination()")

        if strategy is not None and getattr(strategy, "requires_surrogate", False):
            if surrogate_manager is None:
                issues.append(
                    f"{type(strategy).__name__} requires a surrogate_manager; "
                    "call set_surrogate_manager() or set_surrogate()"
                )

        weights = getattr(self.problem.comparator, "weights", None)
        if weights is not None and len(weights) != self.problem.n_obj:
            issues.append(
                f"comparator.weights length ({len(weights)}) does not match "
                f"problem.n_obj ({self.problem.n_obj})"
            )

        if surrogate_manager is not None:
            acq  = getattr(surrogate_manager, "acquisition", None)
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

    def iterate(self) -> Generator[OptimizationContext, None, None]:
        """
        Iterate the optimization process.

        Returns
        -------
        Generator[OptimizationContext]
            Generator of OptimizationContext.
        """
        issues = self.validate()
        if issues:
            raise ValueError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        return Runner(self).iterate()

    def run(self) -> OptimizationContext:
        """
        Run the optimization process.

        Returns
        -------
        OptimizationContext
            The optimization context.
        """
        issues = self.validate()
        if issues:
            raise ValueError(
                "Optimizer misconfigured:\n" + "\n".join(f"  - {m}" for m in issues)
            )
        return Runner(self).run()
