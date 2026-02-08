"""
Optimizer module.

Optimizer class that integrates components to perform
evolutionary optimization with surrogate models.

"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Protocol

from saealib.callback import CallbackEvent, CallbackManager, logging_generation
from saealib.context import OptimizationContext
from saealib.execution.runner import Runner
from saealib.operators.repair import repair_clipping

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
    def surrogate(self) -> Surrogate:
        """Return the surrogate model instance."""
        ...

    @property
    def termination(self) -> Termination:
        """Return the termination condition."""
        ...

    @property
    def cbmanager(self) -> CallbackManager:
        """Return the callback manager."""
        ...

    def dispatch(self, event: CallbackEvent, data=None, **kwargs) -> Any:
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
        # Problem instance
        self.problem: Problem = problem
        # callback event manager
        self.cbmanager: CallbackManager = CallbackManager()
        # initialize callbacks (default)
        self.cbmanager.register(CallbackEvent.GENERATION_START, logging_generation)
        self.cbmanager.register(CallbackEvent.POST_CROSSOVER, repair_clipping)
        self.cbmanager.register(CallbackEvent.POST_MUTATION, repair_clipping)
        # initializer instance
        self.initializer: Initializer = None
        # Optimizer instance name
        self.instance_name: str = ""

    ### SETTERS ###
    def set_initializer(self, initializer: Initializer) -> Optimizer:
        """
        Set Initializer instance.

        Parameters
        ----------
        initializer : Initializer
            Initializer instance.

        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.initializer = initializer
        return self

    def set_algorithm(self, algorithm: Algorithm) -> Optimizer:
        """
        Set Algorithm instance.

        Parameters
        ----------
        algorithm : Algorithm
            Algorithm instance.

        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.algorithm = algorithm
        return self

    def set_surrogate(self, surrogate: Surrogate) -> Optimizer:
        """
        Set Surrogate instance.

        Parameters
        ----------
        surrogate : Surrogate
            Surrogate instance.

        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.surrogate = surrogate
        return self

    def set_strategy(self, strategy: OptimizationStrategy) -> Optimizer:
        """
        Set OptimizationStrategy instance.

        Parameters
        ----------
        strategy : OptimizationStrategy
            OptimizationStrategy instance.

        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.strategy = strategy
        return self

    def set_termination(self, termination: Termination) -> Optimizer:
        """
        Set Termination instance.

        Parameters
        ----------
        termination : Termination
            Termination instance.

        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.termination = termination
        return self

    def set_instance_name(self, name: str) -> Optimizer:
        """
        Set instance name.

        Parameters
        ----------
        name : str
            Instance name.

        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.instance_name = name
        return self

    ### CALLBACKS ###
    def dispatch(self, event: CallbackEvent, data=None, **kwargs) -> any:
        """
        Dispatch an event to the callback manager.

        Parameters
        ----------
        event : CallbackEvent
            The callback event to dispatch.
        data : any, optional
            Data that may be rewritten.
        kwargs : dict, optional
            Additional keyword arguments for the callback.

        Returns
        -------
        any
            The data returned by another callback.
        """
        kwargs["provider"] = self
        return self.cbmanager.dispatch(event, data, **kwargs)

    ### RUN ###
    def iterate(self) -> Generator[OptimizationContext, None, None]:
        """
        Iterate the optimization process.

        Returns
        -------
        Generator[OptimizationContext]
            Generator of OptimizationContext.
        """
        return Runner(self).iterate()

    def run(self) -> OptimizationContext:
        """
        Run the optimization process.

        Returns
        -------
        OptimizationContext
            The optimization context.
        """
        return Runner(self).run()
