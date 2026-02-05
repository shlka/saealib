"""
Optimizer module.

Optimizer class that integrates components to perform
evolutionary optimization with surrogate models.
Optimizer class that integrates components to perform
evolutionary optimization with surrogate models.
"""


from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import scipy.stats

from saealib.callback import CallbackEvent, CallbackManager, logging_generation
from saealib.context import OptimizationContext
from saealib.operators.repair import repair_clipping
from saealib.population import Archive, Population, PopulationAttribute
from saealib.population import Archive, Population, PopulationAttribute

if TYPE_CHECKING:
    from saealib.algorithm import Algorithm
    from saealib.modelmanager import ModelManager
    from saealib.problem import Problem
    from saealib.surrogate.base import Surrogate
    from saealib.problem import Problem
    from saealib.surrogate.base import Surrogate
    from saealib.termination import Termination


class ComponentProvider(Protocol):
    """The interface for components that can be used by the Optimizer."""

    @property
    def algorithm(self) -> Algorithm:
        """Return the algorithm instance."""
        ...

    @property
    def modelmanager(self) -> ModelManager:
        """Return the model manager instance."""
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
    modelmanager : ModelManager
        The surrogate model manager.
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
        # components
        self.problem = problem
        self.algorithm = None
        self.surrogate = None
        self.modelmanager = None
        self.termination = None
        # Archive init parameters
        self.archive_atol = 0.0
        self.archive_rtol = 0.0
        self.archive = None
        self.archive_init_size = 50
        # random setup
        self.seed = 0
        # EA parameters
        self.popsize = 40
        # callback event manager
        self.cbmanager = CallbackManager()
        # instance name
        self.instance_name = ""

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

    def set_modelmanager(self, modelmanager: ModelManager) -> Optimizer:
        """
        Set ModelManager instance.

        Parameters
        ----------
        modelmanager : ModelManager
            ModelManager instance.

        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.modelmanager = modelmanager
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

    def set_archive_init_size(self, size: int) -> Optimizer:
        """
        Set Archive initial size.

        Parameters
        ----------
        size : int
            Initial size of the archive.


        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.archive_init_size = size
        return self

    def set_archive_atol(self, atol: float) -> Optimizer:
        """
        Set Archive absolute tolerance to remove duplicates.

        Parameters
        ----------
        atol : float
            Absolute tolerance for the archive.


        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.archive_atol = atol
        return self


    def set_archive_rtol(self, rtol: float) -> Optimizer:
        """
        Set Archive relative tolerance to remove duplicates.

        Parameters
        ----------
        rtol : float
            Relative tolerance for the archive.


        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.archive_rtol = rtol
        return self

    def set_seed(self, seed: int) -> Optimizer:
        """
        Set random seed and initialize numpy.random.Generator.

        Parameters
        ----------
        seed : int
            Random seed.


        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.seed = seed
        return self

    def set_popsize(self, popsize: int) -> Optimizer:
        """
        Set population size.

        Parameters
        ----------
        popsize : int
            Population size.


        Returns
        -------
        Optimizer
            Returns self for method chaining.
        """
        self.popsize = popsize
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

    def create_context(self) -> OptimizationContext:
        """Create the OptimizationContext and initialize the Optimizer."""
        rng = np.random.default_rng(self.seed)

        # TODO:　modify it to account for the fact that there are n_obj instances of f.
        # default attributes
        attrs = [
            PopulationAttribute("x", float, (self.problem.dim,), default=np.nan),
            # PopulationAttribute("f",  float, (self.problem.n_obj, ), default=np.nan)
            PopulationAttribute("f", float, (), default=np.nan),
            PopulationAttribute("g", float, (self.problem.n_constraint,), default=0.0),
            PopulationAttribute("cv", float, (), default=0.0),
        ]

        # Retrieve attributes and classes according to the algorithm
        if self.algorithm is not None:
            attrs_required = self.algorithm.get_required_attrs(self.problem)
            ex_names = {attr.name for attr in attrs}
            for attr in attrs_required:
                if attr.name not in ex_names:
                    attrs.append(attr)

            pop_class = self.algorithm.population_class
            arc_class = self.algorithm.archive_class
        else:
            # TODO: Consider whether to make it an exception
            pop_class = Population
            arc_class = Archive

        population = pop_class(attrs=attrs, init_capacity=self.popsize)
        archive = arc_class(attrs=attrs, init_capacity=self.archive_init_size)

        archive_x = scipy.stats.qmc.LatinHypercube(d=self.problem.dim, rng=rng).random(
            self.archive_init_size
        )
        archive_x = scipy.stats.qmc.scale(archive_x, self.problem.lb, self.problem.ub)
        archive_f = np.array([self.problem.evaluate(ind) for ind in archive_x])

        # TODO: use cv if constraints are defined
        archive_sort_idx = self.problem.comparator.sort(
            archive_f, np.zeros_like(archive_f)
        )
        archive_x = archive_x[archive_sort_idx]
        archive_f = archive_f[archive_sort_idx]

        for i in range(self.archive_init_size):
            archive.add({"x": archive_x[i], "f": archive_f[i]})

        population.extend(archive[: self.popsize])

        # initialize callbacks
        self.cbmanager.register(CallbackEvent.GENERATION_START, logging_generation)
        self.cbmanager.register(CallbackEvent.POST_CROSSOVER, repair_clipping)
        self.cbmanager.register(CallbackEvent.POST_MUTATION, repair_clipping)

        ctx = OptimizationContext(
            problem=self.problem,
            population=population,
            archive=archive,
            rng=rng,
            fe=self.archive_init_size,
            gen=0,
        )
        return ctx

    def run(self) -> None:
        """
        Run the optimization process.

        Returns
        -------
        None
        """
        ctx = self.create_context()
        self.dispatch(CallbackEvent.RUN_START, ctx=ctx)

        while not self.termination.is_terminated(fe=ctx.fe):
            ctx.count_generation()

            self.dispatch(CallbackEvent.GENERATION_START, ctx=ctx)

            # ask
            cand = self.algorithm.ask(ctx, self)

            self.dispatch(CallbackEvent.SURROGATE_START, ctx=ctx)

            # surrogate
            # TODO: use Population container
            cand_, cand_fit = self.modelmanager.run(ctx, self, cand.get_array("x"))
            cand.clear()
            cand.extend({"x": cand_, "f": cand_fit})

            self.dispatch(CallbackEvent.SURROGATE_END, ctx=ctx)

            # tell
            self.algorithm.tell(ctx, self, cand)

            self.dispatch(CallbackEvent.GENERATION_END, ctx=ctx)

        self.dispatch(CallbackEvent.RUN_END, ctx=ctx)
