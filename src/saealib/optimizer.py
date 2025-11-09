from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.stats

from saealib.population import Population, Archive
from saealib.callback import CallbackEvent, CallbackManager, logging_generation
from saealib.operators.repair import repair_clipping

if TYPE_CHECKING:
    from saealib.algorithm import Algorithm
    from saealib.surrogate.base import Surrogate
    from saealib.modelmanager import ModelManager
    from saealib.termination import Termination
    from saealib.problem import Problem


class Optimizer:
    """
    Optimizer class for evolutionary algorithms.

    Integrates problem definition, evolutionary algorithm, surrogate model, model manager, and termination condition, 
    and manages the optimization process.

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
        self.rng = np.random.default_rng(seed=self.seed)
        # state variables
        self.fe = 0
        self.gen = 0
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
        self.rng = np.random.default_rng(seed=self.seed)
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

    def _initialize(self, n_init_archive: int) -> None:
        """
        Before running the optimization, initialize the archive and population.
        """
        archive_x = scipy.stats.qmc.LatinHypercube(d=self.problem.dim, rng=self.rng).random(n_init_archive)
        archive_x = scipy.stats.qmc.scale(archive_x, self.problem.lb, self.problem.ub)
        archive_y = np.array([self.problem.evaluate(ind) for ind in archive_x])
        # TODO: use cv if constraints are defined
        archive_sort_idx = self.problem.comparator.sort(archive_y, np.zeros_like(archive_y))
        archive_x = archive_x[archive_sort_idx]
        archive_y = archive_y[archive_sort_idx]
        self.archive = Archive.new(archive_x, archive_y, atol=self.archive_atol, rtol=self.archive_rtol)

        self.population = Population.new("x", self.archive.get("x")[:self.popsize])
        self.population.set("f", self.archive.get("y")[:self.popsize])

        self.fe = self.archive_init_size
        self.gen = 0

        self.cbmanager.register(CallbackEvent.GENERATION_START, logging_generation)
        self.cbmanager.register(CallbackEvent.POST_CROSSOVER, repair_clipping)
        self.cbmanager.register(CallbackEvent.POST_MUTATION, repair_clipping)

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
        kwargs["optimizer"] = self
        return self.cbmanager.dispatch(event, data, **kwargs)

    def run(self) -> None:
        """
        Run the optimization process.

        Returns
        -------
        None
        """
        self._initialize(self.archive_init_size)
        self.dispatch(CallbackEvent.RUN_START)

        while not self.termination.is_terminated(fe=self.fe):

            self.gen += 1

            self.dispatch(CallbackEvent.GENERATION_START)

            # ask
            cand = self.algorithm.ask(self)

            self.dispatch(CallbackEvent.SURROGATE_START)

            # surrogate
            cand, cand_fit = self.modelmanager.run(self, cand)

            self.dispatch(CallbackEvent.SURROGATE_END)

            # tell
            self.algorithm.tell(self, cand, cand_fit)

            self.dispatch(CallbackEvent.GENERATION_END)

        self.dispatch(CallbackEvent.RUN_END)
