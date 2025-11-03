from typing import TYPE_CHECKING

import numpy as np
import scipy.stats

from .population import Population, Archive
from .callback import CallbackEvent, CallbackManager, logging_generation
from .operators.repair import repair_clipping

if TYPE_CHECKING:
    from .algorithm import Algorithm
    from .surrogate.base import Surrogate
    from .modelmanager import ModelManager
    from .termination import Termination
    from .problem import Problem


class Optimizer:
    """
    Base class for optimizers.
    """
    def __init__(self, problem: Problem):
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

    def set_algorithm(self, algorithm: Algorithm):
        self.algorithm = algorithm
        return self
    
    def set_surrogate(self, surrogate: Surrogate):
        self.surrogate = surrogate
        return self
    
    def set_modelmanager(self, modelmanager: ModelManager):
        self.modelmanager = modelmanager
        return self

    def set_termination(self, termination: Termination):
        self.termination = termination
        return self

    def set_archive_init_size(self, size: int):
        self.archive_init_size = size
        return self
    
    def set_archive_atol(self, atol: float):
        self.archive_atol = atol
        return self
    
    def set_archive_rtol(self, rtol: float):
        self.archive_rtol = rtol
        return self
    
    def set_seed(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        return self

    def set_popsize(self, popsize: int):
        self.popsize = popsize
        return self

    def set_instance_name(self, name: str):
        self.instance_name = name
        return self

    def _initialize(self, n_init_archive: int):
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

    def dispatch(self, event: CallbackEvent, data=None, **kwargs):
        kwargs["optimizer"] = self
        return self.cbmanager.dispatch(event, data, **kwargs)

    def run(self):
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
