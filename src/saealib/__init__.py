"""saealib: Surrogate-Assisted Evolutionary Algorithms (SAEAs) Library."""

import logging

from saealib.algorithms.base import Algorithm
from saealib.algorithms.ga import GA
from saealib.callback import CallbackEvent, CallbackManager, logging_generation
from saealib.execution.initializer import Initializer, LHSInitializer
from saealib.operators.crossover import Crossover, CrossoverBLXAlpha
from saealib.operators.mutation import Mutation, MutationUniform
from saealib.operators.repair import repair_clipping
from saealib.operators.selection import (
    ParentSelection,
    SequentialSelection,
    SurvivorSelection,
    TournamentSelection,
    TruncationSelection,
)
from saealib.optimizer import Optimizer
from saealib.population import (
    Archive,
    ArchiveMixin,
    Individual,
    Population,
    PopulationAttribute,
)
from saealib.problem import (
    Comparator,
    Constraint,
    ConstraintManager,
    ConstraintType,
    Problem,
    SingleObjectiveComparator,
)
from saealib.strategies.base import OptimizationStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.surrogate.base import Surrogate
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel
from saealib.termination import Termination

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GA",
    "Algorithm",
    "Archive",
    "ArchiveMixin",
    "CallbackEvent",
    "CallbackManager",
    "Comparator",
    "Constraint",
    "ConstraintManager",
    "ConstraintType",
    "Crossover",
    "CrossoverBLXAlpha",
    "Individual",
    "IndividualBasedStrategy",
    "Initializer",
    "LHSInitializer",
    "Mutation",
    "MutationUniform",
    "OptimizationStrategy",
    "Optimizer",
    "ParentSelection",
    "Population",
    "PopulationAttribute",
    "Problem",
    "RBFsurrogate",
    "SequentialSelection",
    "SingleObjectiveComparator",
    "Surrogate",
    "SurvivorSelection",
    "Termination",
    "TournamentSelection",
    "TruncationSelection",
    "gaussian_kernel",
    "logging_generation",
    "repair_clipping",
]
