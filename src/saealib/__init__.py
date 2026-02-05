"""saealib: Surrogate-Assisted Evolutionary Algorithms (SAEAs) Library."""

import logging

from saealib.algorithm import GA, Algorithm
from saealib.callback import CallbackEvent, CallbackManager, logging_generation
from saealib.modelmanager import IndividualBasedStrategy, ModelManager

# operators
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

# core components
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

# surrogate models
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
    # operators
    "Crossover",
    "CrossoverBLXAlpha",
    "Individual",
    "IndividualBasedStrategy",
    "ModelManager",
    "Mutation",
    "MutationUniform",
    # core components
    "Optimizer",
    "ParentSelection",
    "Population",
    "PopulationAttribute",
    "Problem",
    "RBFsurrogate",
    "SequentialSelection",
    "SingleObjectiveComparator",
    # surrogate models
    "Surrogate",
    "SurvivorSelection",
    "Termination",
    "TournamentSelection",
    "TruncationSelection",
    "gaussian_kernel",
    "logging_generation",
    "repair_clipping",
]
