"""
saealib: Surrogate-Assisted Evolutionary Algorithms (SAEAs) Library
"""

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
    # core components
    "Optimizer",
    "Problem",
    "Constraint",
    "ConstraintType",
    "ConstraintManager",
    "Comparator",
    "SingleObjectiveComparator",
    "Algorithm",
    "GA",
    "Termination",
    "ModelManager",
    "IndividualBasedStrategy",
    "CallbackManager",
    "CallbackEvent",
    "logging_generation",
    "Population",
    "Individual",
    "Archive",
    "PopulationAttribute",
    "ArchiveMixin",
    # surrogate models
    "Surrogate",
    "RBFsurrogate",
    "gaussian_kernel",
    # operators
    "Crossover",
    "CrossoverBLXAlpha",
    "Mutation",
    "MutationUniform",
    "ParentSelection",
    "SequentialSelection",
    "TournamentSelection",
    "SurvivorSelection",
    "TruncationSelection",
    "repair_clipping",
]
