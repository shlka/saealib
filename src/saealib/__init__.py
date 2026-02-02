"""
saealib: Surrogate-Assisted Evolutionary Algorithms (SAEAs) Library
"""

import logging

# core components
from saealib.optimizer import Optimizer
from saealib.problem import Problem, Constraint, ConstraintType, ConstraintManager, Comparator, SingleObjectiveComparator
from saealib.algorithm import Algorithm, GA
from saealib.termination import Termination
from saealib.modelmanager import ModelManager, IndividualBasedStrategy
from saealib.callback import CallbackManager, CallbackEvent, logging_generation
from saealib.population import Population, Individual, Archive, PopulationAttribute, ArchiveMixin

# surrogate models
from saealib.surrogate.base import Surrogate
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel

# operators
from saealib.operators.crossover import Crossover, CrossoverBLXAlpha
from saealib.operators.mutation import Mutation, MutationUniform
from saealib.operators.selection import (
    ParentSelection,
    SequentialSelection,
    TournamentSelection,
    SurvivorSelection,
    TruncationSelection,
)
from saealib.operators.repair import repair_clipping

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    # core components
    "Optimizer", 
    "Problem", "Constraint", "ConstraintType", "ConstraintManager", "Comparator", "SingleObjectiveComparator",
    "Algorithm", "GA",
    "Termination",
    "ModelManager", "IndividualBasedStrategy",
    "CallbackManager", "CallbackEvent", "logging_generation",
    "Population", "Individual", "Archive", "PopulationAttribute", "ArchiveMixin",
    # surrogate models
    "Surrogate", "RBFsurrogate", "gaussian_kernel",
    # operators
    "Crossover", "CrossoverBLXAlpha", 
    "Mutation", "MutationUniform", 
    "ParentSelection", "SequentialSelection", "TournamentSelection", 
    "SurvivorSelection", "TruncationSelection", "repair_clipping"
]