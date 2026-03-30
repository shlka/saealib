"""saealib: Surrogate-Assisted Evolutionary Algorithms (SAEAs) Library."""

import logging

from saealib.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    LowerConfidenceBound,
    MaxUncertainty,
    MeanPrediction,
    ProbabilityOfFeasibility,
)
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
    NSGA2Comparator,
    Problem,
    SingleObjectiveComparator,
    WeightedSumComparator,
    crowding_distance,
    crowding_distance_all_fronts,
    non_dominated_sort,
)
from saealib.strategies.base import OptimizationStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.surrogate.base import Surrogate
from saealib.surrogate.manager import (
    EnsembleSurrogateManager,
    GlobalSurrogateManager,
    LocalSurrogateManager,
    SurrogateManager,
)
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel
from saealib.termination import Termination, max_fe, max_gen

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GA",
    "AcquisitionFunction",
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
    "EnsembleSurrogateManager",
    "ExpectedImprovement",
    "GlobalSurrogateManager",
    "Individual",
    "IndividualBasedStrategy",
    "Initializer",
    "LHSInitializer",
    "LocalSurrogateManager",
    "LowerConfidenceBound",
    "MaxUncertainty",
    "MeanPrediction",
    "Mutation",
    "MutationUniform",
    "OptimizationStrategy",
    "Optimizer",
    "ParentSelection",
    "NSGA2Comparator",
    "Population",
    "PopulationAttribute",
    "ProbabilityOfFeasibility",
    "Problem",
    "RBFsurrogate",
    "SequentialSelection",
    "SingleObjectiveComparator",
    "Surrogate",
    "SurrogateManager",
    "SurrogatePrediction",
    "SurvivorSelection",
    "Termination",
    "TournamentSelection",
    "TruncationSelection",
    "WeightedSumComparator",
    "crowding_distance",
    "crowding_distance_all_fronts",
    "gaussian_kernel",
    "logging_generation",
    "max_fe",
    "max_gen",
    "non_dominated_sort",
    "repair_clipping",
]
