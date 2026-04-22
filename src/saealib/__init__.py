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
from saealib.algorithms.ga import GA, RepairFunc
from saealib.algorithms.pso import PSO
from saealib.api import Result, maximize, minimize
from saealib.callback import (
    CallbackManager,
    Event,
    GenerationEndEvent,
    GenerationStartEvent,
    PostAskEvent,
    PostCrossoverEvent,
    PostEvaluationEvent,
    PostMutationEvent,
    PostSurrogateFitEvent,
    RunEndEvent,
    RunStartEvent,
    SurrogateEndEvent,
    SurrogateStartEvent,
    logging_generation,
    logging_generation_hv,
)
from saealib.execution.initializer import Initializer, LHSInitializer
from saealib.operators.crossover import (
    Crossover,
    CrossoverBLXAlpha,
    CrossoverOnePoint,
    CrossoverSBX,
    CrossoverTwoPoint,
    CrossoverUniform,
)
from saealib.operators.mutation import (
    Mutation,
    MutationGaussian,
    MutationPolynomial,
    MutationUniform,
)
from saealib.operators.repair import repair_clipping
from saealib.operators.selection import (
    ParentSelection,
    RouletteWheelSelection,
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
from saealib.comparators import (
    Comparator,
    NSGA2Comparator,
    ParetoComparator,
    SingleObjectiveComparator,
    WeightedSumComparator,
    crowding_distance,
    crowding_distance_all_fronts,
    non_dominated_sort,
)
from saealib.problem import (
    Constraint,
    Problem,
)
from saealib.strategies.base import OptimizationStrategy
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
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
from saealib.utils.indicators import hypervolume

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GA",
    "PSO",
    "AcquisitionFunction",
    "Algorithm",
    "Archive",
    "ArchiveMixin",
    "CallbackManager",
    "Comparator",
    "Constraint",
    "Crossover",
    "CrossoverBLXAlpha",
    "CrossoverOnePoint",
    "CrossoverSBX",
    "CrossoverTwoPoint",
    "CrossoverUniform",
    "EnsembleSurrogateManager",
    "Event",
    "ExpectedImprovement",
    "GenerationBasedStrategy",
    "GenerationEndEvent",
    "GenerationStartEvent",
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
    "MutationGaussian",
    "MutationPolynomial",
    "MutationUniform",
    "NSGA2Comparator",
    "OptimizationStrategy",
    "Optimizer",
    "ParentSelection",
    "ParetoComparator",
    "Population",
    "PopulationAttribute",
    "PostAskEvent",
    "PostCrossoverEvent",
    "PostEvaluationEvent",
    "PostMutationEvent",
    "PostSurrogateFitEvent",
    "PreSelectionStrategy",
    "ProbabilityOfFeasibility",
    "Problem",
    "RBFsurrogate",
    "RepairFunc",
    "Result",
    "RouletteWheelSelection",
    "RunEndEvent",
    "RunStartEvent",
    "SequentialSelection",
    "SingleObjectiveComparator",
    "Surrogate",
    "SurrogateEndEvent",
    "SurrogateManager",
    "SurrogatePrediction",
    "SurrogateStartEvent",
    "SurvivorSelection",
    "Termination",
    "TournamentSelection",
    "TruncationSelection",
    "WeightedSumComparator",
    "crowding_distance",
    "crowding_distance_all_fronts",
    "gaussian_kernel",
    "hypervolume",
    "logging_generation",
    "logging_generation_hv",
    "max_fe",
    "max_gen",
    "maximize",
    "minimize",
    "non_dominated_sort",
    "repair_clipping",
]
