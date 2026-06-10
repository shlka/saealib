"""saealib: Surrogate-Assisted Evolutionary Algorithms (SAEAs) Library."""

import logging
from importlib.metadata import version

__version__ = version("saealib")

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
from saealib.comparators import (
    Comparator,
    Dominator,
    EpsilonDominanceComparator,
    EpsilonDominator,
    NonDominatedSorter,
    NSGA2Comparator,
    ParetoComparator,
    ParetoDominator,
    SingleObjectiveComparator,
    WeightedSumComparator,
    crowding_distance,
    crowding_distance_all_fronts,
    dda_non_dominated_sort,
    non_dominated_sort,
)
from saealib.execution.evaluator import (
    EvaluationResult,
    Evaluator,
    SerialEvaluator,
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
    ParetoArchive,
    ParetoMixin,
    Population,
    PopulationAttribute,
)
from saealib.problem import (
    Constraint,
    ConstraintHandler,
    EqualityConstraint,
    InequalityConstraint,
    Problem,
    StaticToleranceHandler,
)
from saealib.strategies.base import OptimizationStrategy
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.archive_manager import (
    ArchiveBasedManager,
    DensityManager,
    NichingManager,
    NoveltyManager,
)
from saealib.surrogate.base import Surrogate
from saealib.surrogate.manager import (
    EnsembleSurrogateManager,
    GlobalSurrogateManager,
    LocalSurrogateManager,
    SurrogateManager,
)
from saealib.surrogate.per_objective import PerObjectiveSurrogate
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel
from saealib.surrogate.sklearn_surrogate import (
    DTSurrogate,
    GPSurrogate,
    LGBMSurrogate,
    NNSurrogate,
    SklearnSurrogate,
    SVMSurrogate,
    XGBSurrogate,
)
from saealib.surrogate.torch_surrogate import TorchSurrogate
from saealib.termination import (
    Termination,
    TerminationCondition,
    f_target,
    max_fe,
    max_gen,
    stalled,
)
from saealib.utils.indicators import hypervolume

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GA",
    "PSO",
    "AcquisitionFunction",
    "Algorithm",
    "Archive",
    "ArchiveBasedManager",
    "ArchiveMixin",
    "CallbackManager",
    "Comparator",
    "Constraint",
    "ConstraintHandler",
    "Crossover",
    "CrossoverBLXAlpha",
    "CrossoverOnePoint",
    "CrossoverSBX",
    "CrossoverTwoPoint",
    "CrossoverUniform",
    "DTSurrogate",
    "DensityManager",
    "Dominator",
    "EnsembleSurrogateManager",
    "EpsilonDominanceComparator",
    "EpsilonDominator",
    "EqualityConstraint",
    "EvaluationResult",
    "Evaluator",
    "Event",
    "ExpectedImprovement",
    "GPSurrogate",
    "GenerationBasedStrategy",
    "GenerationEndEvent",
    "GenerationStartEvent",
    "GlobalSurrogateManager",
    "Individual",
    "IndividualBasedStrategy",
    "InequalityConstraint",
    "Initializer",
    "LGBMSurrogate",
    "LHSInitializer",
    "LocalSurrogateManager",
    "LowerConfidenceBound",
    "MaxUncertainty",
    "MeanPrediction",
    "Mutation",
    "MutationGaussian",
    "MutationPolynomial",
    "MutationUniform",
    "NNSurrogate",
    "NSGA2Comparator",
    "NichingManager",
    "NonDominatedSorter",
    "NoveltyManager",
    "OptimizationStrategy",
    "Optimizer",
    "ParentSelection",
    "ParetoArchive",
    "ParetoComparator",
    "ParetoDominator",
    "ParetoMixin",
    "PerObjectiveSurrogate",
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
    "SVMSurrogate",
    "SequentialSelection",
    "SerialEvaluator",
    "SingleObjectiveComparator",
    "SklearnSurrogate",
    "StaticToleranceHandler",
    "Surrogate",
    "SurrogateEndEvent",
    "SurrogateManager",
    "SurrogatePrediction",
    "SurrogateStartEvent",
    "SurvivorSelection",
    "Termination",
    "TerminationCondition",
    "TorchSurrogate",
    "TournamentSelection",
    "TruncationSelection",
    "WeightedSumComparator",
    "XGBSurrogate",
    "crowding_distance",
    "crowding_distance_all_fronts",
    "dda_non_dominated_sort",
    "f_target",
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
    "stalled",
]
