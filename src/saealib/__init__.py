"""saealib: Surrogate-Assisted Evolutionary Algorithms (SAEAs) Library."""

import logging
from importlib.metadata import version

__version__ = version("saealib")

# ---------------------------------------------------------------------------
# Tier 1 — eager imports (always available, listed in __all__)
# ---------------------------------------------------------------------------

from saealib.acquisition import AcquisitionFunction, ExpectedImprovement
from saealib.algorithms import GA, PSO, Algorithm
from saealib.api import Result, maximize, minimize
from saealib.callback import (
    CallbackManager,
    Event,
    GenerationEndEvent,
    GenerationStartEvent,
    InitialEvaluationEndEvent,
    InitialEvaluationStartEvent,
    PostEvaluationEvent,
    RunEndEvent,
    RunStartEvent,
)
from saealib.checkpoint import CheckpointCallback
from saealib.comparators import Comparator, NSGA2Comparator, SingleObjectiveComparator
from saealib.exceptions import ConfigurationError, SaealibError, ValidationError
from saealib.execution.evaluator import EvaluationResult, Evaluator, SerialEvaluator
from saealib.execution.initializer import Initializer, LHSInitializer
from saealib.operators import (
    Crossover,
    CrossoverSBX,
    Mutation,
    MutationPolynomial,
    ParentSelection,
    SurvivorSelection,
    TournamentSelection,
    TruncationSelection,
)
from saealib.optimizer import Optimizer
from saealib.pipeline import Pipeline, Stage
from saealib.population import (
    Archive,
    Individual,
    ParetoArchive,
    Population,
    PopulationAttribute,
)
from saealib.problem import (
    ConstraintHandler,
    EpsilonConstraintHandler,
    EqualityConstraint,
    InequalityConstraint,
    Problem,
    StaticToleranceHandler,
)
from saealib.stages import (
    ArchiveUpdateStage,
    AskStage,
    CountGenerationStage,
    InitializationStage,
    SortByScoreStage,
    SurrogateFitStage,
    SurrogateOnlyLoopStage,
    SurrogateScoreStage,
    TellStage,
    TopKSelectionStage,
    TrueEvaluationStage,
)
from saealib.strategies import (
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    OptimizationStrategy,
    PreSelectionStrategy,
)
from saealib.surrogate import GPRSurrogate, Surrogate, SurrogateManager
from saealib.termination import (
    Termination,
    TerminationCondition,
    f_target,
    max_fe,
    max_gen,
    stalled,
)
from saealib.utils.indicators import hypervolume, hypervolume_contributions
from saealib.variables import (
    CategoricalVariable,
    ContinuousVariable,
    IntegerVariable,
    Variable,
)

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GA",
    "PSO",
    "AcquisitionFunction",
    "Algorithm",
    "Archive",
    "ArchiveUpdateStage",
    "AskStage",
    "CallbackManager",
    "CategoricalVariable",
    "CheckpointCallback",
    "Comparator",
    "ConfigurationError",
    "ConstraintHandler",
    "ContinuousVariable",
    "CountGenerationStage",
    "Crossover",
    "CrossoverSBX",
    "EpsilonConstraintHandler",
    "EqualityConstraint",
    "EvaluationResult",
    "Evaluator",
    "Event",
    "ExpectedImprovement",
    "GPRSurrogate",
    "GenerationBasedStrategy",
    "GenerationEndEvent",
    "GenerationStartEvent",
    "Individual",
    "IndividualBasedStrategy",
    "InequalityConstraint",
    "InitialEvaluationEndEvent",
    "InitialEvaluationStartEvent",
    "InitializationStage",
    "Initializer",
    "IntegerVariable",
    "LHSInitializer",
    "Mutation",
    "MutationPolynomial",
    "NSGA2Comparator",
    "OptimizationStrategy",
    "Optimizer",
    "ParentSelection",
    "ParetoArchive",
    "Pipeline",
    "Population",
    "PopulationAttribute",
    "PostEvaluationEvent",
    "PreSelectionStrategy",
    "Problem",
    "Result",
    "RunEndEvent",
    "RunStartEvent",
    "SaealibError",
    "SerialEvaluator",
    "SingleObjectiveComparator",
    "SortByScoreStage",
    "Stage",
    "StaticToleranceHandler",
    "Surrogate",
    "SurrogateFitStage",
    "SurrogateManager",
    "SurrogateOnlyLoopStage",
    "SurrogateScoreStage",
    "SurvivorSelection",
    "TellStage",
    "Termination",
    "TerminationCondition",
    "TopKSelectionStage",
    "TournamentSelection",
    "TrueEvaluationStage",
    "TruncationSelection",
    "ValidationError",
    "Variable",
    "f_target",
    "hypervolume",
    "hypervolume_contributions",
    "max_fe",
    "max_gen",
    "maximize",
    "minimize",
    "stalled",
]

# ---------------------------------------------------------------------------
# Tier 2 — lazy imports (accessible as saealib.<name>, shown in dir())
# ---------------------------------------------------------------------------

_TIER2_MAP: dict[str, str] = {
    # comparators (less common)
    "Dominator": "saealib.comparators",
    "EpsilonDominanceComparator": "saealib.comparators",
    "EpsilonDominator": "saealib.comparators",
    "HypervolumeComparator": "saealib.comparators",
    "NonDominatedSorter": "saealib.comparators",
    "NSGA3Comparator": "saealib.comparators",
    "ParetoComparator": "saealib.comparators",
    "ParetoDominator": "saealib.comparators",
    "RNSGA2Comparator": "saealib.comparators",
    "SPEA2Comparator": "saealib.comparators",
    "WeightedSumComparator": "saealib.comparators",
    "crowding_distance": "saealib.comparators",
    "crowding_distance_all_fronts": "saealib.comparators",
    "dda_non_dominated_sort": "saealib.comparators",
    "non_dominated_sort": "saealib.comparators",
    "spea2_fitness": "saealib.comparators",
    # execution (parallel)
    "JoblibEvaluator": "saealib.execution.evaluator",
    # decomposition
    "Decomposition": "saealib.decomposition",
    "DecompositionComparator": "saealib.decomposition",
    "PBIDecomposition": "saealib.decomposition",
    "TchebycheffDecomposition": "saealib.decomposition",
    "WeightedSumDecomposition": "saealib.decomposition",
    # operators (less common)
    "CrossoverBLXAlpha": "saealib.operators",
    "CrossoverCategorical": "saealib.operators",
    "CrossoverIntegerSBX": "saealib.operators",
    "CrossoverOnePoint": "saealib.operators",
    "CrossoverTwoPoint": "saealib.operators",
    "CrossoverUniform": "saealib.operators",
    "MutationCategorical": "saealib.operators",
    "MutationGaussian": "saealib.operators",
    "MutationIntegerUniform": "saealib.operators",
    "MutationUniform": "saealib.operators",
    "RouletteWheelSelection": "saealib.operators",
    "SequentialSelection": "saealib.operators",
    "repair_clipping": "saealib.operators",
    # acquisition (less common)
    "LowerConfidenceBound": "saealib.acquisition",
    "MaxUncertainty": "saealib.acquisition",
    "MeanPrediction": "saealib.acquisition",
    "ProbabilityOfFeasibility": "saealib.acquisition",
    "ProductOfFeasibility": "saealib.acquisition",
    # surrogate (specialized)
    "ArchiveBasedManager": "saealib.surrogate",
    "CompositeSurrogateManager": "saealib.surrogate",
    "DensityManager": "saealib.surrogate",
    "DTSurrogate": "saealib.surrogate",
    "GlobalSurrogateManager": "saealib.surrogate",
    "LGBMSurrogate": "saealib.surrogate",
    "LocalSurrogateManager": "saealib.surrogate",
    "NichingManager": "saealib.surrogate",
    "NNSurrogate": "saealib.surrogate",
    "NoveltyManager": "saealib.surrogate",
    "PerObjectiveSurrogate": "saealib.surrogate",
    "RBFSurrogate": "saealib.surrogate",
    "SklearnSurrogate": "saealib.surrogate",
    "SVMSurrogate": "saealib.surrogate",
    "SurrogatePrediction": "saealib.surrogate",
    "TorchSurrogate": "saealib.surrogate",
    "XGBSurrogate": "saealib.surrogate",
    "product_combine": "saealib.surrogate",
    "rank_weighted_combine": "saealib.surrogate",
    # problem (less common)
    "Constraint": "saealib.problem",
    "GradientRepairHandler": "saealib.problem",
    "exponential_epsilon_schedule": "saealib.problem",
    "linear_epsilon_schedule": "saealib.problem",
    # population (mixins)
    "ArchiveMixin": "saealib.population",
    "ParetoMixin": "saealib.population",
    # callbacks (less common)
    "PostAskEvent": "saealib.callback",
    "PostCrossoverEvent": "saealib.callback",
    "PostMutationEvent": "saealib.callback",
    "PostSurrogateFitEvent": "saealib.callback",
    "SurrogateEndEvent": "saealib.callback",
    "SurrogateStartEvent": "saealib.callback",
    "logging_generation": "saealib.callback",
    "logging_generation_hv": "saealib.callback",
    # utils
    "gaussian_kernel": "saealib.surrogate.rbf",
    "uniform_weight_vectors": "saealib.utils.weight_vectors",
}


def __getattr__(name: str) -> object:
    if name in _TIER2_MAP:
        import importlib

        mod = importlib.import_module(_TIER2_MAP[name])
        obj = getattr(mod, name)
        globals()[name] = obj  # cache to avoid repeated lookup
        return obj
    if name == "GPSurrogate":
        from saealib.surrogate._deprecated import GPSurrogate

        return GPSurrogate
    if name == "RBFsurrogate":
        from saealib.surrogate.rbf import RBFSurrogate

        return RBFSurrogate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__ + list(_TIER2_MAP))
