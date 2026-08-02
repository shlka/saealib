"""saealib: Surrogate-Assisted Evolutionary Algorithms (SAEAs) Library."""

import logging
from importlib.metadata import version

__version__ = version("saealib")

# ---------------------------------------------------------------------------
# Export tiers
#
# Tier 1 (this section, eager import + __all__): entry points likely to be
#   named in the first script or a subclass definition — the 5 root
#   abstractions (Algorithm, OptimizationStrategy, Surrogate,
#   AcquisitionFunction, SurrogateManager), one or two representative default
#   implementations per concept, and the Comparator/Evaluator/Initializer/
#   Termination/Event bases with their common defaults.
# Tier 2 (_TIER2_MAP below, lazy import via __getattr__): every other public
#   component. A name in a subpackage's __all__ belongs here unless it is
#   namespace-only.
# namespace-only (not listed at the top level at all): generic-named bulk
#   sets or domain toolkits, e.g. saealib.benchmarks (sphere/zdt*/dtlz*/...),
#   saealib.registry.get/build/to_spec, and saealib.defaults (internal).
#   Access these via their subpackage directly. Tracked in
#   tests/test_exports.py's NAMESPACE_ONLY allowlist so new subpackage
#   exports can't silently drift out of this scheme.
# ---------------------------------------------------------------------------

from saealib.acquisition import (
    AcquisitionFunction,
    AcquisitionResult,
    ExpectedImprovement,
    PointwiseAcquisition,
)
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
from saealib.exceptions import (
    CheckpointError,
    ConfigurationError,
    EvaluationFatalError,
    EvaluationProtocolError,
    EvaluationSubmissionError,
    SaealibError,
    ValidationError,
)
from saealib.execution.evaluator import (
    AsyncEvaluator,
    EvaluationErrorInfo,
    EvaluationHandle,
    EvaluationRequest,
    EvaluationResult,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    PendingEvaluation,
    SerialEvaluator,
    ThreadPoolEvaluator,
)
from saealib.execution.initializer import (
    Initializer,
    LHSInitializer,
    RandomInitializer,
    SobolInitializer,
)
from saealib.execution.scheduler import AsyncScheduler
from saealib.operators import (
    Crossover,
    CrossoverSBX,
    DuplicateElimination,
    Mutation,
    MutationPolynomial,
    ParentSelection,
    SurvivorSelection,
    TournamentSelection,
    TruncationSelection,
)
from saealib.optimizer import Optimizer
from saealib.pipeline import Pipeline, Stage
from saealib.policies import (
    ComparatorWorstFallback,
    EvaluateAll,
    EvaluationPolicy,
    FeedbackPolicy,
    FeedbackResult,
    MixedFeedback,
    NoFeedback,
    PredictedFeedback,
    RatioEvaluation,
    TopKEvaluation,
    TrueOnlyFeedback,
)
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
from saealib.registry import register
from saealib.stages import (
    AcquisitionStage,
    ArchiveUpdateStage,
    AskStage,
    AsyncEvaluationSubmitStage,
    CountGenerationStage,
    EvaluationAcknowledgeStage,
    EvaluationApplyStage,
    EvaluationCollectStage,
    EvaluationPlanStage,
    EvaluationSubmitStage,
    FeedbackStage,
    InitializationStage,
    SortByScoreStage,
    SurrogateFitStage,
    SurrogateOnlyLoopStage,
    SurrogatePredictStage,
    TellStage,
    TopKSelectionStage,
    TrueEvaluationStage,
)
from saealib.strategies import (
    DirectStrategy,
    GenerationBasedStrategy,
    IndividualBasedStrategy,
    OptimizationStrategy,
    PreSelectionStrategy,
)
from saealib.surrogate import Surrogate, SurrogateManager
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
    "AcquisitionResult",
    "AcquisitionStage",
    "Algorithm",
    "Archive",
    "ArchiveUpdateStage",
    "AskStage",
    "AsyncEvaluationSubmitStage",
    "AsyncEvaluator",
    "AsyncScheduler",
    "CallbackManager",
    "CategoricalVariable",
    "CheckpointCallback",
    "CheckpointError",
    "Comparator",
    "ComparatorWorstFallback",
    "ConfigurationError",
    "ConstraintHandler",
    "ContinuousVariable",
    "CountGenerationStage",
    "Crossover",
    "CrossoverSBX",
    "DirectStrategy",
    "DuplicateElimination",
    "EpsilonConstraintHandler",
    "EqualityConstraint",
    "EvaluateAll",
    "EvaluationErrorInfo",
    "EvaluationFatalError",
    "EvaluationHandle",
    "EvaluationPolicy",
    "EvaluationProtocolError",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluationStatus",
    "EvaluationSubmissionError",
    "EvaluationUpdate",
    "Evaluator",
    "Event",
    "ExpectedImprovement",
    "FeedbackPolicy",
    "FeedbackResult",
    "FeedbackStage",
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
    "MixedFeedback",
    "Mutation",
    "MutationPolynomial",
    "NSGA2Comparator",
    "NoFeedback",
    "OptimizationStrategy",
    "Optimizer",
    "ParentSelection",
    "ParetoArchive",
    "PendingEvaluation",
    "Pipeline",
    "PointwiseAcquisition",
    "Population",
    "PopulationAttribute",
    "PostEvaluationEvent",
    "PreSelectionStrategy",
    "PredictedFeedback",
    "Problem",
    "RandomInitializer",
    "RatioEvaluation",
    "Result",
    "RunEndEvent",
    "RunStartEvent",
    "SaealibError",
    "SerialEvaluator",
    "SingleObjectiveComparator",
    "SobolInitializer",
    "SortByScoreStage",
    "Stage",
    "StaticToleranceHandler",
    "Surrogate",
    "SurrogateFitStage",
    "SurrogateManager",
    "SurrogateOnlyLoopStage",
    "SurrogatePredictStage",
    "SurvivorSelection",
    "TellStage",
    "Termination",
    "TerminationCondition",
    "ThreadPoolEvaluator",
    "TopKEvaluation",
    "TopKSelectionStage",
    "TournamentSelection",
    "TrueEvaluationStage",
    "TrueOnlyFeedback",
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
    "register",
    "stalled",
]

# ---------------------------------------------------------------------------
# Tier 2 — lazy imports (accessible as saealib.<name>, shown in dir())
# ---------------------------------------------------------------------------

_TIER2_MAP: dict[str, str] = {
    # algorithms (less common)
    "PymooAlgorithm": "saealib.algorithms",
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
    "PymooCrossover": "saealib.operators",
    "PymooMutation": "saealib.operators",
    "RouletteWheelSelection": "saealib.operators",
    "SequentialSelection": "saealib.operators",
    "repair_clipping": "saealib.operators",
    # acquisition (less common)
    "CompositeAcquisition": "saealib.acquisition",
    "DensityAcquisition": "saealib.acquisition",
    "EHVIAcquisition": "saealib.acquisition",
    "LowerConfidenceBound": "saealib.acquisition",
    "MaxUncertainty": "saealib.acquisition",
    "MeanPrediction": "saealib.acquisition",
    "NichingAcquisition": "saealib.acquisition",
    "NoveltyAcquisition": "saealib.acquisition",
    "ParEGOAcquisition": "saealib.acquisition",
    "ProbabilityOfFeasibility": "saealib.acquisition",
    "ProductOfFeasibility": "saealib.acquisition",
    "SMSEGOAcquisition": "saealib.acquisition",
    "WinRateAcquisition": "saealib.acquisition",
    # surrogate (specialized)
    "CompositeSurrogateManager": "saealib.surrogate",
    "GlobalSurrogateManager": "saealib.surrogate",
    "LocalSurrogateManager": "saealib.surrogate",
    "PerObjectiveSurrogate": "saealib.surrogate",
    "PredictionChannel": "saealib.surrogate",
    "RBFSurrogate": "saealib.surrogate",
    "SklearnGPRSurrogate": "saealib.surrogate",
    "SklearnLGBMSurrogate": "saealib.surrogate",
    "SklearnNNSurrogate": "saealib.surrogate",
    "SklearnRFRSurrogate": "saealib.surrogate",
    "SklearnSurrogate": "saealib.surrogate",
    "SklearnSVMSurrogate": "saealib.surrogate",
    "SklearnXGBSurrogate": "saealib.surrogate",
    "SurrogatePrediction": "saealib.surrogate",
    "TorchSurrogate": "saealib.surrogate",
    "product_combine": "saealib.surrogate",
    "rank_weighted_combine": "saealib.surrogate",
    # surrogate (training-set builders)
    "ArchiveObjectiveSet": "saealib.surrogate",
    "ConstraintObjectiveSet": "saealib.surrogate",
    "FeasibilityClassificationSet": "saealib.surrogate",
    "KNNConstraintObjectiveSet": "saealib.surrogate",
    "KNNObjectiveSet": "saealib.surrogate",
    "LevelBasedSet": "saealib.surrogate",
    "PairwiseComparisonSet": "saealib.surrogate",
    "ReferencePointComparisonSet": "saealib.surrogate",
    "TopKBipartitionSet": "saealib.surrogate",
    "TrainingData": "saealib.surrogate",
    "TrainingSet": "saealib.surrogate",
    # surrogate (accuracy evaluation)
    "AccuracyEvaluator": "saealib.surrogate",
    "HeldOutAccuracyEvaluator": "saealib.surrogate",
    "KFoldAccuracyEvaluator": "saealib.surrogate",
    "LOOAccuracyEvaluator": "saealib.surrogate",
    "R2Score": "saealib.surrogate",
    "RMSE": "saealib.surrogate",
    "SpearmanCorrelation": "saealib.surrogate",
    "SurrogateAccuracy": "saealib.surrogate",
    "SurrogateAccuracyMetric": "saealib.surrogate",
    # surrogate (switching)
    "AccuracyBasedSurrogateSwitcher": "saealib.surrogate",
    "GenCtrlSwitcher": "saealib.surrogate",
    "ManagerSwitcher": "saealib.surrogate",
    "StrategySwitcher": "saealib.surrogate",
    # surrogate (other)
    "ComparisonSurrogate": "saealib.surrogate",
    "PairwiseSurrogateManager": "saealib.surrogate",
    "RegressionSurrogate": "saealib.surrogate",
    "SklearnClassificationSurrogate": "saealib.surrogate",
    "SklearnRFCClassificationSurrogate": "saealib.surrogate",
    "SklearnSVCClassificationSurrogate": "saealib.surrogate",
    # problem (less common)
    "GradientRepairHandler": "saealib.problem",
    "PymooProblem": "saealib.problem",
    "exponential_epsilon_schedule": "saealib.problem",
    "linear_epsilon_schedule": "saealib.problem",
    # population (mixins)
    "ArchiveMixin": "saealib.population",
    "ParetoMixin": "saealib.population",
    # callbacks (less common)
    "AcquisitionEndEvent": "saealib.callback",
    "AcquisitionStartEvent": "saealib.callback",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__ + list(_TIER2_MAP))
