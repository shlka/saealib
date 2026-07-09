"""Type stubs for saealib top-level package.

Covers both Tier 1 (eager imports) and Tier 2 (lazy-loaded via __getattr__).
"""

__all__: list[str]

# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------

from saealib.acquisition import AcquisitionFunction as AcquisitionFunction
from saealib.acquisition import ExpectedImprovement as ExpectedImprovement

# acquisition (less common)
from saealib.acquisition import LowerConfidenceBound as LowerConfidenceBound
from saealib.acquisition import MaxUncertainty as MaxUncertainty
from saealib.acquisition import MeanPrediction as MeanPrediction
from saealib.acquisition import ProbabilityOfFeasibility as ProbabilityOfFeasibility
from saealib.acquisition import ProductOfFeasibility as ProductOfFeasibility
from saealib.algorithms import GA as GA
from saealib.algorithms import PSO as PSO
from saealib.algorithms import Algorithm as Algorithm
from saealib.api import Result as Result
from saealib.api import maximize as maximize
from saealib.api import minimize as minimize
from saealib.callback import CallbackManager as CallbackManager
from saealib.callback import Event as Event
from saealib.callback import GenerationEndEvent as GenerationEndEvent
from saealib.callback import GenerationStartEvent as GenerationStartEvent
from saealib.callback import InitialEvaluationEndEvent as InitialEvaluationEndEvent
from saealib.callback import InitialEvaluationStartEvent as InitialEvaluationStartEvent
from saealib.callback import PostAskEvent as PostAskEvent
from saealib.callback import PostCrossoverEvent as PostCrossoverEvent
from saealib.callback import PostEvaluationEvent as PostEvaluationEvent
from saealib.callback import PostMutationEvent as PostMutationEvent
from saealib.callback import PostSurrogateFitEvent as PostSurrogateFitEvent
from saealib.callback import RunEndEvent as RunEndEvent
from saealib.callback import RunStartEvent as RunStartEvent
from saealib.callback import SurrogateEndEvent as SurrogateEndEvent
from saealib.callback import SurrogateStartEvent as SurrogateStartEvent

# callbacks (less common)
from saealib.callback import logging_generation as logging_generation
from saealib.callback import logging_generation_hv as logging_generation_hv
from saealib.checkpoint import CheckpointCallback as CheckpointCallback
from saealib.comparators import Comparator as Comparator
from saealib.comparators import Dominator as Dominator
from saealib.comparators import EpsilonDominanceComparator as EpsilonDominanceComparator
from saealib.comparators import EpsilonDominator as EpsilonDominator
from saealib.comparators import HypervolumeComparator as HypervolumeComparator
from saealib.comparators import NonDominatedSorter as NonDominatedSorter
from saealib.comparators import NSGA2Comparator as NSGA2Comparator
from saealib.comparators import NSGA3Comparator as NSGA3Comparator
from saealib.comparators import ParetoComparator as ParetoComparator
from saealib.comparators import ParetoDominator as ParetoDominator
from saealib.comparators import RNSGA2Comparator as RNSGA2Comparator
from saealib.comparators import SingleObjectiveComparator as SingleObjectiveComparator
from saealib.comparators import SPEA2Comparator as SPEA2Comparator
from saealib.comparators import WeightedSumComparator as WeightedSumComparator

# ---------------------------------------------------------------------------
# Tier 2 (lazy-loaded at runtime via __getattr__)
# ---------------------------------------------------------------------------
# comparators
from saealib.comparators import crowding_distance as crowding_distance
from saealib.comparators import (
    crowding_distance_all_fronts as crowding_distance_all_fronts,
)
from saealib.comparators import dda_non_dominated_sort as dda_non_dominated_sort
from saealib.comparators import non_dominated_sort as non_dominated_sort
from saealib.comparators import spea2_fitness as spea2_fitness

# decomposition
from saealib.decomposition import Decomposition as Decomposition
from saealib.decomposition import DecompositionComparator as DecompositionComparator
from saealib.decomposition import PBIDecomposition as PBIDecomposition
from saealib.decomposition import TchebycheffDecomposition as TchebycheffDecomposition
from saealib.decomposition import WeightedSumDecomposition as WeightedSumDecomposition
from saealib.exceptions import ConfigurationError as ConfigurationError
from saealib.exceptions import SaealibError as SaealibError
from saealib.exceptions import ValidationError as ValidationError
from saealib.execution.evaluator import EvaluationResult as EvaluationResult
from saealib.execution.evaluator import Evaluator as Evaluator
from saealib.execution.evaluator import JoblibEvaluator as JoblibEvaluator
from saealib.execution.evaluator import SerialEvaluator as SerialEvaluator
from saealib.execution.initializer import Initializer as Initializer
from saealib.execution.initializer import LHSInitializer as LHSInitializer
from saealib.execution.initializer import RandomInitializer as RandomInitializer
from saealib.execution.initializer import SobolInitializer as SobolInitializer
from saealib.operators import Crossover as Crossover

# operators (less common)
from saealib.operators import CrossoverBLXAlpha as CrossoverBLXAlpha
from saealib.operators import CrossoverCategorical as CrossoverCategorical
from saealib.operators import CrossoverIntegerSBX as CrossoverIntegerSBX
from saealib.operators import CrossoverOnePoint as CrossoverOnePoint
from saealib.operators import CrossoverSBX as CrossoverSBX
from saealib.operators import CrossoverTwoPoint as CrossoverTwoPoint
from saealib.operators import CrossoverUniform as CrossoverUniform
from saealib.operators import DuplicateElimination as DuplicateElimination
from saealib.operators import Mutation as Mutation
from saealib.operators import MutationCategorical as MutationCategorical
from saealib.operators import MutationGaussian as MutationGaussian
from saealib.operators import MutationIntegerUniform as MutationIntegerUniform
from saealib.operators import MutationPolynomial as MutationPolynomial
from saealib.operators import MutationUniform as MutationUniform
from saealib.operators import ParentSelection as ParentSelection
from saealib.operators import RouletteWheelSelection as RouletteWheelSelection
from saealib.operators import SequentialSelection as SequentialSelection
from saealib.operators import SurvivorSelection as SurvivorSelection
from saealib.operators import TournamentSelection as TournamentSelection
from saealib.operators import TruncationSelection as TruncationSelection
from saealib.operators import repair_clipping as repair_clipping
from saealib.optimizer import Optimizer as Optimizer
from saealib.pipeline import Pipeline as Pipeline
from saealib.pipeline import Stage as Stage
from saealib.population import Archive as Archive

# population (mixins)
from saealib.population import ArchiveMixin as ArchiveMixin
from saealib.population import Individual as Individual
from saealib.population import ParetoArchive as ParetoArchive
from saealib.population import ParetoMixin as ParetoMixin
from saealib.population import Population as Population
from saealib.population import PopulationAttribute as PopulationAttribute

# problem (less common)
from saealib.problem import Constraint as Constraint
from saealib.problem import ConstraintHandler as ConstraintHandler
from saealib.problem import EpsilonConstraintHandler as EpsilonConstraintHandler
from saealib.problem import EqualityConstraint as EqualityConstraint
from saealib.problem import GradientRepairHandler as GradientRepairHandler
from saealib.problem import InequalityConstraint as InequalityConstraint
from saealib.problem import Problem as Problem
from saealib.problem import StaticToleranceHandler as StaticToleranceHandler
from saealib.problem import exponential_epsilon_schedule as exponential_epsilon_schedule
from saealib.problem import linear_epsilon_schedule as linear_epsilon_schedule
from saealib.registry import register as register

# stages
from saealib.stages import ArchiveUpdateStage as ArchiveUpdateStage
from saealib.stages import AskStage as AskStage
from saealib.stages import CountGenerationStage as CountGenerationStage
from saealib.stages import InitializationStage as InitializationStage
from saealib.stages import SortByScoreStage as SortByScoreStage
from saealib.stages import SurrogateFitStage as SurrogateFitStage
from saealib.stages import SurrogateOnlyLoopStage as SurrogateOnlyLoopStage
from saealib.stages import SurrogateScoreStage as SurrogateScoreStage
from saealib.stages import TellStage as TellStage
from saealib.stages import TopKSelectionStage as TopKSelectionStage
from saealib.stages import TrueEvaluationStage as TrueEvaluationStage
from saealib.strategies import DirectStrategy as DirectStrategy
from saealib.strategies import GenerationBasedStrategy as GenerationBasedStrategy
from saealib.strategies import IndividualBasedStrategy as IndividualBasedStrategy
from saealib.strategies import OptimizationStrategy as OptimizationStrategy
from saealib.strategies import PreSelectionStrategy as PreSelectionStrategy

# surrogate (specialized)
from saealib.surrogate import ArchiveBasedManager as ArchiveBasedManager
from saealib.surrogate import CompositeSurrogateManager as CompositeSurrogateManager
from saealib.surrogate import DensityManager as DensityManager
from saealib.surrogate import GlobalSurrogateManager as GlobalSurrogateManager
from saealib.surrogate import LocalSurrogateManager as LocalSurrogateManager
from saealib.surrogate import NichingManager as NichingManager
from saealib.surrogate import NoveltyManager as NoveltyManager
from saealib.surrogate import PerObjectiveSurrogate as PerObjectiveSurrogate
from saealib.surrogate import RBFSurrogate as RBFSurrogate
from saealib.surrogate import SklearnGPRSurrogate as SklearnGPRSurrogate
from saealib.surrogate import SklearnLGBMSurrogate as SklearnLGBMSurrogate
from saealib.surrogate import SklearnNNSurrogate as SklearnNNSurrogate
from saealib.surrogate import SklearnRFRSurrogate as SklearnRFRSurrogate
from saealib.surrogate import SklearnSurrogate as SklearnSurrogate
from saealib.surrogate import SklearnSVMSurrogate as SklearnSVMSurrogate
from saealib.surrogate import SklearnXGBSurrogate as SklearnXGBSurrogate
from saealib.surrogate import Surrogate as Surrogate
from saealib.surrogate import SurrogateManager as SurrogateManager
from saealib.surrogate import SurrogatePrediction as SurrogatePrediction
from saealib.surrogate import TorchSurrogate as TorchSurrogate
from saealib.surrogate import product_combine as product_combine
from saealib.surrogate import rank_weighted_combine as rank_weighted_combine

# deprecated
from saealib.surrogate._deprecated import GPSurrogate as GPSurrogate
from saealib.surrogate.rbf import RBFSurrogate as RBFsurrogate  # noqa: F401

# utils
from saealib.surrogate.rbf import gaussian_kernel as gaussian_kernel
from saealib.termination import Termination as Termination
from saealib.termination import TerminationCondition as TerminationCondition
from saealib.termination import f_target as f_target
from saealib.termination import max_fe as max_fe
from saealib.termination import max_gen as max_gen
from saealib.termination import stalled as stalled
from saealib.utils.indicators import hypervolume as hypervolume
from saealib.utils.indicators import (
    hypervolume_contributions as hypervolume_contributions,
)
from saealib.utils.weight_vectors import (
    uniform_weight_vectors as uniform_weight_vectors,
)
from saealib.variables import CategoricalVariable as CategoricalVariable
from saealib.variables import ContinuousVariable as ContinuousVariable
from saealib.variables import IntegerVariable as IntegerVariable
from saealib.variables import Variable as Variable
