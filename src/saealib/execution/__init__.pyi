"""Type stubs for the public execution facade."""

from saealib.core.runtime import PollResult as PollResult
from saealib.execution.evaluator import AsyncEvaluator as AsyncEvaluator
from saealib.execution.evaluator import EvaluationAdapter as EvaluationAdapter
from saealib.execution.evaluator import EvaluationErrorInfo as EvaluationErrorInfo
from saealib.execution.evaluator import EvaluationHandle as EvaluationHandle
from saealib.execution.evaluator import EvaluationQuery as EvaluationQuery
from saealib.execution.evaluator import EvaluationRequest as EvaluationRequest
from saealib.execution.evaluator import EvaluationResult as EvaluationResult
from saealib.execution.evaluator import EvaluationStatus as EvaluationStatus
from saealib.execution.evaluator import EvaluationUpdate as EvaluationUpdate
from saealib.execution.evaluator import Evaluator as Evaluator
from saealib.execution.evaluator import JoblibEvaluator as JoblibEvaluator
from saealib.execution.evaluator import PendingEvaluation as PendingEvaluation
from saealib.execution.evaluator import SerialEvaluator as SerialEvaluator
from saealib.execution.evaluator import ThreadPoolEvaluator as ThreadPoolEvaluator
from saealib.execution.initializer import GenomeInitializer as GenomeInitializer
from saealib.execution.initializer import Initializer as Initializer
from saealib.execution.initializer import LHSInitializer as LHSInitializer
from saealib.execution.initializer import RandomInitializer as RandomInitializer
from saealib.execution.initializer import SobolInitializer as SobolInitializer
from saealib.execution.runtime import RuntimeFactory as RuntimeFactory
from saealib.execution.runtime import RuntimeRegistration as RuntimeRegistration
from saealib.execution.runtime import RuntimeRegistry as RuntimeRegistry
from saealib.execution.runtime import create_runtime as create_runtime
from saealib.execution.runtime import (
    default_runtime_registry as default_runtime_registry,
)
from saealib.execution.scheduler import (
    AsyncEvaluationScheduler as AsyncEvaluationScheduler,
)

__all__: list[str] = [
    "AsyncEvaluationScheduler",
    "AsyncEvaluator",
    "EvaluationAdapter",
    "EvaluationErrorInfo",
    "EvaluationHandle",
    "EvaluationQuery",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluationStatus",
    "EvaluationUpdate",
    "Evaluator",
    "GenomeInitializer",
    "Initializer",
    "JoblibEvaluator",
    "LHSInitializer",
    "PendingEvaluation",
    "PollResult",
    "RandomInitializer",
    "RuntimeFactory",
    "RuntimeRegistration",
    "RuntimeRegistry",
    "SerialEvaluator",
    "SobolInitializer",
    "ThreadPoolEvaluator",
    "create_runtime",
    "default_runtime_registry",
]
