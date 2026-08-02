"""Public exception hierarchy for saealib."""

from typing import Any

__all__ = [
    "CheckpointError",
    "ConfigurationError",
    "EvaluationFatalError",
    "EvaluationProtocolError",
    "EvaluationSubmissionError",
    "SaealibError",
    "ValidationError",
]


class SaealibError(Exception):
    """Base class for all saealib exceptions."""


class ValidationError(SaealibError, ValueError):
    """Raised when user-supplied arguments fail validation at a public boundary."""


class ConfigurationError(SaealibError, ValueError):
    """Raised when an :class:`~saealib.Optimizer` is misconfigured at run time."""


class CheckpointError(ValidationError):
    """Raised when a checkpoint is invalid or cannot be migrated."""


class EvaluationProtocolError(SaealibError, ValueError):
    """Raised when an evaluation lifecycle message violates its contract."""


class EvaluationFatalError(EvaluationProtocolError, RuntimeError):
    """Raised when a post-effect lifecycle failure cannot be retried."""

    def __init__(self, message: str, state: Any) -> None:
        super().__init__(message)
        self.state = state


class EvaluationSubmissionError(EvaluationProtocolError):
    """Raised when submission leaves externally running work."""

    def __init__(self, message: str, state: Any) -> None:
        super().__init__(message)
        self.state = state
