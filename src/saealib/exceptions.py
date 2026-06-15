"""Public exception hierarchy for saealib."""

__all__ = ["ConfigurationError", "SaealibError", "ValidationError"]


class SaealibError(Exception):
    """Base class for all saealib exceptions."""


class ValidationError(SaealibError, ValueError):
    """Raised when user-supplied arguments fail validation at a public boundary."""


class ConfigurationError(SaealibError, ValueError):
    """Raised when an :class:`~saealib.Optimizer` is misconfigured at run time."""
