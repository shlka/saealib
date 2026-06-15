"""Deprecated surrogate aliases."""

from saealib._deprecated import deprecated_class
from saealib.surrogate.sklearn_surrogate import GPRSurrogate as _GPRSurrogate


@deprecated_class("GPRSurrogate")
class GPSurrogate(_GPRSurrogate):
    pass
