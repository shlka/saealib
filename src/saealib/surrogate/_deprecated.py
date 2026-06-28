"""Deprecated surrogate aliases."""

from saealib._deprecated import deprecated_class
from saealib.surrogate.sklearn_surrogate import (
    SklearnGPRSurrogate as _SklearnGPRSurrogate,
)


@deprecated_class("SklearnGPRSurrogate")
class GPSurrogate(_SklearnGPRSurrogate):
    pass
