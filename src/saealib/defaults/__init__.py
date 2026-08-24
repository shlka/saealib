"""Bundled component-spec presets consumed by Optimizer's default resolution."""

from saealib.defaults.builtin import BUILTIN_DEFAULT_PROVIDER, BuiltinDefaultProvider
from saealib.defaults.context import DefaultContext
from saealib.defaults.keys import (
    INITIAL_ARCHIVE_SIZE,
    MAX_EVALUATIONS,
    POPULATION_SIZE,
    DefaultKey,
)
from saealib.defaults.loader import dump_preset, load_defaults, load_preset
from saealib.defaults.model import (
    DefaultHint,
    DefaultResolution,
    DefaultStrength,
    ResolvedDefault,
)
from saealib.defaults.resolver import (
    DEFAULT_RESOLVER,
    DefaultHintProvider,
    DefaultResolver,
)

__all__ = [
    "BUILTIN_DEFAULT_PROVIDER",
    "DEFAULT_RESOLVER",
    "INITIAL_ARCHIVE_SIZE",
    "MAX_EVALUATIONS",
    "POPULATION_SIZE",
    "BuiltinDefaultProvider",
    "DefaultContext",
    "DefaultHint",
    "DefaultHintProvider",
    "DefaultKey",
    "DefaultResolution",
    "DefaultResolver",
    "DefaultStrength",
    "ResolvedDefault",
    "dump_preset",
    "load_defaults",
    "load_preset",
]
