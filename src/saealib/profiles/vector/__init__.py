"""Vector evolutionary-algorithm profile registrations.

The framework compiler only knows nominal ``DataSpec`` values and generic
adapter registration.  Vector-specific conversion policy is activated by the
standard top-level package through this module.
"""

from __future__ import annotations

from saealib.core.compiler.adapters import (
    DEFAULT_ADAPTER_REGISTRY,
    Adapter,
    AdapterMatchContext,
)
from saealib.core.contracts import DataSpec
from saealib.core.contracts.representation import RepresentationSpec


def _dense_numeric_view_match(match: AdapterMatchContext) -> bool:
    """Match the vector profile's lossless dense genome view."""
    space = getattr(match.compile_context, "space", None)
    services = getattr(space, "services", space)
    getter = getattr(services, "get", None)
    dense_view = getter("DenseNumericView") if callable(getter) else None
    representation = getattr(space, "representation", None)
    return (
        dense_view is not None
        and isinstance(representation, RepresentationSpec)
        and representation.kind == "vector"
    )


def activate() -> None:
    """Register the vector profile's lossless conversion once."""
    if any(
        adapter.name == "dense_numeric_view"
        for adapter in DEFAULT_ADAPTER_REGISTRY.registrations()
    ):
        return
    DEFAULT_ADAPTER_REGISTRY.register(
        Adapter(
            name="dense_numeric_view",
            source=DataSpec(kind="Population"),
            target=DataSpec(kind="FeatureBatch"),
            lossless=True,
            auto_insertable=True,
            category="lossless_view",
            # The standard profile is part of the built-in core distribution;
            # its implementation lives here, while the generic compiler only
            # sees this registered adapter.
            namespace="core",
            matcher=_dense_numeric_view_match,
        )
    )


activate()

__all__ = ["activate"]
