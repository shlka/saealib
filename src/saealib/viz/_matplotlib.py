"""Lazy matplotlib import guard for :mod:`saealib.viz`.

Importing :mod:`saealib.viz` must not pull in matplotlib. Each plotting entry
point calls :func:`_require_matplotlib` at runtime so
environments without the optional ``viz`` extra can still import the package
and read its helpers.

This module never calls :func:`matplotlib.use`; it leaves the backend and the
global ``rcParams`` to the user / environment.
"""

from __future__ import annotations

from types import ModuleType


def _require_matplotlib() -> ModuleType:
    """Return the matplotlib module or raise an actionable ``ImportError``.

    Raises
    ------
    ImportError
        If matplotlib is not installed, with a message pointing at the
        ``saealib[viz]`` extra.
    """
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Visualization requires matplotlib. "
            "Install it with `pip install saealib[viz]`."
        ) from exc
    return matplotlib
