"""Visualization helpers for saealib.

This package is import-safe without matplotlib: importing ``saealib.viz``
does not pull in matplotlib. Plotting entry points lazily import matplotlib
through :mod:`saealib.viz._matplotlib` and raise an actionable
:class:`ImportError` when the optional ``viz`` extra is missing.
"""

from __future__ import annotations

from saealib.viz._plot_history import plot_history
from saealib.viz._plot_result import plot_result

__all__ = ["plot_history", "plot_result"]
