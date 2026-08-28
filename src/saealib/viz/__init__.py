"""Visualization helpers for saealib.

This package is import-safe without matplotlib: importing ``saealib.viz``
does not pull in matplotlib. Plotting entry points lazily import matplotlib
through :mod:`saealib.viz._matplotlib` and raise an actionable
:class:`ImportError` when the optional ``viz`` extra is missing.
"""

from __future__ import annotations

from saealib.viz._objective import (
    plot_objective_heatmap,
    plot_pareto,
    plot_pareto_evolution,
    plot_pcp,
    plot_radar,
)
from saealib.viz._progress import (
    plot_constraint_violation,
    plot_convergence,
    plot_hypervolume,
    plot_indicator,
    plot_running_metric,
)

__all__ = [
    "plot_constraint_violation",
    "plot_convergence",
    "plot_hypervolume",
    "plot_indicator",
    "plot_objective_heatmap",
    "plot_pareto",
    "plot_pareto_evolution",
    "plot_pcp",
    "plot_radar",
    "plot_running_metric",
]
