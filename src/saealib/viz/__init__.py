"""Visualization helpers for saealib.

This package is import-safe without matplotlib: importing ``saealib.viz``
does not pull in matplotlib. Plotting entry points lazily import matplotlib
through :mod:`saealib.viz._matplotlib` and raise an actionable
:class:`ImportError` when the optional ``viz`` extra is missing.
"""

from __future__ import annotations

from saealib.viz._design import (
    plot_archive,
    plot_design_pcp,
    plot_diversity,
    plot_variable_distribution,
)
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
from saealib.viz._surrogate import (
    plot_prescreening,
    plot_surrogate_accuracy,
    plot_surrogate_error_history,
)

__all__ = [
    "plot_archive",
    "plot_constraint_violation",
    "plot_convergence",
    "plot_design_pcp",
    "plot_diversity",
    "plot_hypervolume",
    "plot_indicator",
    "plot_objective_heatmap",
    "plot_pareto",
    "plot_pareto_evolution",
    "plot_pcp",
    "plot_prescreening",
    "plot_radar",
    "plot_running_metric",
    "plot_surrogate_accuracy",
    "plot_surrogate_error_history",
    "plot_variable_distribution",
]
