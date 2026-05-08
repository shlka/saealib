"""
Training set strategy objects for surrogate model fitting.

A ``TrainingSet`` abstracts the construction of ``(train_x, train_y)`` pairs
that are passed to ``Surrogate.fit``.  The two orthogonal axes are:

* **Data source** (where samples come from):
  archive, population, k-NN neighbourhood, pairs, per-individual reference points.
* **Labelling rule** (how target values are assigned):
  raw objectives (regression), binary classification, multi-level ranking,
  pairwise comparison.

Literature patterns covered by this module:

+----+----------------------------------------+----------------------------------+
| P1 | CA-LLSO (Wei et al., 2021)             | ``LevelBasedSet``                |
+----+----------------------------------------+----------------------------------+
| P2 | CPS-MOEA (Zhang et al., 2018)          | ``TopKBipartitionSet``           |
+----+----------------------------------------+----------------------------------+
| P3 | Pairwise SAEA (Hao et al., 2024)       | ``PairwiseComparisonSet``        |
+----+----------------------------------------+----------------------------------+
| P4 | SAPSO pbest (Tian et al., 2019)        | ``ReferencePointComparisonSet``  |
+----+----------------------------------------+----------------------------------+
| P5 | CSEA / pre-selection (general)         | ``KNNObjectiveSet``,             |
|    |                                        | ``ArchiveObjectiveSet``          |
+----+----------------------------------------+----------------------------------+
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from saealib.context import OptimizationContext
    from saealib.population import Archive, Population


@dataclass
class TrainingData:
    """Immutable container for the training data passed to ``Surrogate.fit``.

    Attributes
    ----------
    train_x : np.ndarray
        Design variable matrix. shape: ``(n_train, dim)``.
        For ``PairwiseComparisonSet`` the shape is ``(n_train, 2 * dim)``.
    train_y : np.ndarray
        Target values. shape: ``(n_train, n_obj)`` for regression,
        ``(n_train,)`` for classification or ranking.
    """

    train_x: np.ndarray
    train_y: np.ndarray


class TrainingSet(ABC):
    """Strategy object that builds ``(train_x, train_y)`` for a surrogate.

    Inject an instance into ``LocalSurrogateManager`` or
    ``GlobalSurrogateManager`` via their ``training_set`` constructor argument.
    """

    @abstractmethod
    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationContext | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Build and return the training data.

        Parameters
        ----------
        archive:
            Evaluated solutions archive.
        population:
            Current population. Not used by all implementations.
        ctx:
            Optimization context. Provides ``comparator``, ``problem.eps``,
            etc. Required by ranking- and comparison-based implementations.
        candidate_x:
            Centre point for k-NN queries (shape ``(dim,)``). Passed by
            ``LocalSurrogateManager`` for each candidate; ``None`` when called
            from ``GlobalSurrogateManager``.

        Returns
        -------
        TrainingData
        """
        ...


# ---------------------------------------------------------------------------
# Regression defaults (backward-compatible)
# ---------------------------------------------------------------------------


class ArchiveObjectiveSet(TrainingSet):
    """Use the entire archive as training data with raw objective values.

    Default for ``GlobalSurrogateManager``.  Equivalent to the behaviour
    before Issue #026.
    """

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationContext | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return all archive points with raw objective values."""
        return TrainingData(train_x=archive.x, train_y=archive.f)


class KNNObjectiveSet(TrainingSet):
    """Retrieve the *k* nearest archive neighbours of ``candidate_x``.

    Default for ``LocalSurrogateManager``.  Equivalent to the
    ``n_neighbors``-based behaviour before Issue #026.

    Parameters
    ----------
    n_neighbors:
        Number of nearest neighbours to retrieve.
    """

    def __init__(self, n_neighbors: int = 50) -> None:
        self.n_neighbors = n_neighbors

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationContext | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return k nearest-neighbour archive points around ``candidate_x``."""
        if candidate_x is None:
            raise ValueError("KNNObjectiveSet requires candidate_x.")
        idx, _ = archive.get_knn(candidate_x, self.n_neighbors)
        return TrainingData(train_x=archive.x[idx], train_y=archive.f[idx])


# ---------------------------------------------------------------------------
# Constraint-based classification
# ---------------------------------------------------------------------------


class FeasibilityClassificationSet(TrainingSet):
    """Binary labels derived from constraint violation: 1 if feasible, 0 otherwise.

    A sample is feasible when ``cv <= eps``, where ``eps`` is taken from
    ``ctx.problem.eps`` (defaults to ``1e-6`` when ``ctx`` is ``None``).

    This set is algo-agnostic and works with any surrogate that accepts binary
    classification targets (shape ``(n_train,)``).

    Parameters
    ----------
    source:
        ``"archive"`` to use the evaluated archive, or ``"population"`` to
        use the current population.

    Raises
    ------
    ValueError
        If ``source="population"`` but ``population`` is ``None``.
    """

    def __init__(self, source: str = "archive") -> None:
        self.source = source

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationContext | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return feasibility binary labels (1 = feasible, 0 = infeasible)."""
        if self.source == "population":
            if population is None:
                raise ValueError(
                    "FeasibilityClassificationSet: source='population' requires "
                    "population to be provided."
                )
            src = population
        else:
            src = archive
        eps = ctx.problem.eps if ctx is not None else 1e-6
        labels = (src.get_array("cv") <= eps).astype(float)
        return TrainingData(train_x=src.get_array("x"), train_y=labels)
