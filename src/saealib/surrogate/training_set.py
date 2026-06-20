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
| P6 | Constraint BO (Regis & Shoemaker 2005) | ``ConstraintObjectiveSet``,      |
|    | Letham et al. (2019)                   | ``KNNConstraintObjectiveSet``    |
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
            Optimization context. Provides ``comparator``, ``problem.eps_cv``,
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
# Constraint regression (raw g values)
# ---------------------------------------------------------------------------


class ConstraintObjectiveSet(TrainingSet):
    """Use the entire archive as training data with raw constraint values.

    Analogue of :class:`ArchiveObjectiveSet` for constraint surrogates.
    Returns ``(archive.x, archive.g)`` so that a surrogate can learn to
    predict the raw constraint values ``g(x)``.

    Raises
    ------
    ValueError
        If the archive has no constraint columns (``archive.g.shape[1] == 0``).
        Use this class only with problems that define at least one constraint.
    """

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationContext | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return all archive points with raw constraint values."""
        g = archive.g
        if g.shape[1] == 0:
            raise ValueError(
                "ConstraintObjectiveSet requires at least one constraint "
                "(archive.g has 0 columns)."
            )
        return TrainingData(train_x=archive.x, train_y=g)


class KNNConstraintObjectiveSet(TrainingSet):
    """Retrieve the *k* nearest archive neighbours of ``candidate_x``.

    Analogue of :class:`KNNObjectiveSet` for constraint surrogates.
    Returns the k-NN subset of ``(archive.x, archive.g)`` centred on
    ``candidate_x``.

    Parameters
    ----------
    n_neighbors:
        Number of nearest neighbours to retrieve.

    Raises
    ------
    ValueError
        If ``candidate_x`` is None, or if the archive has no constraint columns.
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
        """Return k nearest-neighbour archive points with raw constraint values."""
        if candidate_x is None:
            raise ValueError("KNNConstraintObjectiveSet requires candidate_x.")
        g = archive.g
        if g.shape[1] == 0:
            raise ValueError(
                "KNNConstraintObjectiveSet requires at least one constraint "
                "(archive.g has 0 columns)."
            )
        idx, _ = archive.get_knn(candidate_x, self.n_neighbors)
        return TrainingData(train_x=archive.x[idx], train_y=g[idx])


# ---------------------------------------------------------------------------
# Constraint-based classification
# ---------------------------------------------------------------------------


class FeasibilityClassificationSet(TrainingSet):
    """Binary labels derived from constraint violation: 1 if feasible, 0 otherwise.

    A sample is feasible when ``cv <= eps_cv``, where ``eps_cv`` is taken from
    ``ctx.problem.eps_cv`` (defaults to ``1e-6`` when ``ctx`` is ``None``).

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
        eps = ctx.problem.eps_cv if ctx is not None else 1e-6
        labels = (src.get_array("cv") <= eps).astype(float)
        return TrainingData(train_x=src.get_array("x"), train_y=labels)


# ---------------------------------------------------------------------------
# Rank-based labelling
# ---------------------------------------------------------------------------


class TopKBipartitionSet(TrainingSet):
    """Binary labels from a sorted bipartition: 1 for top-*k*, 0 for the rest.

    Individuals are sorted by ``ctx.comparator.sort_population`` (best-first),
    then the top ``floor(n * top_ratio)`` receive label 1 (promising) and the
    remainder receive label 0 (unpromising).

    Literature patterns: CPS-MOEA (P2), CSEA / pre-selection (P5).

    Parameters
    ----------
    source:
        ``"archive"`` or ``"population"``.
    top_ratio:
        Fraction of individuals labelled as promising. Clamped to ``[0, 1]``.
        At least one individual always receives label 1.

    Raises
    ------
    ValueError
        If ``ctx`` is ``None`` or ``source="population"`` with ``population=None``.
    """

    def __init__(self, source: str = "archive", top_ratio: float = 0.5) -> None:
        self.source = source
        self.top_ratio = top_ratio

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationContext | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return bipartition labels based on sorted rank."""
        if ctx is None:
            raise ValueError("TopKBipartitionSet requires ctx.")
        if self.source == "population":
            if population is None:
                raise ValueError(
                    "TopKBipartitionSet: source='population' requires population."
                )
            src = population
        else:
            src = archive
        sorted_idx = ctx.comparator.sort_population(src)
        x = src.get_array("x")[sorted_idx]
        n = len(x)
        k = max(1, int(n * self.top_ratio))
        labels = np.concatenate([np.ones(k), np.zeros(n - k)])
        return TrainingData(train_x=x, train_y=labels)


class LevelBasedSet(TrainingSet):
    """Multi-class labels from equal-division of a sorted population (CA-LLSO, P1).

    Individuals are sorted by ``ctx.comparator.sort_population`` (best-first),
    then divided into ``n_levels`` equally-sized groups.  Group 0 (best) receives
    label 0, group ``n_levels - 1`` (worst) receives the highest label.
    Remainder individuals are appended to the last (worst) group.

    Parameters
    ----------
    source:
        ``"archive"`` or ``"population"``.
    n_levels:
        Number of classification levels (≥ 2).

    Raises
    ------
    ValueError
        If ``ctx`` is ``None`` or ``source="population"`` with ``population=None``.
    """

    def __init__(self, source: str = "population", n_levels: int = 5) -> None:
        self.source = source
        self.n_levels = n_levels

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationContext | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return level labels from equal-size sorted groups."""
        if ctx is None:
            raise ValueError("LevelBasedSet requires ctx.")
        if self.source == "population":
            if population is None:
                raise ValueError(
                    "LevelBasedSet: source='population' requires population."
                )
            src = population
        else:
            src = archive
        sorted_idx = ctx.comparator.sort_population(src)
        x = src.get_array("x")[sorted_idx]
        n = len(x)
        per = n // self.n_levels
        labels = np.repeat(np.arange(self.n_levels), per)
        # Remainder goes to the last (worst) level
        remainder = n - len(labels)
        if remainder > 0:
            labels = np.concatenate([labels, np.full(remainder, self.n_levels - 1)])
        return TrainingData(train_x=x, train_y=labels.astype(float))


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


class PairwiseComparisonSet(TrainingSet):
    """Pairwise labels: ``train_x`` is ``[x_a, x_b]``, label is 1 if *a* beats *b*.

    For each pair ``(a, b)``, the feature vector is the concatenation
    ``[x_a, x_b]`` (shape ``(n_pairs, 2 * dim)``), and the label is 1 if
    ``comparator.compare(f_a, cv_a, f_b, cv_b) <= 0`` (a dominates or equals b),
    0 otherwise.

    .. warning::
        ``train_x`` has shape ``(n_pairs, 2 * dim)``, which is incompatible with
        standard regression surrogates (e.g. ``RBFSurrogate``).  Use a
        pairwise-specific surrogate that expects concatenated feature pairs.

    Parameters
    ----------
    source:
        ``"archive"`` or ``"population"``.
    n_pairs:
        Number of pairs to sample.  ``None`` (default) uses all
        ``n * (n-1) / 2`` pairs.  If the requested count exceeds the total
        available pairs, all pairs are used.
    rng:
        Random number generator for pair sampling.  Falls back to
        ``ctx.rng`` when ``ctx`` is provided, otherwise creates a new one.

    Raises
    ------
    ValueError
        If ``ctx`` is ``None`` (comparator required) or
        ``source="population"`` with ``population=None``.
    """

    def __init__(
        self,
        source: str = "archive",
        n_pairs: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.source = source
        self.n_pairs = n_pairs
        self.rng = rng

    def build(
        self,
        archive: Archive,
        population: Population | None,
        ctx: OptimizationContext | None,
        candidate_x: np.ndarray | None = None,
    ) -> TrainingData:
        """Return pairwise comparison training data."""
        if ctx is None:
            raise ValueError("PairwiseComparisonSet requires ctx.")
        if self.source == "population":
            if population is None:
                raise ValueError(
                    "PairwiseComparisonSet: source='population' requires population."
                )
            src = population
        else:
            src = archive
        x = src.get_array("x")
        f = src.get_array("f")
        cv = src.get_array("cv")
        cmp = ctx.comparator
        rng = self.rng if self.rng is not None else ctx.rng
        n = len(x)
        n_all = n * (n - 1) // 2
        if self.n_pairs is None or self.n_pairs >= n_all:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            pairs: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            while len(pairs) < self.n_pairs:
                idx = rng.choice(n, size=2, replace=False)
                key = (int(min(idx[0], idx[1])), int(max(idx[0], idx[1])))
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
        train_x = np.stack([np.concatenate([x[i], x[j]]) for i, j in pairs])
        train_y = np.array(
            [
                1.0 if cmp.compare(f[i], cv[i], f[j], cv[j]) <= 0 else 0.0
                for i, j in pairs
            ]
        )
        return TrainingData(train_x=train_x, train_y=train_y)
