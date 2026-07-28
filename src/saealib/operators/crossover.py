"""Crossover operators for evolutionary algorithms."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from saealib.registry import register

if TYPE_CHECKING:
    from saealib.context import OptimizationState


class Crossover(ABC):
    """Base class for crossover operators.

    Attributes
    ----------
    n_parents : int
        Number of parents required per crossover call. Default is 2.
        Subclasses may override this class attribute if they require a
        different number of parents.
    n_children : int
        Number of offspring produced per crossover call. Default is 2.
        Subclasses may override this class attribute if they produce a
        different number of offspring.
    prob : float
        Individual-level crossover probability.
    """

    n_parents: int = 2
    n_children: int = 2
    prob: float = 1.0

    @abstractmethod
    def crossover(
        self,
        parent: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (n_parents, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Lower and upper bounds for each variable. ``None`` means unbounded.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (n_offspring, dim)
        """
        pass

    def crossover_batch(
        self,
        parents_batch: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray | None:
        """
        Execute crossover on a batch of parent pairs at once.

        Default implementation returns ``None``, meaning this operator does
        not support batched crossover; the caller should fall back to
        invoking :meth:`crossover` once per pair.

        Parameters
        ----------
        parents_batch : np.ndarray
            Batch of parent groups. shape = (n_pair, n_parents, dim), i.e. a
            batch of what :meth:`crossover` normally receives one instance of
            at a time (``parents_batch[k]`` is what a single
            ``crossover(parent=parents_batch[k], ...)`` call would take as
            ``parent``).
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Lower and upper bounds for each variable. ``None`` means unbounded.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray or None
            Offspring individuals. shape = (n_pair, n_children, dim), or
            ``None`` if batched crossover is unsupported for this particular
            call/shape.

        Notes
        -----
        No ``prob`` gating is performed inside ``crossover_batch``, mirroring
        :meth:`crossover`: the caller is responsible for having already
        filtered ``parents_batch`` down to only the pairs that should undergo
        crossover.
        """
        return None

    def post_crossover(
        self,
        offspring: np.ndarray,
        parents: np.ndarray,
        rng: np.random.Generator,
        ctx: OptimizationState | None = None,
    ) -> np.ndarray:
        """Post-crossover lifecycle hook; override to inject custom processing.

        Parameters
        ----------
        offspring : np.ndarray
            Offspring produced by crossover. shape = (n_children, dim)
        parents : np.ndarray
            Parent individuals. shape = (n_parents, dim)
        rng : np.random.Generator
            Random number generator.
        ctx : OptimizationState or None, optional
            Current optimization context.

        Returns
        -------
        np.ndarray
            Processed offspring. shape = (n_children, dim)
        """
        return offspring

    def with_post(
        self,
        fn: Callable[
            [np.ndarray, np.ndarray, np.random.Generator, OptimizationState | None],
            np.ndarray,
        ],
    ) -> Crossover:
        """Return a copy of this operator with ``fn`` appended to the hook.

        Parameters
        ----------
        fn : callable
            ``fn(offspring, parents, rng, ctx) -> np.ndarray``

        Returns
        -------
        Crossover
            Shallow copy with the hook registered.
        """
        new = copy.copy(self)
        prev = self.post_crossover
        new.post_crossover = lambda offspring, parents, rng, ctx=None: fn(  # type: ignore  # lambda hook; slot type stricter than inferred lambda signature
            prev(offspring, parents, rng, ctx), parents, rng, ctx
        )
        return new


@register()
class CrossoverBLXAlpha(Crossover):
    """
    BLX-alpha crossover operator.

    Attributes
    ----------
    prob : float
        Individual-level crossover probability.
    alpha : float
        Alpha parameter for BLX-alpha crossover. Controls how far outside
        the parents' range offspring can be generated.

    Notes
    -----
    Originates from Eshelman & Schaffer (1993); the primary paper has not
    been obtained and is credited here by name only. The interval
    ``[c_min - alpha*I, c_max + alpha*I]`` formula implemented in
    :meth:`crossover` has been verified against Herrera, Lozano &
    Verdegay (1998), Section 4.3, which restates the BLX-alpha definition
    while surveying real-coded GA operators.

    References
    ----------
    :cite:`herrera1998realcoded`: Herrera, F., Lozano, M., & Verdegay,
    J. L. (1998). Tackling real-coded genetic algorithms: Operators and
    tools for behavioural analysis. *Artificial Intelligence Review*, 12,
    265-319.
    """

    def __init__(self, prob: float, alpha: float):
        """
        Initialize BLX-alpha crossover operator.

        Parameters
        ----------
        prob : float
            Individual-level crossover probability.
        alpha : float
            Alpha parameter for BLX-alpha crossover.
        """
        super().__init__()
        self.prob = prob
        self.alpha = alpha

    def crossover(
        self,
        parent: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute BLX-alpha crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Ignored; BLX-alpha is inherently unbounded.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
        dim = len(p1)
        blend = rng.uniform(-self.alpha, 1 + self.alpha, size=dim)
        c1 = blend * p1 + (1 - blend) * p2
        c2 = (1 - blend) * p1 + blend * p2
        return np.array([c1, c2])


class CrossoverSBX(Crossover):
    """
    Simulated Binary Crossover (SBX) operator.

    When *bounds* are provided and finite, the bounded variant is used,
    which constrains offspring to ``[lb, ub]``.  When *bounds* is ``None``
    or contains infinite values, the unbounded variant is used.

    Attributes
    ----------
    prob : float
        Individual-level crossover probability.
    eta : float
        Distribution index. Larger values produce offspring closer to parents.
    prob_var : float
        Per-variable crossover probability. Default is 0.5.

    References
    ----------
    :cite:`deb1995sbx`: Deb, K., & Agrawal, R. B. (1995). Simulated binary
    crossover for continuous search space. *Complex Systems*, 9(2), 115-148.
    """

    def __init__(self, prob: float, eta: float, *, prob_var: float = 0.5):
        """
        Initialize SBX crossover operator.

        Parameters
        ----------
        prob : float
            Individual-level crossover probability.
        eta : float
            Distribution index.
        prob_var : float, optional
            Per-variable crossover probability, by default 0.5.
        """
        super().__init__()
        self.prob = prob
        self.eta = eta
        self.prob_var = prob_var

    def crossover(
        self,
        parent: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute SBX crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Lower and upper bounds. When provided and finite, the bounded
            SBX variant is applied.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
        dim = len(p1)

        use_bounded = (
            bounds is not None
            and np.all(np.isfinite(bounds[0]))
            and np.all(np.isfinite(bounds[1]))
        )

        if use_bounded:
            lb, ub = bounds  # type: ignore[misc]
            y1 = np.minimum(p1, p2)
            y2 = np.maximum(p1, p2)
            diff = y2 - y1
            separated = diff > 1e-14
            safe_diff = np.where(separated, diff, 1.0)
            u = rng.uniform(0.0, 1.0, size=dim)

            def _beta_q(beta_limit: np.ndarray) -> np.ndarray:
                alpha = 2.0 - beta_limit ** (-(self.eta + 1))
                return np.where(
                    u <= 1.0 / alpha,
                    (alpha * u) ** (1.0 / (self.eta + 1)),
                    (1.0 / (2.0 - alpha * u)) ** (1.0 / (self.eta + 1)),
                )

            beta_q1 = _beta_q(1.0 + 2.0 * (y1 - lb) / safe_diff)
            beta_q2 = _beta_q(1.0 + 2.0 * (ub - y2) / safe_diff)
            o1 = np.clip(0.5 * ((y1 + y2) - beta_q1 * diff), lb, ub)
            o2 = np.clip(0.5 * ((y1 + y2) + beta_q2 * diff), lb, ub)
            # assign the lower/upper-side offspring to c1/c2 with a fresh
            # 50/50 draw per dimension, independent of parent identity
            # (matches DEAP's cxSimulatedBinaryBounded and pymoo's net effect)
            swap = rng.random(size=dim) < 0.5
            c1_sbx = np.where(swap, o2, o1)
            c2_sbx = np.where(swap, o1, o2)
            # skip crossover where parents are identical
            c1_sbx = np.where(separated, c1_sbx, p1)
            c2_sbx = np.where(separated, c2_sbx, p2)
        else:
            u = rng.uniform(0.0, 1.0, size=dim)
            beta_q = np.where(
                u <= 0.5,
                (2.0 * u) ** (1.0 / (self.eta + 1)),
                (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1)),
            )
            mid = 0.5 * (p1 + p2)
            half_diff = 0.5 * beta_q * (p2 - p1)
            c1_sbx = mid - half_diff
            c2_sbx = mid + half_diff

        do_cross = rng.random(size=dim) < self.prob_var
        c1 = np.where(do_cross, c1_sbx, p1)
        c2 = np.where(do_cross, c2_sbx, p2)
        return np.array([c1, c2])

    def crossover_batch(
        self,
        parents_batch: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray | None:
        """
        Execute SBX crossover on a batch of parent pairs at once.

        Mirrors :meth:`crossover`, vectorized over an extra leading
        ``n_pair`` axis; the underlying formula is unchanged, only the
        array shapes grow by one axis.

        Parameters
        ----------
        parents_batch : np.ndarray
            Batch of parent pairs. shape = (n_pair, 2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Lower and upper bounds. When provided and finite, the bounded
            SBX variant is applied.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (n_pair, 2, dim)

        Notes
        -----
        For ``n_pair > 1``, per-row results are **not** bit-identical to
        calling :meth:`crossover` once per pair in a loop with the same
        seeded ``rng``: drawing e.g. ``u = rng.uniform(size=(n_pair, dim))``
        for the whole batch in one call is not the same draw order as
        looping ``u = rng.uniform(size=dim)`` once per pair. Only
        ``n_pair == 1`` is guaranteed to match a single :meth:`crossover`
        call with an identically-seeded ``rng``.
        """
        p1 = parents_batch[:, 0, :]
        p2 = parents_batch[:, 1, :]
        dim = p1.shape[1]
        n_pair = p1.shape[0]

        use_bounded = (
            bounds is not None
            and np.all(np.isfinite(bounds[0]))
            and np.all(np.isfinite(bounds[1]))
        )

        if use_bounded:
            lb, ub = bounds  # type: ignore[misc]
            y1 = np.minimum(p1, p2)
            y2 = np.maximum(p1, p2)
            diff = y2 - y1
            separated = diff > 1e-14
            safe_diff = np.where(separated, diff, 1.0)
            u = rng.uniform(0.0, 1.0, size=(n_pair, dim))

            def _beta_q(beta_limit: np.ndarray) -> np.ndarray:
                alpha = 2.0 - beta_limit ** (-(self.eta + 1))
                return np.where(
                    u <= 1.0 / alpha,
                    (alpha * u) ** (1.0 / (self.eta + 1)),
                    (1.0 / (2.0 - alpha * u)) ** (1.0 / (self.eta + 1)),
                )

            beta_q1 = _beta_q(1.0 + 2.0 * (y1 - lb) / safe_diff)
            beta_q2 = _beta_q(1.0 + 2.0 * (ub - y2) / safe_diff)
            o1 = np.clip(0.5 * ((y1 + y2) - beta_q1 * diff), lb, ub)
            o2 = np.clip(0.5 * ((y1 + y2) + beta_q2 * diff), lb, ub)
            # assign the lower/upper-side offspring to c1/c2 with a fresh
            # 50/50 draw per dimension, independent of parent identity
            # (matches DEAP's cxSimulatedBinaryBounded and pymoo's net effect)
            swap = rng.random(size=(n_pair, dim)) < 0.5
            c1_sbx = np.where(swap, o2, o1)
            c2_sbx = np.where(swap, o1, o2)
            # skip crossover where parents are identical
            c1_sbx = np.where(separated, c1_sbx, p1)
            c2_sbx = np.where(separated, c2_sbx, p2)
        else:
            u = rng.uniform(0.0, 1.0, size=(n_pair, dim))
            beta_q = np.where(
                u <= 0.5,
                (2.0 * u) ** (1.0 / (self.eta + 1)),
                (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1)),
            )
            mid = 0.5 * (p1 + p2)
            half_diff = 0.5 * beta_q * (p2 - p1)
            c1_sbx = mid - half_diff
            c2_sbx = mid + half_diff

        do_cross = rng.random(size=(n_pair, dim)) < self.prob_var
        c1 = np.where(do_cross, c1_sbx, p1)
        c2 = np.where(do_cross, c2_sbx, p2)
        return np.stack([c1, c2], axis=1)


class CrossoverUniform(Crossover):
    """
    Uniform crossover operator.

    Each dimension is independently swapped between parents with probability
    ``swap_rate``.

    Attributes
    ----------
    prob : float
        Individual-level crossover probability.
    swap_rate : float
        Per-dimension swap probability. Default is 0.5.

    Notes
    -----
    Originates from Syswerda (1989); the paper is a scanned document whose
    content could not be machine-verified, so it is credited here by name
    only.
    """

    def __init__(self, prob: float, swap_rate: float = 0.5):
        """
        Initialize uniform crossover operator.

        Parameters
        ----------
        prob : float
            Individual-level crossover probability.
        swap_rate : float, optional
            Per-dimension swap probability, by default 0.5.
        """
        super().__init__()
        self.prob = prob
        self.swap_rate = swap_rate

    def crossover(
        self,
        parent: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute uniform crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Ignored.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
        mask = rng.random(len(p1)) < self.swap_rate
        c1 = np.where(mask, p2, p1)
        c2 = np.where(mask, p1, p2)
        return np.array([c1, c2])

    def crossover_batch(
        self,
        parents_batch: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray | None:
        """
        Execute uniform crossover on a batch of parent pairs at once.

        Mirrors :meth:`crossover`, vectorized over an extra leading
        ``n_pair`` axis.

        Parameters
        ----------
        parents_batch : np.ndarray
            Batch of parent pairs. shape = (n_pair, 2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Ignored.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (n_pair, 2, dim)

        Notes
        -----
        For ``n_pair > 1``, per-row results are **not** bit-identical to
        calling :meth:`crossover` once per pair in a loop with the same
        seeded ``rng``: drawing ``rng.random(size=(n_pair, dim))`` for the
        whole batch in one call is not the same draw order as looping
        ``rng.random(dim)`` once per pair. Only ``n_pair == 1`` is
        guaranteed to match a single :meth:`crossover` call with an
        identically-seeded ``rng``.
        """
        p1 = parents_batch[:, 0, :]
        p2 = parents_batch[:, 1, :]
        n_pair, dim = p1.shape
        mask = rng.random(size=(n_pair, dim)) < self.swap_rate
        c1 = np.where(mask, p2, p1)
        c2 = np.where(mask, p1, p2)
        return np.stack([c1, c2], axis=1)


class CrossoverOnePoint(Crossover):
    """
    One-point crossover operator.

    Attributes
    ----------
    prob : float
        Individual-level crossover probability.
    """

    def __init__(self, prob: float):
        """
        Initialize one-point crossover operator.

        Parameters
        ----------
        prob : float
            Individual-level crossover probability.
        """
        super().__init__()
        self.prob = prob

    def crossover(
        self,
        parent: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute one-point crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Ignored.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
        dim = len(p1)
        point = rng.integers(1, dim)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return np.array([c1, c2])


class CrossoverTwoPoint(Crossover):
    """
    Two-point crossover operator.

    Attributes
    ----------
    prob : float
        Individual-level crossover probability.
    """

    def __init__(self, prob: float):
        """
        Initialize two-point crossover operator.

        Parameters
        ----------
        prob : float
            Individual-level crossover probability.
        """
        super().__init__()
        self.prob = prob

    def crossover(
        self,
        parent: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute two-point crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Ignored.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
        dim = len(p1)
        pts = np.sort(rng.choice(dim + 1, size=2, replace=False))
        pt1, pt2 = pts[0], pts[1]
        c1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]])
        c2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]])
        return np.array([c1, c2])


class CrossoverIntegerSBX(Crossover):
    """
    Simulated Binary Crossover (SBX) for integer-valued dimensions.

    Applies SBX then rounds offspring to the nearest integer.

    Attributes
    ----------
    prob : float
        Individual-level crossover probability.
    eta : float
        Distribution index. Larger values produce offspring closer to parents.
    prob_var : float
        Per-variable crossover probability. Default is 0.5.

    References
    ----------
    :cite:`deb1995sbx`: Deb, K., & Agrawal, R. B. (1995). Simulated binary
    crossover for continuous search space. *Complex Systems*, 9(2), 115-148.
    """

    def __init__(self, prob: float, eta: float, *, prob_var: float = 0.5):
        """
        Initialize integer SBX crossover operator.

        Parameters
        ----------
        prob : float
            Individual-level crossover probability.
        eta : float
            Distribution index.
        prob_var : float, optional
            Per-variable crossover probability, by default 0.5.
        """
        super().__init__()
        self.prob = prob
        self.eta = eta
        self.prob_var = prob_var

    def crossover(
        self,
        parent: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute integer SBX crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Lower and upper bounds. When provided and finite, the bounded
            SBX variant is applied before rounding.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals with integer values. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
        dim = len(p1)

        use_bounded = (
            bounds is not None
            and np.all(np.isfinite(bounds[0]))
            and np.all(np.isfinite(bounds[1]))
        )

        if use_bounded:
            lb, ub = bounds  # type: ignore[misc]
            y1 = np.minimum(p1, p2)
            y2 = np.maximum(p1, p2)
            diff = y2 - y1
            separated = diff > 1e-14
            safe_diff = np.where(separated, diff, 1.0)
            u = rng.uniform(0.0, 1.0, size=dim)

            def _beta_q(beta_limit: np.ndarray) -> np.ndarray:
                alpha = 2.0 - beta_limit ** (-(self.eta + 1))
                return np.where(
                    u <= 1.0 / alpha,
                    (alpha * u) ** (1.0 / (self.eta + 1)),
                    (1.0 / (2.0 - alpha * u)) ** (1.0 / (self.eta + 1)),
                )

            beta_q1 = _beta_q(1.0 + 2.0 * (y1 - lb) / safe_diff)
            beta_q2 = _beta_q(1.0 + 2.0 * (ub - y2) / safe_diff)
            o1 = np.clip(np.round(0.5 * ((y1 + y2) - beta_q1 * diff)), lb, ub)
            o2 = np.clip(np.round(0.5 * ((y1 + y2) + beta_q2 * diff)), lb, ub)
            swap = rng.random(size=dim) < 0.5
            c1_sbx = np.where(swap, o2, o1)
            c2_sbx = np.where(swap, o1, o2)
            c1_sbx = np.where(separated, c1_sbx, p1)
            c2_sbx = np.where(separated, c2_sbx, p2)
        else:
            u = rng.uniform(0.0, 1.0, size=dim)
            beta_q = np.where(
                u <= 0.5,
                (2.0 * u) ** (1.0 / (self.eta + 1)),
                (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1)),
            )
            mid = 0.5 * (p1 + p2)
            half_diff = 0.5 * beta_q * (p2 - p1)
            c1_sbx = np.round(mid - half_diff)
            c2_sbx = np.round(mid + half_diff)

        do_cross = rng.random(size=dim) < self.prob_var
        c1 = np.where(do_cross, c1_sbx, p1)
        c2 = np.where(do_cross, c2_sbx, p2)
        return np.array([c1, c2])

    def crossover_batch(
        self,
        parents_batch: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray | None:
        """
        Execute integer SBX crossover on a batch of parent pairs at once.

        Mirrors :meth:`crossover`, vectorized over an extra leading
        ``n_pair`` axis; the underlying formula (including the rounding to
        the nearest integer) is unchanged, only the array shapes grow by
        one axis.

        Parameters
        ----------
        parents_batch : np.ndarray
            Batch of parent pairs. shape = (n_pair, 2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Lower and upper bounds. When provided and finite, the bounded
            SBX variant is applied before rounding.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals with integer values. shape = (n_pair, 2, dim)

        Notes
        -----
        For ``n_pair > 1``, per-row results are **not** bit-identical to
        calling :meth:`crossover` once per pair in a loop with the same
        seeded ``rng``: drawing e.g. ``u = rng.uniform(size=(n_pair, dim))``
        for the whole batch in one call is not the same draw order as
        looping ``u = rng.uniform(size=dim)`` once per pair. Only
        ``n_pair == 1`` is guaranteed to match a single :meth:`crossover`
        call with an identically-seeded ``rng``.
        """
        p1 = parents_batch[:, 0, :]
        p2 = parents_batch[:, 1, :]
        n_pair, dim = p1.shape

        use_bounded = (
            bounds is not None
            and np.all(np.isfinite(bounds[0]))
            and np.all(np.isfinite(bounds[1]))
        )

        if use_bounded:
            lb, ub = bounds  # type: ignore[misc]
            y1 = np.minimum(p1, p2)
            y2 = np.maximum(p1, p2)
            diff = y2 - y1
            separated = diff > 1e-14
            safe_diff = np.where(separated, diff, 1.0)
            u = rng.uniform(0.0, 1.0, size=(n_pair, dim))

            def _beta_q(beta_limit: np.ndarray) -> np.ndarray:
                alpha = 2.0 - beta_limit ** (-(self.eta + 1))
                return np.where(
                    u <= 1.0 / alpha,
                    (alpha * u) ** (1.0 / (self.eta + 1)),
                    (1.0 / (2.0 - alpha * u)) ** (1.0 / (self.eta + 1)),
                )

            beta_q1 = _beta_q(1.0 + 2.0 * (y1 - lb) / safe_diff)
            beta_q2 = _beta_q(1.0 + 2.0 * (ub - y2) / safe_diff)
            o1 = np.clip(np.round(0.5 * ((y1 + y2) - beta_q1 * diff)), lb, ub)
            o2 = np.clip(np.round(0.5 * ((y1 + y2) + beta_q2 * diff)), lb, ub)
            swap = rng.random(size=(n_pair, dim)) < 0.5
            c1_sbx = np.where(swap, o2, o1)
            c2_sbx = np.where(swap, o1, o2)
            c1_sbx = np.where(separated, c1_sbx, p1)
            c2_sbx = np.where(separated, c2_sbx, p2)
        else:
            u = rng.uniform(0.0, 1.0, size=(n_pair, dim))
            beta_q = np.where(
                u <= 0.5,
                (2.0 * u) ** (1.0 / (self.eta + 1)),
                (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1)),
            )
            mid = 0.5 * (p1 + p2)
            half_diff = 0.5 * beta_q * (p2 - p1)
            c1_sbx = np.round(mid - half_diff)
            c2_sbx = np.round(mid + half_diff)

        do_cross = rng.random(size=(n_pair, dim)) < self.prob_var
        c1 = np.where(do_cross, c1_sbx, p1)
        c2 = np.where(do_cross, c2_sbx, p2)
        return np.stack([c1, c2], axis=1)


class CrossoverCategorical(Crossover):
    """
    Uniform crossover for categorical dimensions.

    Each dimension independently inherits one parent's value with equal
    probability (50/50).  Offspring are always exact copies of a parent
    value, preserving the validity of categorical indices.

    Attributes
    ----------
    prob : float
        Individual-level crossover probability.
    """

    def __init__(self, prob: float):
        """
        Initialize categorical crossover operator.

        Parameters
        ----------
        prob : float
            Individual-level crossover probability.
        """
        super().__init__()
        self.prob = prob

    def crossover(
        self,
        parent: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray:
        """
        Execute categorical crossover.

        Parameters
        ----------
        parent : np.ndarray
            Parent individuals. shape = (2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Ignored.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (2, dim)
        """
        p1 = parent[0]
        p2 = parent[1]
        mask = rng.random(len(p1)) < 0.5
        c1 = np.where(mask, p2, p1)
        c2 = np.where(mask, p1, p2)
        return np.array([c1, c2])

    def crossover_batch(
        self,
        parents_batch: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> np.ndarray | None:
        """
        Execute categorical crossover on a batch of parent pairs at once.

        Mirrors :meth:`crossover`, vectorized over an extra leading
        ``n_pair`` axis.

        Parameters
        ----------
        parents_batch : np.ndarray
            Batch of parent pairs. shape = (n_pair, 2, dim)
        bounds : tuple of (np.ndarray, np.ndarray) or None
            Ignored.
        rng : np.random.Generator, optional
            Random number generator, by default np.random.default_rng()

        Returns
        -------
        np.ndarray
            Offspring individuals. shape = (n_pair, 2, dim)

        Notes
        -----
        For ``n_pair > 1``, per-row results are **not** bit-identical to
        calling :meth:`crossover` once per pair in a loop with the same
        seeded ``rng``: drawing ``rng.random(size=(n_pair, dim))`` for the
        whole batch in one call is not the same draw order as looping
        ``rng.random(dim)`` once per pair. Only ``n_pair == 1`` is
        guaranteed to match a single :meth:`crossover` call with an
        identically-seeded ``rng``.
        """
        p1 = parents_batch[:, 0, :]
        p2 = parents_batch[:, 1, :]
        n_pair, dim = p1.shape
        mask = rng.random(size=(n_pair, dim)) < 0.5
        c1 = np.where(mask, p2, p1)
        c2 = np.where(mask, p1, p2)
        return np.stack([c1, c2], axis=1)
