"""Particle Swarm Optimization module."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from saealib.algorithms.base import Algorithm
from saealib.callback import PostAskEvent
from saealib.context import OptimizationContext
from saealib.operators.repair import repair_clipping
from saealib.population import Archive, Population, PopulationAttribute
from saealib.problem import Problem

if TYPE_CHECKING:
    from saealib.optimizer import ComponentProvider

RepairFunc = Callable[[np.ndarray, tuple[np.ndarray, np.ndarray]], np.ndarray]


class PSO(Algorithm):
    """
    Particle Swarm Optimization (PSO) for single-objective problems.

    The global best (leader) is selected from the personal bests of all
    particles using ``ctx.comparator``, so the ranking adapts automatically
    to any single-objective Comparator.

    Multi-objective PSO (MOPSO) requires a dedicated subclass with a
    separate Pareto archive for leader selection.

    Attributes
    ----------
    w : float
        Inertia weight.
    c1 : float
        Cognitive coefficient (personal best attraction).
    c2 : float
        Social coefficient (global best attraction).
    v_max : float or None
        Maximum velocity magnitude per dimension. ``None`` disables clamping.
    repair : RepairFunc or None
        Repair function applied after position update.
    """

    def __init__(
        self,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max: float | None = None,
        repair: RepairFunc | None = repair_clipping,
    ):
        """
        Initialize PSO.

        Parameters
        ----------
        w : float, optional
            Inertia weight. Defaults to 0.7.
        c1 : float, optional
            Cognitive coefficient. Defaults to 1.5.
        c2 : float, optional
            Social coefficient. Defaults to 1.5.
        v_max : float or None, optional
            Maximum velocity per dimension. Defaults to ``None`` (no clamping).
        repair : RepairFunc or None, optional
            Repair function applied after position update.
            Defaults to ``repair_clipping``. Pass ``None`` to disable repair.
        """
        super().__init__()
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.repair = repair

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """
        Return PSO-specific attributes required by the Population.

        Parameters
        ----------
        problem : Problem
            The Problem object being referenced.

        Returns
        -------
        list[PopulationAttribute]
            velocity, pbest_x, pbest_f, and pbest_cv attributes.
        """
        dim = problem.dim
        n_obj = problem.n_obj
        return [
            PopulationAttribute(
                name="velocity", dtype=np.float64, shape=(dim,), default=0.0
            ),
            PopulationAttribute(
                name="pbest_x", dtype=np.float64, shape=(dim,), default=np.nan
            ),
            PopulationAttribute(
                name="pbest_f", dtype=np.float64, shape=(n_obj,), default=np.nan
            ),
            PopulationAttribute(
                name="pbest_cv", dtype=np.float64, shape=(), default=np.nan
            ),
        ]

    @property
    def population_class(self):
        """Return the population class."""
        return Population

    @property
    def archive_class(self):
        """Return the archive class."""
        return Archive

    def ask(
        self,
        ctx: OptimizationContext,
        provider: ComponentProvider,
        n_offspring: int | None = None,
    ) -> Population:
        """
        Update particle velocities and positions.

        Parameters
        ----------
        ctx : OptimizationContext
            Current optimization context.
        provider : ComponentProvider
            Component provider.
        n_offspring : int or None, optional
            Ignored; output count equals population size.

        Returns
        -------
        Population
            Candidates with updated ``x`` and ``velocity``.
        """
        pop = ctx.population
        popsize = len(pop)
        lb = ctx.problem.lb
        ub = ctx.problem.ub

        x = pop.get_array("x").copy()
        f = pop.get_array("f").copy()
        cv = pop.get_array("cv").copy()
        v = pop.get_array("velocity").copy()
        pbest_x = pop.get_array("pbest_x").copy()
        pbest_f = pop.get_array("pbest_f").copy()
        pbest_cv = pop.get_array("pbest_cv").copy()

        # Initialize pbest for particles that have not yet been evaluated.
        # pbest_f has shape (popsize, n_obj); NaN in any objective means uninitialised.
        uninit = np.isnan(pbest_f).any(axis=-1)
        pbest_x[uninit] = x[uninit]
        pbest_f[uninit] = f[uninit]
        pbest_cv[uninit] = cv[uninit]

        leader = self._select_leader(ctx, pbest_x, pbest_f, pbest_cv)

        r1 = ctx.rng.uniform(0.0, 1.0, size=(popsize, ctx.dim))
        r2 = ctx.rng.uniform(0.0, 1.0, size=(popsize, ctx.dim))

        v_new = self.w * v + self.c1 * r1 * (pbest_x - x) + self.c2 * r2 * (leader - x)

        if self.v_max is not None:
            v_new = np.clip(v_new, -self.v_max, self.v_max)

        x_new = x + v_new
        if self.repair is not None:
            x_new = self.repair(x_new, (lb, ub))

        cand_pop = pop.empty_like(capacity=popsize)
        cand_pop.extend(
            {
                "x": x_new,
                "f": np.full((popsize, ctx.n_obj), np.nan),
                "g": np.zeros((popsize, ctx.problem.n_constraints)),
                "cv": np.zeros(popsize),
                "velocity": v_new,
                "pbest_x": pbest_x,
                "pbest_f": pbest_f,
                "pbest_cv": pbest_cv,
            }
        )

        post_ask = PostAskEvent(ctx=ctx, provider=provider, candidates=x_new)
        provider.dispatch(post_ask)

        return cand_pop

    def tell(
        self,
        ctx: OptimizationContext,
        provider: ComponentProvider,
        offspring: Population,
    ) -> None:
        """
        Update the population and personal bests from evaluated offspring.

        Parameters
        ----------
        ctx : OptimizationContext
            Current optimization context.
        provider : ComponentProvider
            Component provider.
        offspring : Population
            Evaluated offspring population.
        """
        x_new = offspring.get_array("x")
        f_new = offspring.get_array("f")
        g_new = offspring.get_array("g")
        cv_new = offspring.get_array("cv")
        v_new = offspring.get_array("velocity")
        pbest_x = offspring.get_array("pbest_x").copy()
        pbest_f = offspring.get_array("pbest_f").copy()
        pbest_cv = offspring.get_array("pbest_cv").copy()

        popsize = len(offspring)
        cmp = ctx.comparator
        for i in range(popsize):
            if np.any(np.isnan(f_new[i])):
                continue
            if (
                np.any(np.isnan(pbest_f[i]))
                or cmp.compare(f_new[i], cv_new[i], pbest_f[i], pbest_cv[i]) == -1
            ):
                pbest_x[i] = x_new[i]
                pbest_f[i] = f_new[i]
                pbest_cv[i] = cv_new[i]

        ctx.population.clear()
        ctx.population.extend(
            {
                "x": x_new,
                "f": f_new,
                "g": g_new,
                "cv": cv_new,
                "velocity": v_new,
                "pbest_x": pbest_x,
                "pbest_f": pbest_f,
                "pbest_cv": pbest_cv,
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_leader(
        self,
        ctx: OptimizationContext,
        pbest_x: np.ndarray,
        pbest_f: np.ndarray,
        pbest_cv: np.ndarray,
    ) -> np.ndarray:
        """
        Select the global-best leader from the personal bests.

        The personal bests are sorted by ``ctx.comparator`` and the
        top-ranked particle's position is returned as the leader.

        Parameters
        ----------
        ctx : OptimizationContext
            Optimization context.
        pbest_x : np.ndarray
            Personal best positions, shape (popsize, dim).
        pbest_f : np.ndarray
            Personal best objective values, shape (popsize, n_obj).
        pbest_cv : np.ndarray
            Personal best constraint violations, shape (popsize,).

        Returns
        -------
        np.ndarray
            Leader position, shape (dim,).
        """
        popsize = len(pbest_x)
        pbest_pop = ctx.population.empty_like(capacity=popsize)
        pbest_pop.extend(
            {
                "x": pbest_x,
                "f": pbest_f,
                "g": np.zeros((popsize, ctx.problem.n_constraints)),
                "cv": pbest_cv,
                "velocity": np.zeros((popsize, ctx.dim)),
                "pbest_x": pbest_x,
                "pbest_f": pbest_f,
                "pbest_cv": pbest_cv,
            }
        )
        sorted_idx = ctx.comparator.sort_population(pbest_pop)
        return pbest_x[sorted_idx[0]]
