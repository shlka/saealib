"""Genetic Algorithm module."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from saealib.algorithms.base import Algorithm
from saealib.callback import PostAskEvent, PostCrossoverEvent, PostMutationEvent
from saealib.context import OptimizationState
from saealib.exceptions import ConfigurationError
from saealib.operators.crossover import CrossoverCategorical, CrossoverIntegerSBX
from saealib.operators.dedup import DuplicateElimination
from saealib.operators.mutation import MutationCategorical, MutationIntegerUniform
from saealib.population import Archive, Population, PopulationAttribute
from saealib.problem import Problem
from saealib.registry import register

if TYPE_CHECKING:
    from saealib.operators.crossover import Crossover
    from saealib.operators.mutation import Mutation
    from saealib.operators.selection import ParentSelection, SurvivorSelection
    from saealib.optimizer import Dispatchable


def _route_crossover(
    parent: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    problem: Problem,
    cont_op: Crossover,
    int_op: Crossover,
    cat_op: Crossover,
) -> np.ndarray:
    """Apply per-type crossover and reassemble offspring."""
    i_mask = problem.integer_mask
    cat_mask = problem.categorical_mask
    if not i_mask.any() and not cat_mask.any():
        return cont_op.crossover(parent, (lb, ub), rng=rng)

    n_children = cont_op.n_children
    dim = parent.shape[1]
    offspring = np.empty((n_children, dim))
    c_mask = problem.continuous_mask

    if c_mask.any():
        offspring[:, c_mask] = cont_op.crossover(
            parent[:, c_mask], (lb[c_mask], ub[c_mask]), rng=rng
        )
    if i_mask.any():
        offspring[:, i_mask] = int_op.crossover(
            parent[:, i_mask], (lb[i_mask], ub[i_mask]), rng=rng
        )
    if cat_mask.any():
        offspring[:, cat_mask] = cat_op.crossover(
            parent[:, cat_mask], (lb[cat_mask], ub[cat_mask]), rng=rng
        )

    return offspring


def _route_mutation(
    p: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    problem: Problem,
    cont_op: Mutation,
    int_op: Mutation,
    cat_op: Mutation,
) -> np.ndarray:
    """Apply per-type mutation and reassemble offspring."""
    i_mask = problem.integer_mask
    cat_mask = problem.categorical_mask
    if not i_mask.any() and not cat_mask.any():
        return cont_op.mutate(p, (lb, ub), rng=rng)

    result = p.copy()
    c_mask = problem.continuous_mask

    if c_mask.any():
        result[c_mask] = cont_op.mutate(p[c_mask], (lb[c_mask], ub[c_mask]), rng=rng)
    if i_mask.any():
        result[i_mask] = int_op.mutate(p[i_mask], (lb[i_mask], ub[i_mask]), rng=rng)
    if cat_mask.any():
        result[cat_mask] = cat_op.mutate(
            p[cat_mask], (lb[cat_mask], ub[cat_mask]), rng=rng
        )

    return result


@register()
class GA(Algorithm):
    """
    Genetic Algorithm class.

    Attributes
    ----------
    crossover : Crossover
        Crossover operator for continuous dimensions.
    mutation : Mutation
        Mutation operator for continuous dimensions.
    parent_selection : ParentSelection
        Parent selection operator.
    survivor_selection : SurvivorSelection
        Survivor selection operator.
    integer_crossover : Crossover
        Crossover operator for integer dimensions.
    integer_mutation : Mutation
        Mutation operator for integer dimensions.
    categorical_crossover : Crossover
        Crossover operator for categorical dimensions.
    categorical_mutation : Mutation
        Mutation operator for categorical dimensions.
    duplicate_elimination : DuplicateElimination or None
        When set, offspring that duplicate any member of the current population
        are replaced by re-generated candidates (up to ``max_retries`` attempts).
        ``None`` disables duplicate elimination (default behaviour).
    """

    def __init__(
        self,
        crossover: Crossover,
        mutation: Mutation,
        parent_selection: ParentSelection,
        survivor_selection: SurvivorSelection,
        *,
        duplicate_elimination: DuplicateElimination | None = None,
        integer_crossover: Crossover | None = None,
        integer_mutation: Mutation | None = None,
        categorical_crossover: Crossover | None = None,
        categorical_mutation: Mutation | None = None,
    ):
        """
        Initialize GA (Genetic Algorithm) class.

        Parameters
        ----------
        crossover : Crossover
            Crossover operator for continuous dimensions.
        mutation : Mutation
            Mutation operator for continuous dimensions.
        parent_selection : ParentSelection
            Parent selection operator.
        survivor_selection : SurvivorSelection
            Survivor selection operator.
        duplicate_elimination : DuplicateElimination, optional
            When provided, offspring that duplicate any member of the current
            population are replaced by re-generated candidates.  ``None``
            (default) disables duplicate elimination.
        integer_crossover : Crossover, optional
            Crossover operator for integer dimensions.
            Defaults to ``CrossoverIntegerSBX`` with the same rate as *crossover*.
        integer_mutation : Mutation, optional
            Mutation operator for integer dimensions.
            Defaults to ``MutationIntegerUniform`` with the same rate as *mutation*.
        categorical_crossover : Crossover, optional
            Crossover operator for categorical dimensions.
            Defaults to ``CrossoverCategorical`` with the same rate as *crossover*.
        categorical_mutation : Mutation, optional
            Mutation operator for categorical dimensions.
            Defaults to ``MutationCategorical`` with the same rate as *mutation*.
        """
        super().__init__()
        self.crossover = crossover
        self.mutation = mutation
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection
        self.duplicate_elimination = duplicate_elimination

        _cr = getattr(crossover, "prob", 1.0)
        _pv = getattr(mutation, "prob_var", None)
        self.integer_crossover: Crossover = (
            integer_crossover
            if integer_crossover is not None
            else CrossoverIntegerSBX(_cr, eta=20.0)
        )
        self.integer_mutation: Mutation = (
            integer_mutation
            if integer_mutation is not None
            else MutationIntegerUniform(prob_var=_pv)
        )
        self.categorical_crossover: Crossover = (
            categorical_crossover
            if categorical_crossover is not None
            else CrossoverCategorical(_cr)
        )
        self.categorical_mutation: Mutation = (
            categorical_mutation
            if categorical_mutation is not None
            else MutationCategorical(prob_var=_pv)
        )

        for _name, _op in [
            ("integer_crossover", self.integer_crossover),
            ("categorical_crossover", self.categorical_crossover),
        ]:
            if _op.n_children != self.crossover.n_children:
                raise ConfigurationError(
                    f"{_name}.n_children={_op.n_children} must equal "
                    f"crossover.n_children={self.crossover.n_children} "
                    "for mixed-variable routing"
                )
            if _op.n_parents != self.crossover.n_parents:
                raise ConfigurationError(
                    f"{_name}.n_parents={_op.n_parents} must equal "
                    f"crossover.n_parents={self.crossover.n_parents} "
                    "for mixed-variable routing"
                )

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Return algorithm-specific attributes (GA needs none beyond the defaults)."""
        return []

    @property
    def population_class(self):
        """Return the population class."""
        return Population

    @property
    def archive_class(self):
        """Return the archive class."""
        return Archive

    @property
    def ask_notation(self) -> list[str]:
        """LaTeX notation lines for GA.ask(): select → crossover → mutate."""
        return [
            r"$I_m \leftarrow \mathrm{select}(P,\, n_{pair})$",
            r"$\mathcal{Q} \leftarrow \mathrm{crossover}(P[I_m])$",
            r"$\mathcal{Q} \leftarrow \mathrm{mutate}(\mathcal{Q})$",
        ]

    @property
    def tell_notation(self) -> list[str]:
        r"""LaTeX notation lines for GA.tell(): $(\mu + \lambda)$ survivor selection."""
        return [
            r"$P \leftarrow \mathrm{select}_{(\mu+\lambda)}"
            r"(P \cup \mathcal{Q},\, \mu)$",
        ]

    def ask(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        n_offspring: int | None = None,
    ) -> Population:
        """
        Generate offspring via crossover and mutation.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : Dispatchable
            Component provider.
        n_offspring : int or None, optional
            Number of offspring. Defaults to the current population size.

        Returns
        -------
        Population
        """
        # Re-validate per-type crossover consistency here because operators may be
        # replaced or mutated after __init__, bypassing constructor checks.
        if ctx.problem.integer_mask.any() or ctx.problem.categorical_mask.any():
            for _name, _op in [
                ("integer_crossover", self.integer_crossover),
                ("categorical_crossover", self.categorical_crossover),
            ]:
                if _op.n_children != self.crossover.n_children:
                    raise ConfigurationError(
                        f"{_name}.n_children={_op.n_children} must equal "
                        f"crossover.n_children={self.crossover.n_children} "
                        "for mixed-variable routing"
                    )
                if _op.n_parents != self.crossover.n_parents:
                    raise ConfigurationError(
                        f"{_name}.n_parents={_op.n_parents} must equal "
                        f"crossover.n_parents={self.crossover.n_parents} "
                        "for mixed-variable routing"
                    )

        pop = ctx.population.get_array("x")
        popsize = len(pop)
        target = n_offspring if n_offspring is not None else popsize
        lb = ctx.problem.lb
        ub = ctx.problem.ub
        n_children = self.crossover.n_children
        n_pair = math.ceil(target / n_children)
        parent_idx_m = (
            self.parent_selection.select(
                ctx,
                ctx.population,
                n_pair=n_pair,
                n_parents=self.crossover.n_parents,
                rng=ctx.rng,
            )
            % popsize
        )
        handler = ctx.problem.handler
        constraints = ctx.problem.constraints

        cand = np.empty((n_pair * n_children, ctx.dim))
        for i in range(n_pair):
            parent = pop[parent_idx_m[i]]
            if ctx.rng.random() < self.crossover.prob:
                c = _route_crossover(
                    parent,
                    lb,
                    ub,
                    ctx.rng,
                    ctx.problem,
                    self.crossover,
                    self.integer_crossover,
                    self.categorical_crossover,
                )
            else:
                c = parent[:n_children].copy()
            c = self.crossover.post_crossover(c, parent, ctx.rng, ctx)
            cand[i * n_children : (i + 1) * n_children] = c
        for i in range(len(cand)):
            cand[i] = handler.repair(cand[i], constraints, lb, ub)
            cand[i] = ctx.problem.repair(cand[i])
        provider.dispatch(PostCrossoverEvent(ctx=ctx, candidates=cand))

        cand_len = len(cand)
        for i in range(cand_len):
            cand[i] = _route_mutation(
                cand[i],
                lb,
                ub,
                ctx.rng,
                ctx.problem,
                self.mutation,
                self.integer_mutation,
                self.categorical_mutation,
            )
            cand[i] = self.mutation.post_mutation(cand[i], (lb, ub), ctx.rng, ctx)
            cand[i] = handler.repair(cand[i], constraints, lb, ub)
            cand[i] = ctx.problem.repair(cand[i])
        provider.dispatch(PostMutationEvent(ctx=ctx, candidates=cand))

        if self.duplicate_elimination is not None:
            pop_x = ctx.population.get_array("x")
            de = self.duplicate_elimination
            for _ in range(de.max_retries):
                dup = de.find_duplicates(cand[:target], pop_x)
                if not dup.any():
                    break
                dup_idx = np.where(dup)[0]
                repl = self._make_offspring(
                    ctx, len(dup_idx), pop, popsize, lb, ub, handler, constraints
                )
                cand[dup_idx] = repl[: len(dup_idx)]

        provider.dispatch(PostAskEvent(ctx=ctx, candidates=cand))

        cand_pop = ctx.population.empty_like(capacity=target)
        cand_pop.extend({"x": cand[:target]})
        return cand_pop

    def _make_offspring(
        self,
        ctx: OptimizationState,
        n_target: int,
        pop: np.ndarray,
        popsize: int,
        lb: np.ndarray,
        ub: np.ndarray,
        handler,
        constraints,
    ) -> np.ndarray:
        """Generate *n_target* offspring without dispatching events.

        Used exclusively by the duplicate-elimination retry loop in
        :meth:`ask` to silently replace duplicate candidates.
        """
        n_children = self.crossover.n_children
        n_pair = math.ceil(n_target / n_children)
        parent_idx = (
            self.parent_selection.select(
                ctx,
                ctx.population,
                n_pair=n_pair,
                n_parents=self.crossover.n_parents,
                rng=ctx.rng,
            )
            % popsize
        )
        batch = np.empty((n_pair * n_children, ctx.dim))
        for i in range(n_pair):
            parent = pop[parent_idx[i]]
            if ctx.rng.random() < self.crossover.prob:
                c = _route_crossover(
                    parent,
                    lb,
                    ub,
                    ctx.rng,
                    ctx.problem,
                    self.crossover,
                    self.integer_crossover,
                    self.categorical_crossover,
                )
            else:
                c = parent[:n_children].copy()
            c = self.crossover.post_crossover(c, parent, ctx.rng, ctx)
            batch[i * n_children : (i + 1) * n_children] = c
        for i in range(len(batch)):
            batch[i] = handler.repair(batch[i], constraints, lb, ub)
            batch[i] = ctx.problem.repair(batch[i])
        for i in range(len(batch)):
            batch[i] = _route_mutation(
                batch[i],
                lb,
                ub,
                ctx.rng,
                ctx.problem,
                self.mutation,
                self.integer_mutation,
                self.categorical_mutation,
            )
            batch[i] = self.mutation.post_mutation(batch[i], (lb, ub), ctx.rng, ctx)
            batch[i] = handler.repair(batch[i], constraints, lb, ub)
            batch[i] = ctx.problem.repair(batch[i])
        return batch

    def tell(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        offspring: Population,
    ):
        """
        Update the population using (μ+λ) survivor selection.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : Dispatchable
            Component provider.
        offspring : Population
            Offspring population.
        """
        popsize = len(ctx.population)

        pool = ctx.population.empty_like(capacity=popsize + len(offspring))
        pool.extend(ctx.population)
        pool.extend(offspring)

        survivor_idx = self.survivor_selection.select(ctx, pool, popsize)

        ctx.population.clear()
        ctx.population.extend(pool.extract(survivor_idx))
