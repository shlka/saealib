"""Adapter exposing a DEAP generate/update strategy as saealib's Algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from saealib.algorithms.base import Algorithm
from saealib.callback import PostAskEvent
from saealib.exceptions import ConfigurationError
from saealib.operators._deap_rng import seeded_global_numpy_random
from saealib.population import Archive, Population, PopulationAttribute

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.optimizer import Dispatchable
    from saealib.problem import Problem


class _DeapGenerateUpdateLike(Protocol):
    """Structural interface of a DEAP generate/update strategy object.

    Matches the protocol ``deap.algorithms.eaGenerateUpdate`` expects from a
    strategy: ``generate(ind_init)`` produces one generation of candidates,
    ``update(population)`` advances the strategy's internal state once those
    candidates have been fitness-evaluated. ``deap.cma.Strategy``
    (single-objective CMA-ES) is the primary real-world implementer.
    """

    def generate(self, ind_init: object) -> list[object]:
        """Return this generation's individuals, built via ``ind_init(row)``."""
        ...

    def update(self, population: list[object]) -> None:
        """Advance internal state given a population with ``.fitness`` set."""
        ...


class _DeapIndividual(np.ndarray):
    """Throwaway ndarray subclass carrying an assignable ``fitness`` attribute.

    ``deap.cma.Strategy.update()`` requires ``population[0:mu]`` to support
    numpy-style subtraction against the centroid (a plain Python list does
    not support this), and DEAP's generate/update protocol requires each
    individual to carry a ``.fitness`` attribute (which a plain
    ``numpy.ndarray`` cannot carry either, hence the subclass). Built
    directly as a plain ``ndarray`` subclass rather than via
    ``deap.creator.create`` -- that function registers into DEAP's shared,
    global ``deap.creator`` module namespace, where a name collision with a
    user's own ``creator.Individual`` is a real risk across repeated
    :class:`DeapGenerateUpdateAlgorithm` construction or the user's own DEAP
    usage in the same process. This class carries no problem-specific state
    (only a per-instance ``fitness`` object set at construction time), so a
    single module-level definition is safe to share across every adapter
    instance without any such collision risk.
    """

    def __array_finalize__(self, obj: object) -> None:
        """Default ``fitness`` to ``None`` so numpy-internal views don't error."""
        self.fitness = getattr(obj, "fitness", None)


class DeapGenerateUpdateAlgorithm(Algorithm):
    """
    Adapter wrapping a DEAP generate/update strategy as saealib's ``Algorithm``.

    Lets researchers who already have a DEAP strategy object exposing the
    ``generate``/``update`` protocol (see ``_DeapGenerateUpdateLike``) --
    concretely, ``deap.cma.Strategy`` -- reuse it unchanged inside saealib's
    ask-tell loop and surrogate-assisted strategies. This is a *generic*
    protocol adapter, not a CMA-ES-specific one; any object shaped like
    ``_DeapGenerateUpdateLike`` can be wrapped.

    Parameters
    ----------
    strategy : _DeapGenerateUpdateLike
        An already-constructed DEAP strategy, e.g.
        ``deap.cma.Strategy(centroid=[0.0] * dim, sigma=1.0)``.
    allow_partial_tell : bool, optional
        ``PreSelectionStrategy`` truncates (and ``SortByScoreStage``
        reorders) offspring before ``tell()``, so ``tell()`` may receive
        only a subset of what ``ask()`` produced, or in a different order.
        By default this raises
        :class:`~saealib.exceptions.ConfigurationError`, since
        ``cma.Strategy.update()``'s recombination weights (``self.weights``,
        sized from ``mu``/``lambda_`` at construction) silently produce
        statistically-invalid results when fed a subset that doesn't match
        those sizes. Set ``True`` to opt in and pass the subset through
        anyway -- the wrapped strategy's own ``update()`` is not otherwise
        guarded against this, and this adapter has no general concept of
        ``mu`` for an arbitrary wrapped strategy. In particular, for
        ``cma.Strategy``, a subset smaller than ``strategy.mu`` does not
        degrade gracefully: ``update()`` raises a bare
        ``numpy.linalg``-style shape-mismatch ``ValueError`` from inside DEAP
        itself (not a saealib :class:`~saealib.exceptions.ConfigurationError`),
        since ``population[0:mu]`` then has fewer rows than ``self.weights``
        has entries. Default: False.

    Notes
    -----
    **Single-objective, unconstrained only.** ``cma.Strategy`` -- the
    intended real-world use of this adapter -- is an unconstrained
    single-objective optimizer; there is no G/H/CV concept anywhere in its
    API, and its ``update()`` sorts by a single DEAP ``Fitness`` value.
    :meth:`get_required_attrs` raises
    :class:`~saealib.exceptions.ConfigurationError` up front for
    ``problem.n_constraints > 0`` or ``problem.n_obj != 1``, rather than
    silently ignoring constraint violations or silently lexicographically
    scalarizing multiple objectives.

    **No population mirroring.** Unlike
    :class:`~saealib.algorithms.pymoo_algorithm.PymooAlgorithm`, the wrapped
    strategy has no ``.pop``-equivalent persistent population to mirror --
    only ``centroid``/``sigma``/``C`` (or equivalent internal state) persist
    across generations, entirely inside the wrapped strategy object, not
    inside a population saealib could read back. :meth:`tell` therefore sets
    ``ctx.population`` directly to the told offspring, with no separate
    state-mirroring step. One consequence: ``ctx.population``'s size becomes
    ``strategy.lambda_`` (or whatever ``generate()`` returns) after the
    first :meth:`tell`, which may differ from the initial population size
    ``Initializer``/``Optimizer.set_popsize`` configured -- nothing else in
    the pipeline assumes a fixed population size across generations, but
    callers relying on a stable ``len(ctx.population)`` should align the two
    explicitly (e.g. ``lambda_=pop_size`` at ``cma.Strategy`` construction).

    **No initialization handshake.** Unlike a wrapped pymoo algorithm (whose
    internal population must be seeded with saealib's initial DoE before its
    first ``ask()``), ``cma.Strategy``'s ``centroid``/``sigma`` are fully
    determined by the constructor arguments the caller already supplied --
    there is no API on ``Strategy`` to accept an externally-provided initial
    population. saealib's own ``Initializer``-evaluated initial population
    is therefore **not** fed into the wrapped strategy's internal state; the
    strategy starts purely from its own ``centroid``/``sigma`` on the first
    :meth:`ask`, independent of ``Initializer``'s archive/population.

    **``n_offspring`` is ignored.** The offspring count is fixed by whatever
    the wrapped strategy's own ``generate()`` returns (``cma.Strategy``'s
    ``lambda_``); strategies that request a larger candidate pool for
    pre-selection cannot enlarge it here.

    **No checkpoint/resume.** ``OptimizationState.save()`` only serializes
    saealib's own arrays; the wrapped strategy's internal state
    (``centroid``, ``sigma``, ``C``, evolution paths, ...) is not captured.

    **Bounds are repaired, not respected by sampling.** ``cma.Strategy`` has
    no notion of ``[lb, ub]`` -- it samples unbounded Gaussian steps. Each
    generated individual is repaired in place (via the problem's constraint
    handler, then ``problem.repair``) before being handed to the true
    objective, mirroring ``GA``'s own post-crossover/mutation repair chain.
    The repaired coordinates are written back into the same individual
    object (not just the returned ``Population``), so the covariance update
    in :meth:`tell` operates on the values that were actually evaluated, not
    the original unclipped Gaussian sample.

    **RNG bridging.** ``cma.Strategy.generate()`` samples from **numpy's
    global** RNG (``numpy.random.standard_normal``), not an injectable
    ``Generator`` and not Python's ``random`` module. Each :meth:`ask` call
    seeds a fresh numpy global RNG state (derived from ``ctx.rng``, consuming
    one ``rng`` draw) via
    :func:`saealib.operators._deap_rng.seeded_global_numpy_random`, lets the
    ``generate()`` call run under that seeded state, then restores the
    previous global numpy RNG state in a ``finally`` block -- even if
    ``generate()`` raises. This swap is process-global and **not safe under
    concurrent/multi-threaded use**. ``update()`` involves no randomness, so
    :meth:`tell` does not touch numpy's global RNG state at all.

    **Scope beyond ``cma.Strategy``.** ``deap.cma.StrategyOnePlusLambda`` and
    ``deap.cma.StrategyMultiObjective`` are not built or tested against here.
    ``StrategyOnePlusLambda`` exposes the same ``generate``/``update`` method
    signatures and is single-objective, so it plausibly fits this adapter
    as-is (not verified) -- but its ``update()`` also requires its
    constructor-supplied ``parent`` individual to already carry a valid
    ``.fitness`` before the first :meth:`tell`, which is the caller's
    responsibility, not this adapter's. ``StrategyMultiObjective`` is
    multi-objective (blocked by the ``n_obj != 1`` guard above) and its
    ``generate()``/``update()`` additionally depend on a private ``._ps``
    parent-tag attribute and an initial population supplied at the
    strategy's own construction (not via this adapter), so it is out of
    scope regardless.
    """

    def __init__(
        self,
        strategy: _DeapGenerateUpdateLike,
        *,
        allow_partial_tell: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.allow_partial_tell = allow_partial_tell
        self._fitness_class: type | None = None
        self._last_individuals: list[_DeapIndividual] | None = None

    def _ensure_setup(self, problem: Problem) -> None:
        """Validate the problem shape and lazily build this run's Fitness class."""
        if problem.n_constraints > 0:
            raise ConfigurationError(
                "DeapGenerateUpdateAlgorithm: the wrapped DEAP strategy (e.g. "
                "cma.Strategy) has no constraint-handling concept -- there is no "
                "G/H/CV in its API. This problem defines "
                f"{problem.n_constraints} constraint(s); use an unconstrained "
                "problem, or handle constraints via the objective/penalty "
                "instead."
            )
        if problem.n_obj != 1:
            raise ConfigurationError(
                "DeapGenerateUpdateAlgorithm: the wrapped DEAP strategy (e.g. "
                "cma.Strategy) is single-objective -- its update() sorts "
                "individuals by a single DEAP Fitness value, which would "
                f"silently lexicographically scalarize the {problem.n_obj} "
                "objectives this problem defines rather than optimizing them "
                "jointly. Use a single-objective problem instead."
            )
        if self._fitness_class is None:
            from deap import base as deap_base

            # A dedicated Fitness subclass per algorithm instance, carrying
            # this problem's direction as the class-level `weights` DEAP's
            # base.Fitness requires. Built with `type()` directly rather than
            # `deap.creator.create` (see _DeapIndividual's docstring for why):
            # this stays entirely within this module, never touching DEAP's
            # shared global creator namespace.
            self._fitness_class = type(
                f"_SaealibDeapFitness_{id(self)}",
                (deap_base.Fitness,),
                {"weights": tuple(problem.direction)},
            )

    def _ind_init(self, row: np.ndarray) -> _DeapIndividual:
        """Build one ``_DeapIndividual`` with a fresh, unset Fitness object."""
        assert self._fitness_class is not None
        ind = np.asarray(row, dtype=float).view(_DeapIndividual)
        ind.fitness = self._fitness_class()
        return ind

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Validate the problem shape and add a ``deap_idx`` tracking column."""
        self._ensure_setup(problem)
        return [PopulationAttribute("deap_idx", np.int64, (), default=-1)]

    @property
    def population_class(self) -> type[Population]:
        """Return the population class."""
        return Population

    @property
    def archive_class(self) -> type[Archive]:
        """Return the archive class."""
        return Archive

    @property
    def ask_notation(self) -> list[str]:
        """LaTeX notation line for ask(): delegate to the wrapped strategy."""
        return [r"$\mathcal{Q} \leftarrow \text{strategy.generate(ind\_init)}$"]

    @property
    def tell_notation(self) -> list[str]:
        """LaTeX notation lines for tell(): delegate, then adopt the offspring."""
        return [
            r"$\text{strategy.update}(\mathcal{Q})$",
            r"$P \leftarrow \mathcal{Q}$",
        ]

    def ask(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        n_offspring: int | None = None,
    ) -> Population:
        """
        Generate offspring via the wrapped strategy's own ``generate()``.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : Dispatchable
            Component provider.
        n_offspring : int or None, optional
            Ignored; the offspring count is fixed by the wrapped strategy's
            own ``generate()``.

        Returns
        -------
        Population
            Candidates with ``x`` and ``deap_idx`` set.
        """
        self._ensure_setup(ctx.problem)

        with seeded_global_numpy_random(ctx.rng):
            individuals = list(self.strategy.generate(self._ind_init))
        self._last_individuals = individuals

        x = np.stack([np.asarray(ind, dtype=float) for ind in individuals])

        handler = ctx.problem.handler
        constraints = ctx.problem.constraints
        lb, ub = ctx.problem.lb, ctx.problem.ub
        for i in range(len(x)):
            x[i] = handler.repair(x[i], constraints, lb, ub)
        x = ctx.problem.repair(x)
        # Write the repaired coordinates back into the individual objects
        # themselves, so update()'s covariance computation (in tell()) is
        # consistent with what was actually evaluated, not the raw sample.
        for ind, xi in zip(individuals, x, strict=True):
            ind[:] = xi

        provider.dispatch(PostAskEvent(ctx=ctx, candidates=x))

        cand = ctx.population.empty_like(capacity=len(x))
        cand.extend({"x": x, "deap_idx": np.arange(len(x), dtype=np.int64)})
        return cand

    def tell(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        offspring: Population,
    ) -> None:
        """
        Update the wrapped strategy, then adopt ``offspring`` as ``ctx.population``.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : Dispatchable
            Component provider.
        offspring : Population
            Offspring population, possibly reordered or truncated relative
            to what :meth:`ask` produced.
        """
        assert self._last_individuals is not None
        idx = offspring.get_array("deap_idx").astype(np.int64, copy=False)

        if len(idx) != len(self._last_individuals) and not self.allow_partial_tell:
            raise ConfigurationError(
                f"DeapGenerateUpdateAlgorithm.tell() received {len(idx)} of "
                f"{len(self._last_individuals)} candidates ask() produced. This "
                "happens with PreSelectionStrategy's top-k truncation; the "
                "wrapped strategy's update() (e.g. cma.Strategy's recombination "
                "weights, sized from mu/lambda_ at construction) would silently "
                "produce statistically-invalid results under a partial tell. "
                "Pass allow_partial_tell=True to opt in anyway, or use "
                "DirectStrategy/IndividualBasedStrategy instead."
            )

        f = offspring.get_array("f")
        told = []
        for row, i in enumerate(idx):
            ind = self._last_individuals[int(i)]
            ind.fitness.values = tuple(float(v) for v in f[row])
            told.append(ind)

        self.strategy.update(told)

        ctx.population.clear()
        ctx.population.extend(offspring)
