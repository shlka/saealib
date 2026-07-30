"""Adapter exposing a Nevergrad optimizer as saealib's Algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from saealib.algorithms.base import Algorithm
from saealib.callback import PostAskEvent
from saealib.exceptions import ConfigurationError
from saealib.population import Archive, Population, PopulationAttribute

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.optimizer import Dispatchable
    from saealib.problem import Problem


class _NevergradParameterLike(Protocol):
    """Structural interface of a ``nevergrad.parametrization.core.Parameter`` instance.

    ``ask()`` returns one of these per call; it is frozen immediately (no
    further in-place mutation through its public setters is possible -- see
    :class:`NevergradAlgorithm`'s docstring), and must be passed back to
    ``tell()`` unchanged so Nevergrad can match it by ``uid`` to the internal
    state it created at ``ask()`` time.
    """

    uid: str

    @property
    def value(self) -> object:
        """User-facing value (a ``numpy.ndarray`` for a ``p.Array`` parametrization)."""
        ...


class _NevergradParametrizationLike(Protocol):
    """Structural interface of ``optimizer.parametrization``."""

    dimension: int
    random_state: object  # np.random.RandomState; settable


class _NevergradOptimizerLike(Protocol):
    """Structural interface of a ``nevergrad.optimization.base.Optimizer`` instance."""

    num_ask: int
    num_tell: int
    budget: int | None
    no_parallelization: bool

    @property
    def parametrization(self) -> _NevergradParametrizationLike:
        """The search space; owns bounds and the per-instance RNG state."""
        ...

    def ask(self) -> _NevergradParameterLike:
        """Return one new candidate. Callable an arbitrary number of times."""
        ...

    def tell(self, candidate: _NevergradParameterLike, loss: object) -> object:
        """Record ``loss`` (a float) for ``candidate``."""
        ...


class NevergradAlgorithm(Algorithm):
    """
    Adapter wrapping a Nevergrad optimizer as saealib's ``Algorithm``.

    Lets researchers who already have a Nevergrad optimizer instance reuse it
    unchanged inside saealib's ask-tell loop and surrogate-assisted
    strategies. Unlike :class:`~saealib.algorithms.pymoo_algorithm.PymooAlgorithm`
    and :class:`~saealib.algorithms.deap_algorithm.DeapGenerateUpdateAlgorithm`,
    Nevergrad's own ``ask()`` has no batch-size limit fixed at construction --
    it can be called any number of times before a ``tell()`` -- so this
    adapter honors ``n_offspring`` directly (see :meth:`ask`) instead of
    ignoring it.

    Concretely verified against three behaviorally distinct registry
    families -- ``CMA``, ``OnePlusOne``, and ``DE`` -- which track
    outstanding asked candidates in genuinely different ways (CMA batches its
    internal engine update once a full ``popsize`` worth of tells arrives;
    DE/evolution-strategy-style optimizers track outstanding candidates via
    an internal UID queue; ``OnePlusOne`` uses a simpler incumbent-based
    update with no such queue). Other ``nevergrad.optimizers.registry``
    entries are not built or tested against here and should be treated as
    best-effort, not supported.

    Parameters
    ----------
    optimizer : nevergrad.optimization.base.Optimizer
        An already-constructed, **not-yet-used** (``num_ask == 0`` and
        ``num_tell == 0``) Nevergrad optimizer instance, e.g.
        ``ng.optimizers.registry["CMA"](parametrization=param, budget=...)``
        with ``param = ng.p.Array(shape=(dim,)).set_bounds(lb, ub)``. Its
        ``parametrization`` owns the search-space bounds and dimension; keep
        these in sync with the wrapped :class:`~saealib.problem.Problem` (see
        the Bounds note below). Must be a flat ``p.Array`` parametrization --
        composite parametrizations (``p.Dict``, ``p.Instrumentation``, ...)
        are rejected (see the Parametrization type note below).
    allow_partial_tell : bool, optional
        ``PreSelectionStrategy`` truncates (and ``SortByScoreStage``
        reorders) offspring before ``tell()``, so ``tell()`` may receive only
        a subset of what ``ask()`` produced. By default this raises
        :class:`~saealib.exceptions.ConfigurationError`: concretely verified
        that untold candidates are not simply ignored -- e.g. CMA's own
        ``InjectionWarning: orphanated injected solution`` fires internally
        when an asked candidate is never told, and CMA only advances its
        engine once a full ``popsize`` batch of tells has arrived, so a
        partial batch silently delays or skews that update; DE-family
        optimizers track outstanding candidates via an internal UID queue
        that a permanently-untold candidate leaves dangling. Set ``True`` to
        opt in and pass the subset through anyway. Default: False.

    Notes
    -----
    **Engine mode, no population mirroring, no init handshake.** Like
    :class:`~saealib.algorithms.deap_algorithm.DeapGenerateUpdateAlgorithm`,
    the wrapped optimizer's internal state (archive, current
    centroid/covariance or equivalent, per-family internal buffers) is opaque
    and has no externally-readable ``.pop``-equivalent to mirror; ``tell()``
    therefore sets ``ctx.population`` directly to the told offspring. There
    is also no way to seed the wrapped optimizer's internal search
    distribution with saealib's ``Initializer``-evaluated initial
    population: Nevergrad's ``tell()`` does accept "non-asked" candidates
    (built via ``optimizer.parametrization.spawn_child(new_value=...)``,
    per its own docstring), so telling an externally-evaluated DoE without
    ever calling ``ask()`` first *is* possible in principle -- but concretely
    verified against ``CMA`` here, doing so only updates the optimizer's
    archive/recommendation bookkeeping, not the internal sampling
    distribution: a fresh optimizer's first real ``ask()`` was bit-identical
    with and without 10 such non-asked ``tell()`` calls beforehand. So this
    hook exists but is a no-op for the families exercised by this adapter's
    tests, and is not used here.

    **No checkpoint/resume.** ``OptimizationState.save()`` only serializes
    saealib's own arrays; the wrapped optimizer's internal state is not
    captured (Nevergrad has its own separate ``Optimizer.dump()``/``.load()``
    pickle-based mechanism, not wired up here).

    **``n_offspring``.** Honored: :meth:`ask` calls the wrapped optimizer's
    own ``ask()`` exactly ``n_offspring`` times, or ``len(ctx.population)``
    times if ``n_offspring`` is ``None`` (mirroring ``GA.ask()``'s default).
    Rejected outright (raises :class:`~saealib.exceptions.ConfigurationError`)
    when the resolved count is greater than 1 and the wrapped optimizer's own
    ``no_parallelization`` flag is set -- such optimizers (several registry
    entries, e.g. scipy-backed ones) assert single-candidate-at-a-time use
    internally.

    **Budget.** The wrapped optimizer's own ``budget`` (if not ``None``) is
    treated as a hard cap: :meth:`ask` raises
    :class:`~saealib.exceptions.ConfigurationError` rather than silently
    asking past it, since some families use remaining budget internally for
    annealing/decay schedules that assume it is respected.

    **Single-objective, unconstrained only.** Following
    ``DeapGenerateUpdateAlgorithm``'s precedent exactly:
    :meth:`get_required_attrs` raises
    :class:`~saealib.exceptions.ConfigurationError` up front for
    ``problem.n_constraints > 0`` (Nevergrad's own constraint machinery --
    ``parametrization.register_cheap_constraint()``,
    ``tell(..., constraint_violation=...)`` -- is opt-in at the *caller's*
    construction time and orthogonal to this adapter; wiring it up is out of
    scope) or ``problem.n_obj != 1``. Multi-objective was investigated and
    deliberately not supported in this version: without an explicit
    ``MultiobjectiveReference``, Nevergrad's internal
    ``HypervolumePareto`` auto-estimates its reference bounds over an initial
    window of tells (during which it can return a scalar loss of ``0.0`` --
    i.e. the optimizer "sees" a run of tied zero losses before real signal
    kicks in), and the window length itself varies by family (e.g. DE can
    extend it to its own population size). Supporting this properly would
    need an explicit reference-point API and a per-family compatibility
    matrix, which is out of scope for this unit.

    **Parametrization type.** Equal ``parametrization.dimension`` does not by
    itself guarantee ``.value`` is a flat numeric vector -- composite
    parametrizations (``p.Dict``, ``p.Instrumentation``, ...) can have a
    matching *total* dimension while ``.value`` is a tuple/dict/non-vector
    structure. :meth:`get_required_attrs` therefore also requires the
    wrapped optimizer's ``parametrization`` to be a
    ``nevergrad.parametrization.data.Array`` whose ``.value`` is 1-D, raising
    :class:`~saealib.exceptions.ConfigurationError` otherwise.

    **Freshness.** The wrapped optimizer must not have been asked or told
    before being wrapped (``num_ask == 0`` and ``num_tell == 0``), checked
    once at the first :meth:`ask`: seeding ``random_state`` after the
    optimizer has already built up archive/UID-queue/internal-engine state
    would not reset that state, silently leaving a partially-seeded,
    partially-pre-used optimizer.

    **Dimension validation.** ``optimizer.parametrization.dimension`` is
    validated against ``problem.dim`` lazily, on first ``get_required_attrs``/
    ``ask``, raising :class:`~saealib.exceptions.ConfigurationError` on
    mismatch.

    **RNG.** Unlike DEAP's operators (which reseed and restore a *global*
    RNG on every call -- necessary only because DEAP's calls are otherwise
    stateless snapshots against shared global state), Nevergrad's RNG is
    per-instance for the common families: ``optimizer.parametrization.random_state``
    is a plain ``numpy.random.RandomState``, and base
    ``Optimizer._rng`` reads from it, so the optimizer keeps evolving that
    same state naturally across its own subsequent ``ask()``/``tell()``
    calls. This adapter seeds it **once**, lazily on the first :meth:`ask`,
    from a single ``ctx.rng`` draw; reseeding on every call (copying DEAP's
    per-call pattern) would reset the optimizer's random walk repeatedly and
    is deliberately *not* done here. Concretely verified reproducible this
    way for ``CMA``, ``OnePlusOne``, and ``DE``. **This is not universal
    across the whole registry**: some families (certain ``OnePlusOne``
    lognormal/smoothing variants, DE's Voronoi crossover, ``recastlib``-based
    wrappers around external optimizers, ...) call numpy's *global* legacy
    RNG directly, bypassing ``parametrization.random_state`` entirely --
    reproducibility for those is not guaranteed by this adapter.

    **Bounds and coordinate write-back.** The wrapped optimizer's
    ``parametrization`` owns its own bounds (e.g.
    ``ng.p.Array(shape=(dim,)).set_bounds(lb, ub)``), set by the caller at
    construction -- keeping these in sync with the problem's actual bounds is
    the caller's responsibility, the same pattern as the pymoo/DEAP adapters.
    Generated candidates are still run through the problem's own
    constraint-handler repair chain (``handler.repair`` then
    ``problem.repair``) after :meth:`ask`, in case the two bound sources
    drift. Unlike the pymoo/DEAP adapters, this repair step is not optional
    bookkeeping: concretely verified that CMA/DE/OnePlusOne all re-derive
    their internal engine update from the candidate's own cached value
    (``get_standardized_data()`` reads back whatever ``.value`` currently
    holds), so a ``tell()`` call for a repaired point whose candidate object
    still holds the *original*, unrepaired coordinates teaches the optimizer
    "this loss occurred at the unrepaired point" -- silently corrupting its
    model. Candidates are frozen immediately upon return from ``ask()``
    (public setters raise ``RuntimeError: Cannot modify frozen Parameter``),
    so :meth:`ask` writes the repaired coordinates back by mutating the
    candidate's underlying data buffer directly
    (``candidate._value[:] = repaired_x``), bypassing the frozen check the
    same way DEAP's mutable individual view (``ind[:] = xi``) does for
    ``DeapGenerateUpdateAlgorithm``. This is safe as long as the written
    values are already within the parametrization's own bounds (guaranteed
    here, since ``problem.repair`` clips to ``[lb, ub]`` beforehand) --
    writing out-of-bounds values this way would raise on the *next* read of
    ``.value``, since the bounds-clipping layer re-validates on read.

    A ``nevergrad_idx`` column tracks each candidate's position in this
    generation's ask-order, so that :meth:`tell` can recover the correct
    ``Parameter`` object even after ``SortByScoreStage``/``TopKSelectionStage``
    reorder or truncate ``state.offspring`` between ``ask`` and ``tell``.
    """

    def __init__(
        self,
        optimizer: _NevergradOptimizerLike,
        *,
        allow_partial_tell: bool = False,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.allow_partial_tell = allow_partial_tell
        self._validated = False
        self._rng_seeded = False
        self._asked: list[_NevergradParameterLike] = []

    def _ensure_validated(self, problem: Problem) -> None:
        """Reject constrained/multi-objective/dimension-or-type-mismatched problems.

        Cheap; safe to call on every ``get_required_attrs``/``ask``.
        """
        if problem.n_constraints > 0:
            raise ConfigurationError(
                "NevergradAlgorithm: the wrapped Nevergrad optimizer's own "
                "constraint machinery (parametrization.register_cheap_constraint(), "
                "tell(..., constraint_violation=...)) is opt-in at construction time "
                "and not wired up by this adapter. This problem defines "
                f"{problem.n_constraints} constraint(s); use an unconstrained "
                "problem, or handle constraints via the objective/penalty instead."
            )
        if problem.n_obj != 1:
            raise ConfigurationError(
                "NevergradAlgorithm: multi-objective support was investigated and "
                "deliberately not implemented -- Nevergrad's internal "
                "HypervolumePareto auto-bound estimation (without an explicit "
                "MultiobjectiveReference) can return tied zero losses during its "
                f"initial window. This problem defines {problem.n_obj} objectives; "
                "use a single-objective problem instead."
            )

        from nevergrad.parametrization.data import Array as NgArray

        parametrization = self.optimizer.parametrization
        if not isinstance(parametrization, NgArray):
            raise ConfigurationError(
                "NevergradAlgorithm: the wrapped optimizer's parametrization must "
                f"be a nevergrad.p.Array; got {type(parametrization).__name__}. "
                "Composite parametrizations (p.Dict, p.Instrumentation, ...) can "
                "match problem.dim in total size while producing a non-vector "
                "value, which this adapter does not handle."
            )
        if np.asarray(parametrization.value).ndim != 1:
            raise ConfigurationError(
                "NevergradAlgorithm: the wrapped optimizer's parametrization must "
                "be a flat (1-D) p.Array; got shape "
                f"{np.asarray(parametrization.value).shape}."
            )

        dimension = parametrization.dimension
        if dimension != problem.dim:
            raise ConfigurationError(
                f"NevergradAlgorithm: the wrapped optimizer's parametrization "
                f"dimension ({dimension}) does not match problem.dim "
                f"({problem.dim}). Construct the parametrization with "
                "shape=(problem.dim,)."
            )
        self._validated = True

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Validate the problem shape and add a ``nevergrad_idx`` tracking column."""
        self._ensure_validated(problem)
        return [PopulationAttribute("nevergrad_idx", np.int64, (), default=-1)]

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
        """LaTeX notation line for ask(): delegate to the wrapped optimizer."""
        return [r"$\mathcal{Q} \leftarrow \{\text{optimizer.ask()}\}_{i=1}^{n}$"]

    @property
    def tell_notation(self) -> list[str]:
        """LaTeX notation lines for tell(): delegate, then adopt the offspring."""
        return [
            r"$\text{optimizer.tell}(q_i,\, -\vec{w} \odot f(q_i)) \,\forall q_i$",
            r"$P \leftarrow \mathcal{Q}$",
        ]

    def _ensure_ready_for_first_ask(self, ctx: OptimizationState) -> None:
        """Validate freshness and seed the optimizer's RNG, once, lazily."""
        if self._rng_seeded:
            return
        if self.optimizer.num_ask != 0 or self.optimizer.num_tell != 0:
            raise ConfigurationError(
                "NevergradAlgorithm: the wrapped optimizer has already been used "
                f"(num_ask={self.optimizer.num_ask}, "
                f"num_tell={self.optimizer.num_tell}) before being wrapped. Seeding "
                "its random_state at this point would not reset already-built "
                "archive/UID-queue/internal-engine state -- construct a fresh, "
                "not-yet-used optimizer instance instead."
            )
        seed = int(ctx.rng.integers(0, 2**32))
        self.optimizer.parametrization.random_state = np.random.RandomState(seed)
        self._rng_seeded = True

    def ask(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        n_offspring: int | None = None,
    ) -> Population:
        """
        Generate offspring by calling the wrapped optimizer's own ``ask()`` repeatedly.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : Dispatchable
            Component provider.
        n_offspring : int or None, optional
            Number of times to call the wrapped optimizer's ``ask()``. If
            ``None``, defaults to ``len(ctx.population)``.

        Returns
        -------
        Population
            Candidates with ``x`` and ``nevergrad_idx`` set.
        """
        self._ensure_validated(ctx.problem)
        n = n_offspring if n_offspring is not None else len(ctx.population)

        if self.optimizer.no_parallelization and n > 1:
            raise ConfigurationError(
                "NevergradAlgorithm: the wrapped optimizer sets "
                "no_parallelization=True (asserts single-candidate-at-a-time use "
                f"internally), but n_offspring resolved to {n} > 1. Use "
                "n_offspring=1, or a different (parallel-capable) optimizer."
            )
        budget = self.optimizer.budget
        if budget is not None and self.optimizer.num_ask + n > budget:
            raise ConfigurationError(
                f"NevergradAlgorithm: asking for {n} more candidates would exceed "
                f"the wrapped optimizer's own budget ({budget}; "
                f"{self.optimizer.num_ask} already asked). Some optimizer families "
                "use remaining budget internally for annealing/decay schedules "
                "that assume it is respected; raise the optimizer's budget or "
                "reduce n_offspring/generation count instead."
            )
        self._ensure_ready_for_first_ask(ctx)

        candidates = [self.optimizer.ask() for _ in range(n)]
        self._asked = candidates

        x = np.stack([np.asarray(c.value, dtype=float) for c in candidates])

        handler = ctx.problem.handler
        constraints = ctx.problem.constraints
        lb, ub = ctx.problem.lb, ctx.problem.ub
        for i in range(len(x)):
            x[i] = handler.repair(x[i], constraints, lb, ub)
        x = ctx.problem.repair(x)

        # Write the repaired coordinates back into each candidate's own
        # underlying data buffer -- required for correctness (see the
        # "Bounds and coordinate write-back" docstring note), not just
        # bookkeeping. Candidates are frozen, so their public value setter
        # cannot be used; mutating the private buffer in place bypasses that
        # check, mirroring DeapGenerateUpdateAlgorithm's `ind[:] = xi`.
        for candidate, xi in zip(candidates, x, strict=True):
            candidate._value[:] = xi

        provider.dispatch(PostAskEvent(ctx=ctx, candidates=x))

        cand = ctx.population.empty_like(capacity=len(x))
        cand.extend({"x": x, "nevergrad_idx": np.arange(len(x), dtype=np.int64)})
        return cand

    def tell(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        offspring: Population,
    ) -> None:
        """
        Tell the wrapped optimizer, then adopt ``offspring`` as ``ctx.population``.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : Dispatchable
            Component provider.
        offspring : Population
            Offspring population, possibly reordered relative to what
            :meth:`ask` produced (e.g. by ``SortByScoreStage``), or truncated
            (e.g. by ``PreSelectionStrategy``'s top-k truncation) -- the
            latter is rejected unless ``allow_partial_tell=True`` was passed
            at construction (see its docstring).
        """
        idx = offspring.get_array("nevergrad_idx").astype(np.int64, copy=False)

        if len(idx) != len(self._asked) and not self.allow_partial_tell:
            raise ConfigurationError(
                f"NevergradAlgorithm.tell() received {len(idx)} of "
                f"{len(self._asked)} candidates ask() produced. This happens with "
                "PreSelectionStrategy's top-k truncation; concretely verified that "
                "the wrapped optimizer does not treat untold candidates as simply "
                "ignorable (e.g. CMA's own 'orphanated injected solution' warning, "
                "DE-family UID-queue bookkeeping left dangling). Pass "
                "allow_partial_tell=True to opt in anyway, or use "
                "DirectStrategy/IndividualBasedStrategy instead."
            )

        f = offspring.get_array("f")
        loss = (-ctx.problem.direction * f).ravel()
        for row, i in enumerate(idx):
            candidate = self._asked[int(i)]
            self.optimizer.tell(candidate, float(loss[row]))

        ctx.population.clear()
        ctx.population.extend(offspring)
