"""Adapter exposing an already-constructed pymoo Algorithm as saealib's Algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from saealib.algorithms.base import Algorithm
from saealib.callback import PostAskEvent
from saealib.exceptions import ConfigurationError
from saealib.population import Archive, Population, PopulationAttribute
from saealib.problem.constraint import EqualityConstraint

if TYPE_CHECKING:
    from pymoo.core.problem import Problem as PymooCoreProblem

    from saealib.context import OptimizationState
    from saealib.identity import IDAllocator
    from saealib.optimizer import Dispatchable
    from saealib.problem import Problem


class _PymooPopulationLike(Protocol):
    """Structural interface of a ``pymoo.core.population.Population`` instance."""

    def get(self, *args: object, **kwargs: object) -> object:
        """Read one or more fields (e.g. "X", "F", "G", "H", "CV")."""
        ...

    def set(self, *args: object, **kwargs: object) -> object:
        """Write one or more fields (e.g. "X", "F", "G", "H")."""
        ...

    def __len__(self) -> int:
        """Return the number of individuals."""
        ...


class _PymooAlgorithmLike(Protocol):
    """Structural interface of a ``pymoo.core.algorithm.Algorithm`` instance."""

    @property
    def pop(self) -> _PymooPopulationLike | None:
        """Internal population; None before setup(). Read-only from here."""
        ...

    def setup(self, problem: PymooCoreProblem, **kwargs: object) -> object:
        """Bind the algorithm to a problem and initialize its internal state."""
        ...

    def ask(self) -> _PymooPopulationLike | None:
        """Return the next infill population, or None if exhausted."""
        ...

    def tell(self, *args: object, **kwargs: object) -> object:
        """Advance internal state given evaluated infills."""
        ...


class PymooAlgorithm(Algorithm):
    """
    Adapter wrapping a pymoo Algorithm (e.g. ``NSGA2()``) as saealib's ``Algorithm``.

    Lets researchers who already have a pymoo algorithm reuse it unchanged
    inside saealib's ask-tell loop and surrogate-assisted strategies.

    Parameters
    ----------
    pymoo_algorithm : pymoo.core.algorithm.Algorithm
        An already-constructed pymoo algorithm instance (not yet ``setup()``).
    allow_partial_tell : bool, optional
        ``PreSelectionStrategy`` truncates offspring before ``tell()``, so
        ``tell()`` may receive only a subset of what ``ask()`` produced. By
        default this raises :class:`~saealib.exceptions.ConfigurationError`,
        since index-coupled pymoo algorithms (e.g. differential evolution)
        would silently misalign parents and offspring under a subset. Set
        ``True`` to opt in and pass the subset through anyway. Default: False.

    Notes
    -----
    **Engine mode.** The wrapped pymoo algorithm owns its own population and
    internal state (survival mechanics, index-coupled operators, adaptive
    parameters). ``ctx.population`` is a *mirror*, refreshed from the pymoo
    algorithm's own ``.pop`` at the end of every :meth:`tell`, not the source
    of truth. This is the only way to reuse a pymoo algorithm's own
    tested logic unchanged, but it comes with real limitations:

    - **No checkpoint/resume.** ``OptimizationState.save()`` only
      serializes saealib's own arrays; the wrapped pymoo algorithm's
      internal state is not captured.
    - **``n_offspring`` is ignored.** The offspring count is fixed by the
      wrapped algorithm's own configuration (e.g. its ``pop_size``);
      strategies that request a larger candidate pool for pre-selection
      cannot enlarge it here.
    - **Constrained problems combined with surrogate-only generations are
      unsupported.** When a candidate never receives a true evaluation,
      saealib leaves its ``g``/``cv`` at schema defaults (``0.0``, looking
      falsely feasible) — this is an existing, library-wide limitation
      (:class:`~saealib.surrogate.prediction.SurrogatePrediction` has no
      ``cv`` field), not something specific to this adapter, but it means
      ``PymooAlgorithm``'s constraint handling should not be trusted under
      ``SurrogateOnlyLoopStage``/IB's untrue tail.

    Initialization is handled by mirroring saealib's already-evaluated
    initial population into the pymoo algorithm's own initial ``ask()``
    result before calling ``tell()`` on it, so the initial design-of-
    experiments is not duplicated or discarded.

    A ``pymoo_idx`` column tracks each candidate's position in the pymoo
    infill population, so that :meth:`tell` can recover the correct mapping
    even after ``SortByScoreStage``/``TopKSelectionStage`` reorder or
    truncate ``state.offspring`` between ``ask`` and ``tell``.
    """

    def __init__(
        self,
        pymoo_algorithm: _PymooAlgorithmLike,
        *,
        allow_partial_tell: bool = False,
    ) -> None:
        super().__init__()
        self.pymoo_algorithm = pymoo_algorithm
        self.allow_partial_tell = allow_partial_tell
        self._initialized = False
        self._ieq_idx: np.ndarray = np.empty(0, dtype=np.int64)
        self._eq_idx: np.ndarray = np.empty(0, dtype=np.int64)
        self._infills: _PymooPopulationLike | None = None

    def get_required_attrs(self, problem: Problem) -> list[PopulationAttribute]:
        """Add a ``pymoo_idx`` column tracking position in the pymoo infill pop."""
        return [PopulationAttribute("pymoo_idx", np.int64, (), default=-1)]

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
        """LaTeX notation line for ask(): delegate to the wrapped pymoo algorithm."""
        return [r"$\mathcal{Q} \leftarrow \text{pymoo\_algorithm.ask()}$"]

    @property
    def tell_notation(self) -> list[str]:
        """LaTeX notation lines for tell(): delegate, then mirror the population."""
        return [
            r"$\text{pymoo\_algorithm.tell}(\mathcal{Q})$",
            r"$P \leftarrow \text{pymoo\_algorithm.pop}$",
        ]

    def _build_pymoo_problem(self, problem: Problem) -> PymooCoreProblem:
        """Synthesize a minimal pymoo Problem shim; never evaluated by pymoo itself."""
        from pymoo.core.problem import Problem as PymooProblem

        eq_mask = np.array(
            [isinstance(c, EqualityConstraint) for c in problem.constraints],
            dtype=bool,
        )
        self._ieq_idx = np.where(~eq_mask)[0]
        self._eq_idx = np.where(eq_mask)[0]

        return PymooProblem(
            n_var=problem.dim,
            n_obj=problem.n_obj,
            n_ieq_constr=len(self._ieq_idx),
            n_eq_constr=len(self._eq_idx),
            xl=np.asarray(problem.lb, dtype=float),
            xu=np.asarray(problem.ub, dtype=float),
        )

    def _assign_objectives(
        self,
        pymoo_pop: _PymooPopulationLike,
        offspring: Population,
        problem: Problem,
        scatter_order: np.ndarray | None = None,
    ) -> None:
        """Write F (sign-converted) and split G into pymoo's G/H onto pymoo_pop.

        ``scatter_order`` maps each row ``k`` of ``offspring`` to its
        original ask-order position ``scatter_order[k]`` in ``pymoo_pop``
        (``pymoo_pop`` has ``offspring``'s *original*, un-reordered length).
        This is a scatter, not a gather: row ``k`` of ``offspring`` writes to
        position ``scatter_order[k]`` of ``pymoo_pop``, preserving each
        pymoo individual's original identity/order — required for
        index-coupled algorithms (e.g. DE) that compare ``pop[k]`` against
        ``infills[k]`` positionally. ``None`` means ``offspring`` is already
        in the same order as ``pymoo_pop`` (the initialization handshake, or
        an already-realigned partial-tell subset).
        """
        f = offspring.get_array("f")
        g = offspring.get_array("g")
        if scatter_order is not None:
            n = len(pymoo_pop)
            f_full = np.empty((n, f.shape[1]), dtype=float)
            f_full[scatter_order] = f
            f = f_full
            if g.shape[1] > 0:
                g_full = np.zeros((n, g.shape[1]), dtype=float)
                g_full[scatter_order] = g
                g = g_full
        pymoo_pop.set("F", -problem.direction * f)
        if len(self._ieq_idx) > 0:
            pymoo_pop.set("G", g[:, self._ieq_idx])
        if len(self._eq_idx) > 0:
            pymoo_pop.set("H", g[:, self._eq_idx])

    def _ensure_initialized(self, ctx: OptimizationState) -> None:
        """Bind the wrapped algorithm to the problem and seed it with saealib's DoE."""
        if self._initialized:
            return
        from pymoo.core.termination import NoTermination

        pymoo_problem = self._build_pymoo_problem(ctx.problem)
        self.pymoo_algorithm.setup(
            pymoo_problem,
            termination=NoTermination(),
            seed=int(ctx.rng.integers(0, 2**31 - 1)),
            verbose=False,
        )
        init = self.pymoo_algorithm.ask()
        if init is None or len(init) != len(ctx.population):
            got = 0 if init is None else len(init)
            raise ConfigurationError(
                f"pymoo algorithm's initial population size ({got}, from its own "
                f"pop_size) must equal saealib's initial population size "
                f"({len(ctx.population)}, from Initializer/Optimizer.set_popsize). "
                "Set both to the same value."
            )
        init.set("X", ctx.population.get_array("x"))
        self._assign_objectives(init, ctx.population, ctx.problem)
        self.pymoo_algorithm.tell(infills=init)
        self._initialized = True

    def ask(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        n_offspring: int | None = None,
    ) -> Population:
        """
        Generate offspring via the wrapped pymoo algorithm's own ``ask()``.

        Parameters
        ----------
        ctx : OptimizationState
            Current optimization context.
        provider : Dispatchable
            Component provider.
        n_offspring : int or None, optional
            Ignored; the offspring count is fixed by the wrapped pymoo
            algorithm's own configuration.

        Returns
        -------
        Population
            Candidates with ``x`` and ``pymoo_idx`` set.
        """
        self._ensure_initialized(ctx)
        infills = self.pymoo_algorithm.ask()
        if infills is None:
            raise ConfigurationError(
                "The wrapped pymoo algorithm's ask() returned None (has_next() is "
                "False). Check its termination/pop_size configuration."
            )
        self._infills = infills
        x = np.asarray(infills.get("X"), dtype=float)

        handler = ctx.problem.handler
        constraints = ctx.problem.constraints
        lb, ub = ctx.problem.lb, ctx.problem.ub
        for i in range(len(x)):
            x[i] = handler.repair(x[i], constraints, lb, ub)
            x[i] = ctx.problem.repair(x[i])

        provider.dispatch(PostAskEvent(ctx=ctx, candidates=x))

        cand = ctx.population.empty_like(capacity=len(x))
        cand.extend({"x": x, "pymoo_idx": np.arange(len(x), dtype=np.int64)})
        return cand

    def tell(
        self,
        ctx: OptimizationState,
        provider: Dispatchable,
        offspring: Population,
    ) -> None:
        """
        Update the wrapped pymoo algorithm, then mirror its population into ``ctx``.

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
        assert self._infills is not None
        idx = offspring.get_array("pymoo_idx").astype(np.int64, copy=False)

        if len(idx) == len(self._infills):
            # Full reorder (possibly identity). Scatter F/G/H back into
            # self._infills's *original* ask-order positions so the object
            # passed to tell() has the exact identity/order ask() produced,
            # which index-coupled algorithms (e.g. DE) rely on positionally.
            self._assign_objectives(
                self._infills, offspring, ctx.problem, scatter_order=idx
            )
            infills = self._infills
        else:
            if not self.allow_partial_tell:
                raise ConfigurationError(
                    f"PymooAlgorithm.tell() received {len(idx)} of "
                    f"{len(self._infills)} candidates ask() produced. This happens "
                    "with PreSelectionStrategy's top-k truncation; index-coupled "
                    "pymoo algorithms (e.g. differential evolution) would silently "
                    "misalign parents and offspring under a partial tell. Pass "
                    "allow_partial_tell=True to opt in anyway, or use "
                    "DirectStrategy/IndividualBasedStrategy instead."
                )
            # offspring row k already corresponds to original ask-position
            # idx[k]; subsetting self._infills by idx realigns it to match
            # offspring directly, so no further reordering is needed here.
            infills = self._infills[idx]  # type: ignore  # pymoo Population is ndarray-subscriptable; not worth modeling numpy's full __getitem__ overload set in the Protocol
            self._assign_objectives(infills, offspring, ctx.problem)

        self.pymoo_algorithm.tell(infills=infills)
        self._sync_population(ctx)

    @staticmethod
    def _match_survivor_ids(
        prev_x: np.ndarray,
        prev_ids: np.ndarray,
        new_x: np.ndarray,
        allocator: IDAllocator,
    ) -> np.ndarray:
        """Match rows by exact ``x``; unmatched rows get freshly minted ids."""
        available: dict[bytes, list[int]] = {}
        for i in range(len(prev_x)):
            key = np.ascontiguousarray(prev_x[i]).tobytes()
            available.setdefault(key, []).append(int(prev_ids[i]))

        n = len(new_x)
        ids = np.empty(n, dtype=np.int64)
        unmatched: list[int] = []
        for i in range(n):
            key = np.ascontiguousarray(new_x[i]).tobytes()
            bucket = available.get(key)
            if bucket:
                ids[i] = bucket.pop()
            else:
                unmatched.append(i)

        if unmatched:
            fresh = allocator.allocate(len(unmatched))
            for j, i in enumerate(unmatched):
                ids[i] = fresh[j]
        return ids

    def _sync_population(self, ctx: OptimizationState) -> None:
        """Rebuild ctx.population in-place from the wrapped algorithm's own .pop."""
        alg_pop = self.pymoo_algorithm.pop
        assert alg_pop is not None  # setup()/tell() already ran by this point
        if len(alg_pop) != len(ctx.population):
            raise ConfigurationError(
                f"Wrapped pymoo algorithm's internal population size ({len(alg_pop)}) "
                f"no longer matches saealib's population size ({len(ctx.population)}). "
                "This can happen with archive-growing pymoo algorithms that are not "
                "supported by this adapter."
            )
        problem = ctx.problem
        n = len(alg_pop)
        x = np.asarray(alg_pop.get("X"), dtype=float)
        f = -problem.direction * np.asarray(alg_pop.get("F"), dtype=float)
        g = np.zeros((n, problem.n_constraints), dtype=float)
        if len(self._ieq_idx) > 0:
            g[:, self._ieq_idx] = np.asarray(alg_pop.get("G"), dtype=float)
        if len(self._eq_idx) > 0:
            g[:, self._eq_idx] = np.asarray(alg_pop.get("H"), dtype=float)
        cv = np.asarray(alg_pop.get("CV"), dtype=float).reshape(n)

        new_pop_data = {
            "x": x,
            "f": f,
            "g": g,
            "cv": cv,
            "pymoo_idx": np.full(n, -1, dtype=np.int64),
        }
        if "id" in ctx.population.schema:
            new_pop_data["id"] = self._match_survivor_ids(
                ctx.population.get_array("x"),
                ctx.population.get_array("id"),
                x,
                ctx.candidate_id_allocator,
            )

        ctx.population.clear()
        ctx.population._extend_internal(new_pop_data, preserve_ids=True)
