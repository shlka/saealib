"""Concrete Stage implementations for the optimization pipeline.

Each stage receives an OptimizationState, performs one well-defined operation,
and returns an updated state via ``state.replace()``.

Standard pipeline fields on OptimizationState
----------------------------------------------
``offspring``
    Current candidate population (Population), set by AskStage.
``scores``
    1-D acquisition score array (np.ndarray), set by AcquisitionStage.
``predictions``
    Batched SurrogatePrediction for ``offspring``, set by SurrogatePredictStage.
``evaluated_offspring``
    Sub-population with true objective values, set by TrueEvaluationStage.

Custom stages may store additional values in ``state.data`` (user-extensible
dict) via ``state.replace(data={**state.data, "key": value})``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from saealib.acquisition.base import AcquisitionResult
from saealib.callback import (
    AcquisitionEndEvent,
    AcquisitionStartEvent,
    PostEvaluationEvent,
    PostSurrogateFitEvent,
    SurrogateEndEvent,
    SurrogateStartEvent,
)
from saealib.exceptions import ValidationError
from saealib.pipeline import Pipeline, Stage
from saealib.strategies.base import assign_tell_f
from saealib.surrogate.manager import _split_prediction

if TYPE_CHECKING:
    from collections.abc import Callable

    from saealib.acquisition.base import AcquisitionFunction
    from saealib.algorithms.base import Algorithm
    from saealib.callback import CallbackManager, Event
    from saealib.context import OptimizationState
    from saealib.execution.evaluator import Evaluator
    from saealib.execution.initializer import Initializer
    from saealib.optimizer import ComponentProvider
    from saealib.problem import Problem
    from saealib.surrogate.manager import SurrogateManager


class _DispatchProxy:
    """Minimal ComponentProvider used to thread callbacks through Algorithm.ask/tell.

    This shim preserves compatibility with the current Algorithm interface while
    the full provider is being phased out.  It will be removed once Algorithm.ask
    and Algorithm.tell no longer accept a provider argument.
    """

    def __init__(self, cbmanager: CallbackManager | None = None) -> None:
        self._cbmanager = cbmanager

    def dispatch(self, event: Event) -> None:
        if self._cbmanager is not None:
            self._cbmanager.dispatch(event)

    @property
    def algorithm(self) -> None:
        return None

    @property
    def strategy(self) -> None:
        return None

    @property
    def surrogate_manager(self) -> None:
        return None

    @property
    def evaluator(self) -> None:
        return None

    @property
    def termination(self) -> None:
        return None

    @property
    def cbmanager(self) -> CallbackManager | None:
        return self._cbmanager

    @property
    def seed(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Concrete stages
# ---------------------------------------------------------------------------


class CountGenerationStage(Stage):
    """Increment the generation counter by one."""

    name = "count_generation"
    label = "Count generation"
    notation = r"$gen \leftarrow gen + 1$"

    def execute(self, state: OptimizationState) -> OptimizationState:
        return state.replace(gen=state.gen + 1)


class AskStage(Stage):
    """Generate offspring candidates via the algorithm's ask() method.

    Writes the offspring population to ``state.offspring``.

    Parameters
    ----------
    algorithm : Algorithm
        The evolutionary algorithm that generates candidates.
    n_offspring : int or None
        Number of offspring to request.  Passed directly to
        ``algorithm.ask()``.  ``None`` lets the algorithm decide.
    cbmanager : CallbackManager or None
        If provided, PostCrossoverEvent / PostMutationEvent / PostAskEvent
        are dispatched through this manager.
    """

    name = "ask"
    label = "Generate offspring"
    notation = r"$\mathcal{Q} \leftarrow \text{ask}(P, n)$"

    def __init__(
        self,
        algorithm: Algorithm,
        n_offspring: int | None = None,
        cbmanager: CallbackManager | None = None,
    ) -> None:
        super().__init__()
        self._algorithm = algorithm
        self._n_offspring = n_offspring
        self._proxy = _DispatchProxy(cbmanager)

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        r"""Expand into per-operator lines via ``Algorithm.ask_notation``."""
        prefix = "  " * indent
        ask_notation: list[str] | None = getattr(self._algorithm, "ask_notation", None)
        if expand and ask_notation:
            label = self.label or self.name
            lines = "\n".join(f"{prefix}  \\State {n}" for n in ask_notation)
            return f"{prefix}\\Comment{{{label}}}\n{lines}"
        return f"{prefix}\\State {self.notation}"

    def execute(self, state: OptimizationState) -> OptimizationState:
        candidates = self._algorithm.ask(state, self._proxy, self._n_offspring)
        if "id" in candidates.schema:
            id_arr = candidates.get_array("id")
            unassigned = np.where(id_arr == -1)[0]
            if len(unassigned) > 0:
                new_ids = state.candidate_id_allocator.allocate(len(unassigned))
                candidates._assign_ids(unassigned, new_ids)
            assigned = candidates.get_array("id")
            real = assigned[assigned != -1]
            if len(real) != len(np.unique(real)):
                raise ValidationError(
                    "AskStage received offspring with duplicate candidate ids"
                )
        return state.replace(offspring=candidates)


class SurrogatePredictStage(Stage):
    """Predict offspring with the surrogate model; assign per-candidate tell_f.

    Reads ``state.offspring``, writes the batched prediction to
    ``state.predictions``.  Also assigns predicted objective values
    (``tell_f``) to each candidate -- via
    :func:`~saealib.strategies.base.assign_tell_f` -- so that
    ``TellStage`` can use them for surrogate-only generations.  Does not
    compute acquisition scores; pair with :class:`AcquisitionStage` for that.

    Parameters
    ----------
    surrogate_manager : SurrogateManager
        Manager that coordinates fit / predict.
    cbmanager : CallbackManager or None
        If provided, SurrogateStartEvent and SurrogateEndEvent are dispatched.
    refit : bool
        Passed directly to ``surrogate_manager.predict()``.
        Set to ``False`` inside inner loops where the surrogate was already
        fitted by an explicit ``SurrogateFitStage``.
    """

    name = "surrogate_predict"
    label = "Surrogate prediction"
    notation = r"$\hat{y} \leftarrow \text{predict}(\mathcal{Q}, \mathcal{A})$"

    def __init__(
        self,
        surrogate_manager: SurrogateManager,
        cbmanager: CallbackManager | None = None,
        *,
        refit: bool = True,
    ) -> None:
        super().__init__()
        self._sm = surrogate_manager
        self._cbmanager = cbmanager
        self._refit = refit

    def execute(self, state: OptimizationState) -> OptimizationState:
        candidates = state.offspring
        assert candidates is not None

        if self._cbmanager is not None:
            self._cbmanager.dispatch(
                SurrogateStartEvent(ctx=state, offspring=candidates)
            )

        prediction = self._sm.predict(
            candidates.x, state.archive, state, refit=self._refit
        )
        if self._refit and self._cbmanager is not None:
            self._cbmanager.dispatch(
                PostSurrogateFitEvent(
                    ctx=state,
                    surrogate=getattr(self._sm, "surrogate", None),
                )
            )
        for i, pred in enumerate(_split_prediction(prediction)):
            assign_tell_f(candidates[i], pred, state)

        if self._cbmanager is not None:
            self._cbmanager.dispatch(SurrogateEndEvent(ctx=state, offspring=candidates))

        return state.replace(offspring=candidates, predictions=prediction)


class AcquisitionStage(Stage):
    """Score offspring via an independent AcquisitionFunction.

    Reads ``state.offspring``/``state.predictions``, writes the resulting
    score array to ``state.scores``.

    Caches ``acquisition.prepare()``'s result per ``(acquisition instance
    identity, generation, archive.value_version, archive.structure_version)``
    A stage instance running the same acquisition
    against an unchanged archive within one generation does not recompute the
    reference, while a mid-generation archive append (structure_version bump)
    or value-only mutation (value_version bump) correctly invalidates it.

    Empty candidate input (``len(state.offspring) == 0``) skips straight to
    an empty ``AcquisitionResult`` without touching the cache or calling
    ``prepare()``/``evaluate()``, so it never advances RNG state.

    Parameters
    ----------
    acquisition : AcquisitionFunction
        Acquisition function that scores ``state.offspring`` against
        ``state.predictions``.
    cbmanager : CallbackManager or None
        If provided, AcquisitionStartEvent and AcquisitionEndEvent are
        dispatched.  AcquisitionEndEvent is not dispatched if
        ``acquisition.evaluate()`` raises.
    """

    name = "acquisition"
    label = "Acquisition scoring"
    notation = (
        r"$\mathbf{s} \leftarrow \text{acquire}(\mathcal{Q}, \hat{y}, \mathcal{A})$"
    )

    def __init__(
        self,
        acquisition: AcquisitionFunction,
        cbmanager: CallbackManager | None = None,
    ) -> None:
        super().__init__()
        self._acquisition = acquisition
        self._cbmanager = cbmanager
        # Single-entry cache: id(self._acquisition) is constant for the life
        # of this stage instance, so only (gen, value_version,
        # structure_version) actually varies -- a growing dict would retain
        # every past generation's prepared reference for as long as this
        # stage instance is reused.
        self._prepared_cache_key: tuple[int, int, int, int] | None = None
        self._prepared_cache_value: object = None

    def execute(self, state: OptimizationState) -> OptimizationState:
        candidates = state.offspring
        assert candidates is not None

        if self._cbmanager is not None:
            self._cbmanager.dispatch(
                AcquisitionStartEvent(ctx=state, offspring=candidates)
            )

        if len(candidates) == 0:
            result = AcquisitionResult(scores=np.empty(0, dtype=np.float64))
        else:
            archive = state.archive
            key = (
                id(self._acquisition),
                state.gen,
                archive.value_version,
                archive.structure_version,
            )
            if key != self._prepared_cache_key:
                self._prepared_cache_value = self._acquisition.prepare(archive, state)
                self._prepared_cache_key = key
            prepared = self._prepared_cache_value

            raw = self._acquisition.evaluate(
                candidates.x, state.predictions, archive, state, prepared=prepared
            )
            if raw.scores is None:
                raise ValidationError(
                    f"{type(self._acquisition).__name__}.evaluate() returned an "
                    "AcquisitionResult with scores=None; AcquisitionStage requires "
                    "a non-None score array."
                )
            scores = np.array(raw.scores, dtype=np.float64, copy=True)
            if scores.shape != (len(candidates),):
                raise ValidationError(
                    f"{type(self._acquisition).__name__}.evaluate() returned "
                    f"scores with shape {scores.shape}, expected "
                    f"({len(candidates)},)."
                )
            result = AcquisitionResult(scores=scores, artifacts=raw.artifacts)

        if self._cbmanager is not None:
            self._cbmanager.dispatch(
                AcquisitionEndEvent(ctx=state, offspring=candidates, result=result)
            )

        return state.replace(scores=result.scores)


class SurrogateFitStage(Stage):
    """Pre-fit the surrogate on the current archive.

    Use this before a surrogate-only inner loop where the archive does not
    change between iterations.  Pass ``refit=False`` to the downstream
    :class:`SurrogatePredictStage` to skip redundant refitting.

    Parameters
    ----------
    surrogate_manager : SurrogateManager
        Manager to pre-fit.
    """

    name = "surrogate_fit"
    label = "Fit surrogate"
    notation = r"$\hat{f} \leftarrow \text{fit}(\mathcal{A})$"

    def __init__(
        self,
        surrogate_manager: SurrogateManager,
        cbmanager: CallbackManager | None = None,
    ) -> None:
        super().__init__()
        self._sm = surrogate_manager
        self._cbmanager = cbmanager

    def execute(self, state: OptimizationState) -> OptimizationState:
        self._sm.fit(state.archive, state)
        if self._cbmanager is not None:
            self._cbmanager.dispatch(
                PostSurrogateFitEvent(
                    ctx=state, surrogate=getattr(self._sm, "surrogate", None)
                )
            )
        return state


class TopKSelectionStage(Stage):
    """Select the top-k offspring by surrogate score.

    Reads ``state.scores`` and ``state.offspring``, replaces
    ``state.offspring`` with the top-k candidates sorted highest-score first.

    Parameters
    ----------
    k : int
        Number of candidates to keep.
    """

    name = "top_k_selection"
    label = "Top-k pre-selection"
    notation = r"$\mathcal{Q} \leftarrow \text{top-}k(\mathcal{Q}, \mathbf{s})$"

    def __init__(self, k: int) -> None:
        super().__init__()
        self._k = k

    def execute(self, state: OptimizationState) -> OptimizationState:
        assert state.offspring is not None
        assert state.scores is not None
        idx = np.argsort(-state.scores)
        selected = state.offspring.extract(idx[: self._k])
        return state.replace(offspring=selected)


class SortByScoreStage(Stage):
    """Sort all offspring by surrogate score descending, keeping every candidate.

    Unlike :class:`TopKSelectionStage`, no candidates are discarded.  Used in
    IB-style strategies where :class:`TellStage` receives *all* offspring sorted
    by score while only a top fraction receives true evaluation.

    Reads ``state.scores`` and ``state.offspring``, returns state with both
    arrays reordered by descending score.
    """

    name = "sort_by_score"
    label = "Sort offspring by score"
    notation = r"$\mathcal{Q} \leftarrow \text{sort\_desc}(\mathcal{Q},\,\mathbf{s})$"

    def execute(self, state: OptimizationState) -> OptimizationState:
        assert state.offspring is not None
        assert state.scores is not None
        idx = np.argsort(-state.scores)
        return state.replace(
            offspring=state.offspring.extract(idx),
            scores=state.scores[idx],
        )


class TrueEvaluationStage(Stage):
    """Evaluate offspring with the true objective function.

    Reads ``state.offspring``, evaluates all candidates, updates their
    ``f / g / cv`` attributes in-place, increments ``state.fe``, and writes
    the evaluated sub-population to ``state.evaluated_offspring``.

    Parameters
    ----------
    evaluator : Evaluator
        Evaluator that calls the true objective function.
    cbmanager : CallbackManager or None
        If provided, PostEvaluationEvent is dispatched after evaluation.
    n_eval : int, callable, or None
        Number of candidates to evaluate from the head of the offspring
        population.  If callable, it receives the current
        :class:`~saealib.context.OptimizationState` and must return an int
        (e.g. ``lambda s: max(1, int(ratio * len(s.offspring)))``).
        ``None`` means evaluate all.
    """

    name = "true_evaluation"
    label = "True objective evaluation"
    notation = r"$\mathcal{Q}_{eval} \leftarrow \text{eval}(\mathcal{Q})$"

    def __init__(
        self,
        evaluator: Evaluator,
        cbmanager: CallbackManager | None = None,
        n_eval: int | Callable[[OptimizationState], int] | None = None,
    ) -> None:
        super().__init__()
        self._evaluator = evaluator
        self._cbmanager = cbmanager
        self._n_eval = n_eval

    def execute(self, state: OptimizationState) -> OptimizationState:
        candidates = state.offspring
        assert candidates is not None
        if self._n_eval is None:
            n = len(candidates)
        elif isinstance(self._n_eval, int):
            n = self._n_eval
        else:
            n = self._n_eval(state)
        n = min(n, len(candidates))

        result = self._evaluator.evaluate_batch(candidates.x[:n], state.problem)
        candidates.update_rows(
            np.arange(n), {"f": result.f, "g": result.g, "cv": result.cv}
        )

        evaluated = candidates.extract(list(range(n)))

        if self._cbmanager is not None:
            self._cbmanager.dispatch(
                PostEvaluationEvent(ctx=state, offspring=evaluated)
            )

        return state.replace(
            fe=state.fe + n,
            offspring=candidates,
            evaluated_offspring=evaluated,
        )


class ArchiveUpdateStage(Stage):
    """Append evaluated offspring to archive and Pareto archive.

    Reads ``state.evaluated_offspring`` and appends each individual to
    ``state.archive`` and ``state.pareto_archive`` (both are controlled
    mutable exceptions — append-only in-place updates).
    """

    name = "archive_update"
    label = "Archive update"
    notation = r"$\mathcal{A} \leftarrow \mathcal{A} \cup \mathcal{Q}_{eval}$"

    def execute(self, state: OptimizationState) -> OptimizationState:
        evaluated = state.evaluated_offspring
        assert evaluated is not None
        has_id = "id" in evaluated.schema
        for i in range(len(evaluated)):
            ind = evaluated[i]
            entry = {"x": ind.x, "f": ind.f, "g": ind.g, "cv": float(ind.cv)}
            if has_id:
                entry["id"] = int(ind.id)
            state.archive.add(entry)
            state.pareto_archive.add(entry)
        return state


class TellStage(Stage):
    """Update the population via the algorithm's tell() method.

    Reads ``state.offspring`` (the full candidate population, including
    both surrogate-scored and true-evaluated individuals, as the algorithm
    expects) and calls ``algorithm.tell()``.

    Parameters
    ----------
    algorithm : Algorithm
        The evolutionary algorithm that updates the population.
    """

    name = "tell"
    label = "Update population"
    notation = r"$P \leftarrow \text{tell}(P, \mathcal{Q})$"

    def __init__(self, algorithm: Algorithm) -> None:
        super().__init__()
        self._algorithm = algorithm
        self._proxy = _DispatchProxy()

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        r"""Expand into per-step lines via ``Algorithm.tell_notation``."""
        prefix = "  " * indent
        tell_notation: list[str] | None = getattr(
            self._algorithm, "tell_notation", None
        )
        if expand and tell_notation:
            label = self.label or self.name
            lines = "\n".join(f"{prefix}  \\State {n}" for n in tell_notation)
            return f"{prefix}\\Comment{{{label}}}\n{lines}"
        return f"{prefix}\\State {self.notation}"

    def execute(self, state: OptimizationState) -> OptimizationState:
        assert state.offspring is not None
        self._algorithm.tell(state, self._proxy, state.offspring)
        return state


class SurrogateOnlyLoopStage(Stage):
    """Run *gen_ctrl* surrogate-only generations before real evaluation.

    Fits the surrogate model once on the current archive, then repeats
    ``gen_ctrl`` times: CountGeneration → Ask → SurrogatePredict(refit=False)
    → Acquisition → Tell.  If *gen_ctrl* is 0 this stage is a no-op.

    Used by :class:`~saealib.strategies.gb.GenerationBasedStrategy` to
    execute inner surrogate-driven generations before a single true-evaluation
    generation.

    Parameters
    ----------
    algorithm : Algorithm
        Evolutionary algorithm for ask/tell.
    surrogate_manager : SurrogateManager
        Manager used for fitting and prediction.
    gen_ctrl : int
        Number of surrogate-only generations.
    cbmanager : CallbackManager or None
        Forwarded to inner stages for event dispatching.
    acquisition : AcquisitionFunction
        Acquisition function used to score offspring inside the inner loop.
        Keyword-only so existing positional
        ``SurrogateOnlyLoopStage(algorithm, surrogate_manager, gen_ctrl,
        cbmanager)`` calls stay valid.
    """

    name = "surrogate_only_loop"
    label = "Surrogate-only generations"
    notation = (
        r"$\text{for}\;i=1\dots gen\_ctrl$: "
        r"$P \leftarrow \mathrm{tell}(P,\,"
        r"\mathrm{acquire}(\mathrm{predict}(\mathrm{ask}(P))))$"
    )

    def __init__(
        self,
        algorithm: Algorithm,
        surrogate_manager: SurrogateManager,
        gen_ctrl: int,
        cbmanager: CallbackManager | None = None,
        *,
        acquisition: AcquisitionFunction,
    ) -> None:
        super().__init__()
        self._gen_ctrl = gen_ctrl
        self._sm = surrogate_manager
        self._cbmanager = cbmanager
        if gen_ctrl > 0:
            self._inner = Pipeline(
                [
                    CountGenerationStage(),
                    AskStage(algorithm, cbmanager=cbmanager),
                    SurrogatePredictStage(
                        surrogate_manager, cbmanager=cbmanager, refit=False
                    ),
                    AcquisitionStage(acquisition, cbmanager=cbmanager),
                    TellStage(algorithm),
                ]
            )
            self.stages = self._inner.stages
        else:
            self._inner = Pipeline([])

    def to_pseudocode(self, *, expand: bool = False, indent: int = 0) -> str:
        r"""Render as a ``\For`` loop block when *expand* is True."""
        prefix = "  " * indent
        if expand and self.stages:
            inner_lines = "\n".join(
                s.to_pseudocode(expand=True, indent=indent + 1) for s in self.stages
            )
            return (
                f"{prefix}\\For{{$i = 1, \\ldots, gen\\_ctrl$}}\n"
                f"{inner_lines}\n"
                f"{prefix}\\EndFor"
            )
        return f"{prefix}\\State {self.notation}"

    def execute(self, state: OptimizationState) -> OptimizationState:
        if self._gen_ctrl > 0:
            self._sm.fit(state.archive, state)
            if self._cbmanager is not None:
                self._cbmanager.dispatch(
                    PostSurrogateFitEvent(
                        ctx=state, surrogate=getattr(self._sm, "surrogate", None)
                    )
                )
            for _ in range(self._gen_ctrl):
                state = self._inner.execute(state)
        return state


class InitializationStage(Stage):
    """Wrap an :class:`~saealib.execution.initializer.Initializer` as a Stage.

    Delegates to ``initializer.initialize(provider, problem)`` and returns the
    resulting :class:`~saealib.context.OptimizationState`.  The *state*
    argument passed to :meth:`execute` is **ignored** — initialization always
    produces a fresh state from scratch.

    This stage is intended for use at the head of a user-defined Pipeline when
    the initializer itself should participate in the pipeline abstraction (e.g.
    to build custom init-then-optimize flows or to inspect / swap the
    initialization step via ``Pipeline["initialization"]``).

    Parameters
    ----------
    initializer : Initializer
        The concrete initializer (e.g.
        :class:`~saealib.execution.initializer.LHSInitializer`).
    provider : ComponentProvider
        Component provider forwarded to ``Initializer.initialize()``.
    problem : Problem
        The optimization problem.
    """

    name = "initialization"
    label = "Initialize population"
    notation = r"$\mathcal{A}_0,\,P_0 \leftarrow \mathrm{init}(n_{\mathrm{init}})$"

    def __init__(
        self,
        initializer: Initializer,
        provider: ComponentProvider,
        problem: Problem,
    ) -> None:
        super().__init__()
        self._initializer = initializer
        self._provider = provider
        self._problem = problem

    def execute(self, state: OptimizationState) -> OptimizationState:
        return self._initializer.initialize(self._provider, self._problem)
