"""Concrete Stage implementations for the optimization pipeline.

Each stage receives an OptimizationState, performs one well-defined operation,
and returns an updated state via ``state.replace()``.

Standard pipeline fields on OptimizationState
----------------------------------------------
``offspring``
    Current candidate population (Population), set by AskStage.
``scores``
    1-D acquisition score array (np.ndarray), set by SurrogateScoreStage.
``predictions``
    Per-candidate SurrogatePrediction list, set by SurrogateScoreStage.
``evaluated_offspring``
    Sub-population with true objective values, set by TrueEvaluationStage.

Custom stages may store additional values in ``state.data`` (user-extensible
dict) via ``state.replace(data={**state.data, "key": value})``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from saealib.callback import (
    PostEvaluationEvent,
    SurrogateEndEvent,
    SurrogateStartEvent,
)
from saealib.pipeline import Pipeline, Stage
from saealib.strategies.base import assign_tell_f

if TYPE_CHECKING:
    from collections.abc import Callable

    from saealib.algorithms.base import Algorithm
    from saealib.callback import CallbackManager
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

    def dispatch(self, event: object) -> None:
        if self._cbmanager is not None:
            self._cbmanager.dispatch(event)  # type: ignore[arg-type]

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
        return state.replace(offspring=candidates)


class SurrogateScoreStage(Stage):
    """Score offspring with the surrogate model.

    Reads ``state.offspring``, writes scores and predictions to
    ``state.scores`` and ``state.predictions``.  Also assigns predicted
    objective values (``tell_f``) to each candidate so that ``TellStage``
    can use them for surrogate-only generations.

    Parameters
    ----------
    surrogate_manager : SurrogateManager
        Manager that coordinates fit / predict / score.
    cbmanager : CallbackManager or None
        If provided, SurrogateStartEvent and SurrogateEndEvent are dispatched.
    refit : bool
        Passed directly to ``surrogate_manager.score_candidates()``.
        Set to ``False`` inside inner loops where the surrogate was already
        fitted by an explicit ``SurrogateFitStage``.
    """

    name = "surrogate_score"
    label = "Surrogate scoring"
    notation = r"$\mathbf{s} \leftarrow \text{score}(\mathcal{Q}, \mathcal{A})$"

    def __init__(
        self,
        surrogate_manager: SurrogateManager,
        cbmanager: CallbackManager | None = None,
        *,
        refit: bool = True,
    ) -> None:
        self._sm = surrogate_manager
        self._cbmanager = cbmanager
        self._refit = refit

    def execute(self, state: OptimizationState) -> OptimizationState:
        candidates = state.offspring

        if self._cbmanager is not None:
            self._cbmanager.dispatch(
                SurrogateStartEvent(ctx=state, offspring=candidates)
            )

        scores, predictions = self._sm.score_candidates(
            candidates.x, state.archive, state, refit=self._refit
        )
        for i, pred in enumerate(predictions):
            assign_tell_f(candidates[i], pred, state)

        if self._cbmanager is not None:
            self._cbmanager.dispatch(SurrogateEndEvent(ctx=state, offspring=candidates))

        return state.replace(
            offspring=candidates, scores=scores, predictions=predictions
        )


class SurrogateFitStage(Stage):
    """Pre-fit the surrogate on the current archive.

    Use this before a surrogate-only inner loop where the archive does not
    change between iterations.  Pass ``refit=False`` to the downstream
    :class:`SurrogateScoreStage` to skip redundant refitting.

    Parameters
    ----------
    surrogate_manager : SurrogateManager
        Manager to pre-fit.
    """

    name = "surrogate_fit"
    label = "Fit surrogate"
    notation = r"$\hat{f} \leftarrow \text{fit}(\mathcal{A})$"

    def __init__(self, surrogate_manager: SurrogateManager) -> None:
        self._sm = surrogate_manager

    def execute(self, state: OptimizationState) -> OptimizationState:
        self._sm.fit(state.archive, state)
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
        self._k = k

    def execute(self, state: OptimizationState) -> OptimizationState:
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
        self._evaluator = evaluator
        self._cbmanager = cbmanager
        self._n_eval = n_eval

    def execute(self, state: OptimizationState) -> OptimizationState:
        candidates = state.offspring
        if callable(self._n_eval):
            n = self._n_eval(state)
        elif self._n_eval is not None:
            n = self._n_eval
        else:
            n = len(candidates)
        n = min(n, len(candidates))

        result = self._evaluator.evaluate_batch(candidates.x[:n], state.problem)
        for i in range(n):
            candidates[i].f = result.f[i]
            candidates[i].g = result.g[i]
            candidates[i].cv = float(result.cv[i])

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
        for i in range(len(evaluated)):
            ind = evaluated[i]
            entry = {"x": ind.x, "f": ind.f, "g": ind.g, "cv": float(ind.cv)}
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
        self._algorithm = algorithm
        self._proxy = _DispatchProxy()

    def execute(self, state: OptimizationState) -> OptimizationState:
        self._algorithm.tell(state, self._proxy, state.offspring)
        return state


class SurrogateOnlyLoopStage(Stage):
    """Run *gen_ctrl* surrogate-only generations before real evaluation.

    Fits the surrogate model once on the current archive, then repeats
    ``gen_ctrl`` times: CountGeneration → Ask → SurrogateScore(refit=False)
    → Tell.  If *gen_ctrl* is 0 this stage is a no-op.

    Used by :class:`~saealib.strategies.gb.GenerationBasedStrategy` to
    execute inner surrogate-driven generations before a single true-evaluation
    generation.

    Parameters
    ----------
    algorithm : Algorithm
        Evolutionary algorithm for ask/tell.
    surrogate_manager : SurrogateManager
        Manager used for fitting and scoring.
    gen_ctrl : int
        Number of surrogate-only generations.
    cbmanager : CallbackManager or None
        Forwarded to inner stages for event dispatching.
    """

    name = "surrogate_only_loop"
    label = "Surrogate-only generations"
    notation = (
        r"$\text{for}\;i=1\dots gen\_ctrl$: "
        r"$P \leftarrow \mathrm{tell}(P,\,\mathrm{score}(\mathrm{ask}(P)))$"
    )

    def __init__(
        self,
        algorithm: Algorithm,
        surrogate_manager: SurrogateManager,
        gen_ctrl: int,
        cbmanager: CallbackManager | None = None,
    ) -> None:
        self._gen_ctrl = gen_ctrl
        self._sm = surrogate_manager
        if gen_ctrl > 0:
            self._inner: Pipeline | None = Pipeline(
                [
                    CountGenerationStage(),
                    AskStage(algorithm, cbmanager=cbmanager),
                    SurrogateScoreStage(
                        surrogate_manager, cbmanager=cbmanager, refit=False
                    ),
                    TellStage(algorithm),
                ]
            )
            self.stages = self._inner.stages
        else:
            self._inner = None
            self.stages = []

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
            for _ in range(self._gen_ctrl):
                state = self._inner.execute(state)  # type: ignore[union-attr]
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
        self._initializer = initializer
        self._provider = provider
        self._problem = problem

    def execute(self, state: OptimizationState) -> OptimizationState:
        return self._initializer.initialize(self._provider, self._problem)
