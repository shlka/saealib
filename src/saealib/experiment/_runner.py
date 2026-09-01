"""Execution of a whole sweep of trials."""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.experiment._config import (
    CONFIG_FILE,
    ExperimentConfig,
    TrialSpec,
    _resolve_trial_directory,
)
from saealib.experiment._progress import ProgressReporter, SilentProgress
from saealib.experiment._summary import write_aggregate, write_summary
from saealib.experiment._trial import RunResult, _write_result, run_trial
from saealib.result import Result


def execute_trial(
    config: ExperimentConfig, trial: TrialSpec, overwrite: bool = False
) -> RunResult:
    """Build and run one trial of *config*.

    Defined at module level and taking only picklable arguments so a worker
    process can call it. The optimizer is built inside the call rather than
    passed in, which is what lets a problem defined with a lambda take part
    in a parallel sweep at all.

    Parameters
    ----------
    config : ExperimentConfig
        The experiment the trial belongs to.
    trial : TrialSpec
        The sweep point to run.

    Returns
    -------
    RunResult
        Outcome of the trial.
    """
    directory = _validated_trial_directory(Path(config.output_dir), trial)
    if overwrite and directory.exists():
        shutil.rmtree(directory)
    optimizer = config.build_optimizer(trial)
    return run_trial(
        optimizer,
        seed=trial.seed,
        output_dir=directory,
        labels=trial.labels,
        checkpoint_interval=config.checkpoint_interval,
        result_format=config.result_format,
    )


def latest_checkpoint(directory: Path) -> Path:
    """Return the checkpoint holding the most generations.

    Parameters
    ----------
    directory : Path
        Checkpoint directory written during the trial.

    Returns
    -------
    Path
        The newest ``checkpoint_<gen>.npz``, selected by its integer
        generation number.

    Raises
    ------
    ValidationError
        If the directory holds no checkpoint.
    """
    checkpoints = list(Path(directory).glob("checkpoint_*.npz"))
    if not checkpoints:
        raise ValidationError(f"No checkpoint found in {directory}.")
    pattern = re.compile(r"^checkpoint_(\d+)\.npz$")
    numbered = [
        (int(match.group(1)), path)
        for path in checkpoints
        if (match := pattern.match(path.name))
    ]
    if not numbered:
        raise ValidationError(f"No checkpoint found in {directory}.")
    return max(numbered, key=lambda item: item[0])[1]


def resume_trial(config: ExperimentConfig, trial: TrialSpec) -> RunResult:
    """Resume a trial from its newest portable checkpoint.

    If ``meta.json`` is absent or malformed, the returned wall time covers
    only the resumed interval; otherwise it includes the prior wall time.
    """
    optimizer = config.build_optimizer(trial)
    optimizer.resolve_defaults()
    directory = _validated_trial_directory(Path(config.output_dir), trial)
    ctx = OptimizationState.load(
        latest_checkpoint(directory / "checkpoint"), optimizer.problem
    )
    try:
        metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        recorded = float(metadata["wall_time"])
    except (OSError, ValueError, TypeError, KeyError):
        recorded = 0.0
    previous_wall_time = recorded if math.isfinite(recorded) else 0.0
    started = time.perf_counter()
    state = optimizer.run_from(
        ctx,
        checkpoint_path=directory / "checkpoint",
        checkpoint_interval=config.checkpoint_interval or 1,
    )
    wall_time = previous_wall_time + time.perf_counter() - started
    _write_result(
        directory,
        state,
        seed=trial.seed,
        wall_time=wall_time,
        labels=trial.labels,
        result_format=config.result_format,
    )
    result = Result.from_state(state)
    return RunResult(
        seed=trial.seed,
        fe=result.fe,
        gen=result.gen,
        best_f=result.f,
        best_x=result.x,
        wall_time=wall_time,
        output_dir=directory,
        labels=trial.labels,
        direction=state.problem.direction.copy(),
    )


class ExperimentRunner:
    """Run every trial of an experiment and collect the results.

    Parameters
    ----------
    config : ExperimentConfig
        The experiment to run.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        progress: ProgressReporter | None = None,
        overwrite: bool = False,
    ):
        self.config = config
        self.progress = SilentProgress() if progress is None else progress
        self.overwrite = overwrite

    def run(self) -> list[RunResult]:
        """Run the sweep and return one result per trial.

        A snapshot of the configuration is written to the experiment root
        before anything runs, so the results stay readable once the source
        configuration has moved on.

        Trial-level parallelism is used when ``config.n_workers`` exceeds one.
        A trial that raises propagates: a sweep that silently drops trials
        would report a comparison that was never actually run.

        Summary and aggregate files are written to the experiment root.

        Returns
        -------
        list of RunResult
            Results in sweep order, whatever order the trials finished in.
        """
        trials = self.config.trials()
        root = Path(self.config.output_dir)
        trial_directories = [
            _validated_trial_directory(root, trial) for trial in trials
        ]
        existing = [
            directory
            for directory in trial_directories
            if directory.is_dir() and any(directory.iterdir())
        ]
        if existing and not self.overwrite:
            raise ValidationError(
                f"{len(existing)} existing trial(s) are non-empty; "
                "rerun with --overwrite."
            )
        root.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(root / CONFIG_FILE)

        results: list[RunResult | None] = [None] * len(trials)
        try:
            self.progress.start(len(trials))
            if self.config.n_workers == 1:
                for index, trial in enumerate(trials):
                    result = execute_trial(self.config, trial, self.overwrite)
                    results[index] = result
                    self.progress.advance(result)
            else:
                with ProcessPoolExecutor(max_workers=self.config.n_workers) as pool:
                    futures = {
                        pool.submit(
                            execute_trial, self.config, trial, self.overwrite
                        ): index
                        for index, trial in enumerate(trials)
                    }
                    for future in as_completed(futures):
                        result = future.result()
                        results[futures[future]] = result
                        self.progress.advance(result)
            completed = [result for result in results if result is not None]
            summary = write_summary(completed, root, self.config.indicator)
            write_aggregate(summary, completed, self.config.indicator)
            return completed
        finally:
            self.progress.finish()


def _validated_trial_directory(root: Path, trial: TrialSpec) -> Path:
    return _resolve_trial_directory(root, trial)
