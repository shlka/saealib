"""Single-trial execution and the files it leaves on disk."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.exceptions import ValidationError
from saealib.execution.history import SUPPORTED_HISTORY_CHANNELS
from saealib.result import Result

if TYPE_CHECKING:
    from saealib.context import OptimizationState
    from saealib.optimizer import Optimizer

ARCHIVE_FILE = "archive.csv"
HISTORY_FILE = "history.csv"
META_FILE = "meta.json"


@dataclass(frozen=True)
class RunResult:
    """Outcome of one trial.

    The dataclass deliberately holds no reference to the optimization state:
    a sweep keeps one of these per trial, so a live archive or population in
    here would make peak memory scale with the number of trials.

    Attributes
    ----------
    seed : int or None
        Seed the trial ran under, or ``None`` when it was left unset.
    fe : int
        True function evaluations consumed.
    gen : int
        Generations completed.
    best_f : np.ndarray
        Best objective values, shaped as :attr:`Result.f`.
    best_x : np.ndarray or None
        Best design variables, or ``None`` for a non-dense space.
    wall_time : float
        Seconds spent inside the optimization run.
    output_dir : Path or None
        Directory the trial's files were written to, or ``None`` when the
        trial was run without persistence.
    labels : Mapping of str to str
        Caller-supplied identifiers for the trial, such as the problem and
        algorithm it belongs to. Empty for a trial run on its own.
    direction : np.ndarray
        Per-objective optimization direction of the problem that ran, kept so
        an aggregate can tell a better score from a worse one without
        reloading the problem.
    """

    seed: int | None
    fe: int
    gen: int
    best_f: np.ndarray
    best_x: np.ndarray | None
    wall_time: float
    output_dir: Path | None
    labels: Mapping[str, str] = field(default_factory=dict)
    direction: np.ndarray = field(default_factory=lambda: np.array([-1.0]))


def run_trial(
    optimizer: Optimizer,
    *,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    labels: Mapping[str, str] | None = None,
    checkpoint_interval: int | None = None,
    result_format: str = "csv",
) -> RunResult:
    """Run one optimization and record it.

    Parameters
    ----------
    optimizer : Optimizer
        Configured optimizer. It is run as given; only *seed* is applied on
        top, so history channels and every component stay the caller's choice.
    seed : int or None, optional
        Seed to run under. ``None`` leaves the optimizer's own seed in place.
    output_dir : str, Path, or None, optional
        Directory to write ``archive.csv``, ``history.csv`` and ``meta.json``
        into, created if missing. ``None`` runs the trial without writing.
    labels : Mapping of str to str, optional
        Identifiers to carry on the result and into ``meta.json``.
    checkpoint_interval : int or None, optional
        Generations between checkpoints.
    result_format : str, optional
        Result file format, either ``"csv"`` or ``"hdf5"``.

    Returns
    -------
    RunResult
        Summary of the trial.

    Notes
    -----
    ``history.csv`` holds the ``summary`` channel alone. The ``front`` and
    ``population`` channels record a variable-length block per generation,
    which a flat CSV cannot represent, so they stay in memory on the returned
    state and are not written here.
    """
    if checkpoint_interval is not None and output_dir is None:
        raise ValidationError(
            "checkpoint_interval requires output_dir so checkpoints have a destination."
        )
    if result_format not in {"csv", "hdf5"}:
        raise ValidationError("result_format must be 'csv' or 'hdf5'.")
    if seed is not None:
        optimizer.set_seed(seed)
    effective_seed = optimizer.seed
    labels = dict(labels or {})

    started = time.perf_counter()
    directory = None if output_dir is None else Path(output_dir)
    if checkpoint_interval is None:
        state = optimizer.run()
    else:
        assert directory is not None
        state = optimizer.run(
            checkpoint_path=directory / "checkpoint",
            checkpoint_interval=checkpoint_interval,
        )
    wall_time = time.perf_counter() - started

    result = Result.from_state(state)
    if directory is not None:
        _write_result(
            directory,
            state,
            seed=effective_seed,
            wall_time=wall_time,
            labels=labels,
            result_format=result_format,
        )

    return RunResult(
        seed=effective_seed,
        fe=result.fe,
        gen=result.gen,
        best_f=result.f,
        best_x=result.x,
        wall_time=wall_time,
        output_dir=directory,
        labels=labels,
        direction=np.asarray(state.problem.direction, dtype=float).copy(),
    )


def _write_result(
    directory: Path,
    state: OptimizationState,
    *,
    seed: int | None,
    wall_time: float,
    labels: Mapping[str, str],
    result_format: str = "csv",
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if result_format == "csv":
        _write_archive(directory / ARCHIVE_FILE, state)
        _write_history(directory / HISTORY_FILE, state)
    elif result_format == "hdf5":
        from saealib.experiment._hdf5 import write_trial

        write_trial(
            directory / "trial.h5", state, seed=seed, wall_time=wall_time, labels=labels
        )
    else:
        raise ValidationError("result_format must be 'csv' or 'hdf5'.")
    _write_meta(
        directory / META_FILE,
        state,
        seed=seed,
        wall_time=wall_time,
        labels=labels,
    )


def _archive_columns(state: OptimizationState) -> dict[str, np.ndarray]:
    """Return the archive's writable columns, skipping ones it does not hold."""
    archive = state.archive
    columns: dict[str, np.ndarray] = {}
    for name in ("id", "x", "f", "g", "cv"):
        try:
            array = archive.get_array(name)
        except (AttributeError, KeyError, ValueError):
            continue
        if array is None:
            continue
        columns[name] = np.asarray(array)
    return columns


def _expand(name: str, array: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return one (header, column) pair per component of *array*."""
    if array.ndim == 1:
        return [(name, array)]
    return [(f"{name}{i}", array[:, i]) for i in range(array.shape[1])]


def _write_archive(path: Path, state: OptimizationState) -> None:
    columns: list[tuple[str, np.ndarray]] = []
    for name, array in _archive_columns(state).items():
        columns.extend(_expand(name, array))
    if not columns:
        return
    height = len(columns[0][1])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([header for header, _ in columns])
        for row in range(height):
            writer.writerow([column[row] for _, column in columns])


def _write_history(path: Path, state: OptimizationState) -> None:
    history = state.history
    if history is None or not history.is_enabled("summary"):
        return
    channel = history.channel("summary")
    columns = sorted(channel)
    if not columns:
        return
    height = len(channel[columns[0]])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in range(height):
            writer.writerow([channel[name][row] for name in columns])


def _write_meta(
    path: Path,
    state: OptimizationState,
    *,
    seed: int | None,
    wall_time: float,
    labels: Mapping[str, str],
) -> None:
    meta = trial_metadata(state, seed=seed, wall_time=wall_time, labels=labels)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def trial_metadata(
    state: OptimizationState,
    *,
    seed: int | None,
    wall_time: float,
    labels: Mapping[str, str],
) -> dict[str, Any]:
    """Return the metadata every result format records for one trial.

    Both the CSV and the HDF5 writer call this, so the two formats cannot
    drift apart on what a trial's index holds.

    Parameters
    ----------
    state : OptimizationState
        State the trial finished on.
    seed : int or None
        Seed the trial ran under.
    wall_time : float
        Seconds spent in the run.
    labels : Mapping of str to str
        Sweep coordinates of the trial.

    Returns
    -------
    dict
        JSON-serializable metadata.
    """
    from saealib import __version__

    history = state.history
    return {
        "labels": dict(labels),
        "seed": seed,
        "fe": int(state.fe),
        "gen": int(state.gen),
        "decision_count": int(state.decision_count),
        "wall_time": wall_time,
        "n_obj": int(state.problem.n_obj),
        "direction": [float(value) for value in np.asarray(state.problem.direction)],
        "archive_size": len(state.archive),
        "pareto_size": len(state.pareto_archive),
        "history_channels": _enabled_channels(history),
        "saealib_version": __version__,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def _enabled_channels(history: Any) -> list[str]:
    """Return enabled channel names without reaching into History internals."""
    if history is None:
        return []
    return sorted(
        name for name in SUPPORTED_HISTORY_CHANNELS if history.is_enabled(name)
    )
