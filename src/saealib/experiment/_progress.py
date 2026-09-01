"""Progress reporters for experiment sweeps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import ModuleType

from saealib.experiment._trial import RunResult


class ProgressReporter(ABC):
    """Report the progress of an experiment sweep."""

    @abstractmethod
    def start(self, total: int) -> None:
        """Start reporting a sweep with *total* trials."""

    @abstractmethod
    def advance(self, result: RunResult) -> None:
        """Report one completed trial."""

    @abstractmethod
    def finish(self) -> None:
        """Finish reporting the sweep."""


class SilentProgress(ProgressReporter):
    """Report no progress."""

    def start(self, total: int) -> None:
        """Start without producing output."""

    def advance(self, result: RunResult) -> None:
        """Ignore one completed trial."""

    def finish(self) -> None:
        """Finish without producing output."""


def _require_tqdm() -> ModuleType:
    try:
        import tqdm
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Tqdm progress requires tqdm. Install it with `pip install saealib[tqdm]`."
        ) from exc
    return tqdm


def _require_rich() -> ModuleType:
    try:
        import rich.progress
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Rich progress requires rich. Install it with `pip install saealib[rich]`."
        ) from exc
    return rich.progress


class TqdmProgress(ProgressReporter):
    """Report sweep progress with tqdm."""

    def __init__(self) -> None:
        self._tqdm = _require_tqdm().tqdm
        self._bar = None

    def start(self, total: int) -> None:
        """Start a tqdm progress bar."""
        self._bar = self._tqdm(total=total)

    def advance(self, result: RunResult) -> None:
        """Advance the tqdm progress bar."""
        if self._bar is not None:
            self._bar.update(1)

    def finish(self) -> None:
        """Close the tqdm progress bar."""
        if self._bar is not None:
            self._bar.close()


class RichProgress(ProgressReporter):
    """Report sweep progress with rich."""

    def __init__(self) -> None:
        progress = _require_rich()
        self._progress = progress.Progress()
        self._task_id = None

    def start(self, total: int) -> None:
        """Start a rich progress bar."""
        self._progress.start()
        self._task_id = self._progress.add_task("Experiment", total=total)

    def advance(self, result: RunResult) -> None:
        """Advance the rich progress bar."""
        if self._task_id is not None:
            self._progress.advance(self._task_id)

    def finish(self) -> None:
        """Stop the rich progress bar."""
        self._progress.stop()
