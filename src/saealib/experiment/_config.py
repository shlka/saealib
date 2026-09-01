"""Declarative configuration for a multi-trial experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from saealib.exceptions import ValidationError
from saealib.execution.history import SUPPORTED_HISTORY_CHANNELS

if TYPE_CHECKING:
    from saealib.optimizer import Optimizer

CONFIG_FILE = "config.yaml"
INDICATOR_SPECS = {
    "hypervolume": "reference_point",
    "spacing": None,
    "gd": "reference_front",
    "gd_plus": "reference_front",
    "igd": "reference_front",
    "igd_plus": "reference_front",
    "spread": "reference_front",
}


@dataclass(frozen=True)
class AlgorithmEntry:
    """One algorithm to compare, named so results can be told apart.

    Attributes
    ----------
    name : str
        Label used for the result directory and the summary column.
    preset : str, Path, dict, or None
        Preset applied when building the optimizer, in any form
        :meth:`Optimizer.set_preset` accepts. ``None`` leaves the problem
        file's own components and saealib's defaults in place.
    """

    name: str
    preset: str | Path | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.preset, Path):
            object.__setattr__(self, "preset", self.preset.resolve())


@dataclass(frozen=True)
class TrialSpec:
    """One point of the sweep: a problem, an algorithm and a seed."""

    problem: Path
    algorithm: AlgorithmEntry
    seed: int

    @property
    def labels(self) -> dict[str, str]:
        """Return the sweep coordinates as string labels."""
        return {
            "problem": self.problem.stem,
            "algorithm": self.algorithm.name,
            "seed": str(self.seed),
        }

    @property
    def relative_dir(self) -> Path:
        """Return this trial's directory, relative to the experiment root."""
        return Path(self.problem.stem) / self.algorithm.name / f"seed{self.seed}"


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to run and lay out a sweep.

    Objective and constraint functions are callables and cannot be written in
    YAML, so a problem is named by the path of a ``.py`` file defining a
    top-level ``problem`` variable, the form
    :meth:`Optimizer.from_problem_file` already reads. Component choices stay
    in YAML through the preset mechanism.

    Attributes
    ----------
    problems : sequence of Path
        Problem files to sweep over. Normalized to a tuple.
    algorithms : sequence of AlgorithmEntry
        Algorithm configurations to sweep over. Normalized to a tuple.
    seeds : sequence of int
        Seeds to sweep over. Normalized to a tuple.
    output_dir : Path
        Root directory results are written under.
    termination : Mapping or None
        Shared budget as ``{condition_name: value}``, for example
        ``{"max_fe": 500}``. Conditions are combined with OR, matching
        :class:`~saealib.termination.Termination`. When given it replaces the
        termination a problem file or preset sets, so every trial in the
        sweep runs on one budget.
    n_workers : int
        Trials to run at once.
    history_channels : sequence of str
        History channels recorded for every trial. Normalized to a tuple.
    indicator : Mapping or None
        Indicator selection and parameters used to aggregate a
        multi-objective trial, for example
        ``{"type": "hypervolume", "params": {"reference_point": [11.0, 11.0]}}``.
        ``None`` leaves multi-objective runs without an aggregate column.
    checkpoint_interval : int or None
        Generations between checkpoints, or ``None`` to disable checkpoints.
    result_format : str
        Result file format, either ``"csv"`` or ``"hdf5"``.
    """

    problems: Sequence[Path]
    algorithms: Sequence[AlgorithmEntry]
    seeds: Sequence[int]
    output_dir: Path
    termination: Mapping[str, Any] | None = None
    n_workers: int = 1
    history_channels: Sequence[str] = ("summary",)
    indicator: Mapping[str, Any] | None = None
    checkpoint_interval: int | None = None
    result_format: str = "csv"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "problems", tuple(Path(item).resolve() for item in self.problems)
        )
        object.__setattr__(self, "algorithms", tuple(self.algorithms))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())
        object.__setattr__(self, "history_channels", tuple(self.history_channels))
        if not self.problems:
            raise ValidationError("An experiment needs at least one problem.")
        if not self.algorithms:
            raise ValidationError("An experiment needs at least one algorithm.")
        if not self.seeds:
            raise ValidationError("An experiment needs at least one seed.")
        names = [entry.name for entry in self.algorithms]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValidationError(
                f"Algorithm names must be unique; repeated: {', '.join(duplicates)}."
            )
        stems = [path.stem for path in self.problems]
        repeated = sorted({stem for stem in stems if stems.count(stem) > 1})
        if repeated:
            raise ValidationError(
                "Problem file names must be unique, since they name the result "
                f"directories; repeated: {', '.join(repeated)}."
            )
        unknown = sorted(set(self.history_channels) - SUPPORTED_HISTORY_CHANNELS)
        if unknown:
            raise ValidationError(f"Unknown history channel(s): {', '.join(unknown)}.")
        if self.n_workers < 1:
            raise ValidationError("n_workers must be at least 1.")
        if self.checkpoint_interval is not None and self.checkpoint_interval < 1:
            raise ValidationError("checkpoint_interval must be at least 1.")
        if self.result_format not in {"csv", "hdf5"}:
            raise ValidationError("result_format must be 'csv' or 'hdf5'.")
        _validate_indicator(self.indicator)
        _validate_trial_directories(self.output_dir, self.trials())

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Read a configuration, resolving its paths against the file's own.

        Parameters
        ----------
        path : str or Path
            Configuration file to read.

        Returns
        -------
        ExperimentConfig
            The parsed configuration.
        """
        source = Path(path)
        try:
            data = yaml.safe_load(source.read_text("utf-8"))
        except yaml.YAMLError as exc:
            raise ValidationError(f"{source} is not valid YAML: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ValidationError(f"{source} must hold a mapping at the top level.")
        return cls._from_mapping(data, base=source.parent)

    @classmethod
    def _from_mapping(cls, data: Mapping[str, Any], *, base: Path) -> ExperimentConfig:
        known = {
            "problems",
            "algorithms",
            "seeds",
            "output_dir",
            "termination",
            "n_workers",
            "history_channels",
            "indicator",
            "checkpoint_interval",
            "result_format",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValidationError(
                f"Unknown configuration key(s): {', '.join(unknown)}. "
                f"Known keys are {sorted(known)}."
            )
        missing = sorted({"problems", "algorithms", "seeds", "output_dir"} - set(data))
        if missing:
            raise ValidationError(
                f"Missing required configuration key(s): {', '.join(missing)}."
            )
        return cls(
            problems=tuple(
                _resolve(base, item)
                for item in _as_sequence(data["problems"], "problems")
            ),
            algorithms=tuple(
                _algorithm_entry(item, base=base)
                for item in _as_sequence(data["algorithms"], "algorithms")
            ),
            seeds=tuple(int(seed) for seed in _as_sequence(data["seeds"], "seeds")),
            output_dir=_resolve(base, data["output_dir"]),
            termination=_as_mapping(data.get("termination"), "termination"),
            n_workers=int(data.get("n_workers", 1)),
            history_channels=tuple(data.get("history_channels", ("summary",))),
            indicator=_as_mapping(data.get("indicator"), "indicator"),
            checkpoint_interval=(
                None
                if data.get("checkpoint_interval") is None
                else int(data["checkpoint_interval"])
            ),
            result_format=str(data.get("result_format", "csv")),
        )

    def to_yaml(self, path: str | Path) -> Path:
        """Write the configuration out, with paths relative to the file.

        Parameters
        ----------
        path : str or Path
            Destination. A ``.yaml`` suffix is added when absent.

        Returns
        -------
        Path
            The path written to.
        """
        destination = Path(path)
        if destination.suffix not in {".yaml", ".yml"}:
            destination = destination.with_suffix(".yaml")
        destination.parent.mkdir(parents=True, exist_ok=True)
        base = destination.parent
        data: dict[str, Any] = {
            "problems": [_relative(base, item) for item in self.problems],
            "algorithms": [
                _algorithm_data(entry, base=base) for entry in self.algorithms
            ],
            "seeds": [int(seed) for seed in self.seeds],
            "output_dir": _relative(base, self.output_dir),
            "n_workers": self.n_workers,
            "history_channels": list(self.history_channels),
            "result_format": self.result_format,
        }
        if self.checkpoint_interval is not None:
            data["checkpoint_interval"] = self.checkpoint_interval
        if self.termination is not None:
            data["termination"] = dict(self.termination)
        if self.indicator is not None:
            data["indicator"] = dict(self.indicator)
        destination.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return destination

    def trials(self) -> list[TrialSpec]:
        """Expand the sweep into one specification per trial.

        Returns
        -------
        list of TrialSpec
            The problem-algorithm-seed product, in problem-major order so a
            partially finished sweep still covers whole problems.
        """
        return [
            TrialSpec(problem, algorithm, seed)
            for problem in self.problems
            for algorithm in self.algorithms
            for seed in self.seeds
        ]

    def build_optimizer(self, trial: TrialSpec) -> Optimizer:
        """Build the optimizer one trial runs on.

        Parameters
        ----------
        trial : TrialSpec
            The sweep point to build for.

        Returns
        -------
        Optimizer
            A configured, not-yet-run optimizer.
        """
        from saealib.optimizer import Optimizer

        preset = trial.algorithm.preset
        optimizer = Optimizer.from_problem_file(
            trial.problem,
            preset=dict(preset) if isinstance(preset, Mapping) else preset,
        )
        optimizer.set_seed(trial.seed)
        optimizer.set_history(list(self.history_channels))
        if self.termination is not None:
            optimizer.set_termination(build_termination(self.termination))
        return optimizer


def build_termination(spec: Mapping[str, Any]):
    """Build a Termination from ``{condition_name: value}`` entries.

    Parameters
    ----------
    spec : Mapping
        Condition names mapped to a scalar argument, or to a mapping of
        keyword arguments for conditions that take several.

    Returns
    -------
    Termination
        The conditions combined with OR.
    """
    from saealib.registry import build
    from saealib.termination import Termination

    if not spec:
        raise ValidationError("termination needs at least one condition.")
    conditions = []
    for name, value in spec.items():
        params = dict(value) if isinstance(value, Mapping) else [value]
        conditions.append(build({"type": name, "params": params}))
    return Termination(*conditions)


def _as_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValidationError(f"{field} must be a list.")
    return value


def _as_mapping(value: Any, field: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field} must be a mapping.")
    return dict(value)


def _validate_indicator(indicator: Any) -> None:
    if indicator is None:
        return
    if not isinstance(indicator, Mapping):
        raise ValidationError("indicator must be a mapping.")
    name = indicator.get("type")
    if not isinstance(name, str) or name not in INDICATOR_SPECS:
        raise ValidationError(f"Unknown indicator: {name!r}.")
    params = indicator.get("params", {})
    if not isinstance(params, Mapping):
        raise ValidationError("indicator params must be a mapping.")
    required = INDICATOR_SPECS[name]
    if required is not None and required not in params:
        raise ValidationError(f"Indicator {name!r} requires parameter {required!r}.")


def _resolve(base: Path, value: Any) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _resolve_trial_directory(root: Path, trial: TrialSpec) -> Path:
    root = root.resolve()
    problem_root = (root / trial.problem.stem).resolve()
    algorithm_directory = (problem_root / trial.algorithm.name).resolve()
    try:
        algorithm_directory.relative_to(problem_root)
    except ValueError as exc:
        raise ValidationError(
            f"Trial {trial.labels!r} has an algorithm path outside its problem "
            f"directory: {algorithm_directory}."
        ) from exc

    directory = (root / trial.relative_dir).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ValidationError(
            f"Trial {trial.labels!r} resolves outside experiment root: {directory}."
        ) from exc
    return directory


def _validate_trial_directories(root: Path, trials: Sequence[TrialSpec]) -> None:
    directories: dict[Path, TrialSpec] = {}
    for trial in trials:
        directory = _resolve_trial_directory(root, trial)
        previous = directories.get(directory)
        if previous is not None:
            raise ValidationError(
                "Trial directory collision: "
                f"trials {previous.labels!r} and {trial.labels!r} both resolve "
                f"to {directory}."
            )
        directories[directory] = trial


def _relative(base: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _algorithm_entry(item: Any, *, base: Path) -> AlgorithmEntry:
    if isinstance(item, str):
        return AlgorithmEntry(name=item, preset=item)
    if not isinstance(item, Mapping):
        raise ValidationError("Each algorithm must be a name or a mapping.")
    unknown = sorted(set(item) - {"name", "preset"})
    if unknown:
        raise ValidationError(
            f"Unknown algorithm key(s): {', '.join(unknown)}. Known keys are "
            "['name', 'preset']."
        )
    preset = item.get("preset")
    if isinstance(preset, str) and preset.endswith((".yaml", ".yml")):
        preset = _resolve(base, preset)
    name = item.get("name")
    if name is None:
        if not isinstance(preset, str):
            raise ValidationError(
                "An algorithm needs a name unless its preset is a bundled "
                "preset name that can serve as one."
            )
        name = preset
    return AlgorithmEntry(name=str(name), preset=preset)


def _algorithm_data(entry: AlgorithmEntry, *, base: Path) -> dict[str, Any]:
    data: dict[str, Any] = {"name": entry.name}
    preset = entry.preset
    if isinstance(preset, Path):
        data["preset"] = _relative(base, preset)
    elif isinstance(preset, Mapping):
        data["preset"] = dict(preset)
    elif preset is not None:
        data["preset"] = preset
    return data
