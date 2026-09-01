"""Command-line entry point for experiment sweeps and trial resumption."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from saealib.exceptions import ConfigurationError, ValidationError
from saealib.experiment._config import CONFIG_FILE, ExperimentConfig, TrialSpec
from saealib.experiment._progress import RichProgress, TqdmProgress
from saealib.experiment._runner import ExperimentRunner, resume_trial


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Returns 0 for success, 2 for configuration errors, and 1 for trial errors.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config = ExperimentConfig.from_yaml(args.config)
            if args.workers is not None:
                config = replace(config, n_workers=args.workers)
        else:
            config, trial = _resume_target(args.run)
    except (
        ValidationError,
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        print(f"configuration error: {exc}")
        return 2

    try:
        if args.command == "run":
            ExperimentRunner(config, _progress(args.progress), args.overwrite).run()
        else:
            resume_trial(config, trial)
        return 0
    except (ConfigurationError, ValidationError) as exc:
        print(f"configuration error: {exc}")
        return 2
    except Exception as exc:
        print(f"trial execution failed: {exc}")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="saealib")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--workers", type=int)
    run.add_argument("--progress", choices=("silent", "tqdm", "rich"), default="silent")
    run.add_argument("--overwrite", action="store_true")
    resume = commands.add_parser("resume")
    resume.add_argument("--run", type=Path, required=True)
    return parser


def _progress(name: str):
    if name == "tqdm":
        return TqdmProgress()
    if name == "rich":
        return RichProgress()
    return None


def _resume_target(directory: Path) -> tuple[ExperimentConfig, TrialSpec]:
    """Recover the experiment and trial a result directory belongs to.

    The trial is identified from the directory's ``meta.json`` labels rather
    than from the directory names, which a person can rename or move after
    the fact while the metadata stays a record of what actually ran.
    """
    directory = directory.resolve()
    root = next(
        (
            candidate
            for candidate in (directory, *directory.parents)
            if (candidate / CONFIG_FILE).is_file()
        ),
        None,
    )
    if root is None:
        raise ValidationError(f"Could not find {CONFIG_FILE} above {directory}.")
    config = ExperimentConfig.from_yaml(root / CONFIG_FILE)
    try:
        metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        labels = metadata["labels"]
        problem = str(labels["problem"])
        algorithm = str(labels["algorithm"])
        seed = int(labels["seed"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        raise ValidationError(
            f"Could not read trial metadata from {directory}: {exc}"
        ) from exc
    for trial in config.trials():
        if trial.labels == {
            "problem": problem,
            "algorithm": algorithm,
            "seed": str(seed),
        }:
            return config, trial
    raise ValidationError(
        f"Trial labels {labels!r} are not present in {root / CONFIG_FILE}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
