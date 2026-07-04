"""Loader for the bundled component-spec presets (see ``presets.yaml``)."""

from __future__ import annotations

import functools
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from saealib.exceptions import ValidationError

_PRESET_KEYS = {
    "schema_version",
    "algorithm",
    "surrogate_manager",
    "strategy",
    "termination",
}


@functools.lru_cache(maxsize=1)
def load_defaults() -> dict[str, Any]:
    """Load and cache the bundled defaults/presets file as a plain dict."""
    text = (
        resources.files("saealib.defaults").joinpath("presets.yaml").read_text("utf-8")
    )
    return yaml.safe_load(text)


def load_preset(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load and validate a user-defined preset.

    Parameters
    ----------
    source : str, Path, or dict
        A path to a YAML file, or an already-parsed preset dict.

    Returns
    -------
    dict
        The validated preset dict.

    Raises
    ------
    ValidationError
        If the source is not a mapping, ``schema_version`` is not 1, or an
        unknown top-level key is present.
    """
    if isinstance(source, str | Path):
        preset = yaml.safe_load(Path(source).read_text("utf-8"))
    else:
        preset = source
    if not isinstance(preset, dict):
        raise ValidationError(f"Preset must be a mapping, got {type(preset).__name__}.")
    schema_version = preset.get("schema_version")
    if schema_version is not None and schema_version != 1:
        raise ValidationError(f"Unsupported preset schema_version: {schema_version!r}.")
    unknown = set(preset) - _PRESET_KEYS
    if unknown:
        raise ValidationError(
            f"Unknown preset key(s): {sorted(unknown)}. Allowed keys: "
            f"{sorted(_PRESET_KEYS)}."
        )
    return preset


def dump_preset(preset: dict[str, Any], path: str | Path) -> Path:
    """Write a preset dict to a YAML file.

    Parameters
    ----------
    preset : dict
        The preset dict to write. ``schema_version: 1`` is added if absent.
    path : str or Path
        Destination file path. The ``.yaml`` extension is added if absent.

    Returns
    -------
    Path
        The path the preset was written to.
    """
    p = Path(path)
    if not p.suffix:
        p = p.with_suffix(".yaml")
    preset = {"schema_version": preset.get("schema_version", 1), **preset}
    p.write_text(yaml.safe_dump(preset, sort_keys=False), encoding="utf-8")
    return p
