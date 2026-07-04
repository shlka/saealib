"""Loader for the bundled component-spec presets (see ``presets.yaml``)."""

from __future__ import annotations

import functools
from importlib import resources
from typing import Any

import yaml


@functools.lru_cache(maxsize=1)
def load_defaults() -> dict[str, Any]:
    """Load and cache the bundled defaults/presets file as a plain dict."""
    text = (
        resources.files("saealib.defaults").joinpath("presets.yaml").read_text("utf-8")
    )
    return yaml.safe_load(text)
