"""Contract tests for dynamic-environment named archives."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from saealib.callback import GenerationStartEvent, logging_generation


def _load_example():
    path = Path(__file__).parents[1] / "examples" / "generality_dynamic_archive.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("generality_dynamic_archive", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_archives_are_distinct_and_keep_distinct_rows():
    result = _load_example().main()
    ctx = result["state"]
    env_0 = ctx.archives["env_0"]
    env_10 = ctx.archives["env_10"]

    assert env_0 is not env_10
    assert len(env_0) > 0
    assert len(env_10) > 0
    assert not np.array_equal(env_0.get_array("x"), env_10.get_array("x"))


def test_active_archive_selection_comes_from_state():
    result = _load_example().main()
    ctx = result["state"]

    assert result["selected_name"] == "env_10"
    assert result["selected_name"] != "env_0"
    assert result["selected"] is ctx.archives[result["selected_name"]]
    assert ctx.archive is ctx.archives["env_10"]
    assert ctx.fe == 12
    assert result["fe_before_third"] < ctx.fe
    assert len(ctx.archives["env_10"]) > result["selected_size_before_third"]


def test_logging_generation_allows_an_empty_single_objective_archive():
    result = _load_example().main()
    ctx = result["state"]
    ctx.archive = type(ctx.archive)(ctx.archive.attrs, duplicate_policy="keep_first")

    logging_generation(GenerationStartEvent(ctx=ctx))
