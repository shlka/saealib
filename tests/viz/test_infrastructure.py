"""Tests for shared :mod:`saealib.viz` helpers."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.execution.history import History


def test_import_does_not_import_matplotlib() -> None:
    # Run in a fresh interpreter so an already-imported matplotlib in this
    # process cannot mask the behavior.
    code = (
        "import sys\n"
        "import saealib.viz\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib was imported'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_matplotlib_actionable_error() -> None:
    from saealib.viz._matplotlib import _require_matplotlib, _require_pyplot

    class _MatplotlibBlocker:
        def find_spec(self, name: str, path: object, target: object = None):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    blocker = _MatplotlibBlocker()
    saved = sys.modules.pop("matplotlib", None)
    sys.meta_path.insert(0, blocker)
    try:
        for require in (_require_matplotlib, _require_pyplot):
            with pytest.raises(ImportError) as excinfo:
                require()
            assert "pip install saealib[viz]" in str(excinfo.value)
    finally:
        if blocker in sys.meta_path:
            sys.meta_path.remove(blocker)
        if saved is not None:
            sys.modules["matplotlib"] = saved


def test_resolve_axes_creates_when_none() -> None:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from saealib.viz._common import _resolve_axes

    fig, ax = _resolve_axes()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax.figure is fig


def test_resolve_axes_uses_provided() -> None:
    from matplotlib.figure import Figure

    from saealib.viz._common import _resolve_axes

    fig = Figure()
    ax = fig.add_subplot(111)
    out_fig, out_ax = _resolve_axes(ax=ax)
    assert out_fig is fig
    assert out_ax is ax


def test_resolve_axes_returns_root_figure_for_subfigure_axes() -> None:
    from matplotlib.figure import Figure, SubFigure

    from saealib.viz._common import _resolve_axes

    fig = Figure()
    subfig = fig.subfigures(1)
    assert isinstance(subfig, SubFigure)
    ax = subfig.add_subplot(111)
    out_fig, out_ax = _resolve_axes(ax=ax)
    assert out_fig is fig
    assert out_ax is ax


def test_resolve_axes_projection_when_creating() -> None:
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the "3d" projection)
    from matplotlib.figure import Figure

    from saealib.viz._common import _resolve_axes

    fig, ax = _resolve_axes(projection="3d")
    assert isinstance(fig, Figure)
    assert ax.name == "3d"


def test_resolve_axes_rejects_projection_mismatch_when_provided() -> None:
    from matplotlib.figure import Figure

    from saealib.viz._common import _resolve_axes

    fig = Figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValidationError, match=r'projection="3d"') as excinfo:
        _resolve_axes(ax=ax, projection="3d")
    assert 'fig.add_subplot(111, projection="3d")' in str(excinfo.value)


def test_minimize_sign_scalar() -> None:
    from saealib.viz._common import _minimize_sign

    assert _minimize_sign(-1) == 1
    assert _minimize_sign(1) == -1


def test_minimize_sign_none() -> None:
    from saealib.viz._common import _minimize_sign

    assert _minimize_sign(None) == 1.0


def test_minimize_sign_mixed() -> None:
    from saealib.viz._common import _minimize_sign

    out = _minimize_sign(np.array([-1.0, 1.0, -1.0]))
    assert np.array_equal(out, np.array([1.0, -1.0, 1.0]))


def _make_result(history: History | None):
    class _ResultStub:
        def __init__(self, history: History | None) -> None:
            self.history = history

    return _ResultStub(history)


def test_require_history_none() -> None:
    from saealib.viz._history import _require_history

    with pytest.raises(ValidationError) as excinfo:
        _require_history(_make_result(None), "plot_convergence")
    assert "history" in str(excinfo.value).lower()
    assert "history_channels" in str(excinfo.value)


def test_require_channel_missing() -> None:
    from saealib.viz._history import _require_channel

    history = History(("summary",))
    with pytest.raises(ValidationError) as excinfo:
        _require_channel(
            _make_result(history), "decision_candidates", "plot_prescreening"
        )
    msg = str(excinfo.value)
    assert "decision_candidates" in msg
    assert "set_history" in msg
    assert "history_channels" in msg


def test_require_channel_present() -> None:
    from saealib.viz._history import _require_channel

    history = History(("summary",))
    history.append("summary", gen=0, fe=10)
    mapping = _require_channel(_make_result(history), "summary", "plot_convergence")
    assert isinstance(mapping, Mapping)
    assert mapping["gen"][0] == 0
    assert mapping["fe"][0] == 10


def test_require_block_channel_missing() -> None:
    from saealib.viz._history import _require_block

    history = History(("summary",))
    with pytest.raises(ValidationError) as excinfo:
        _require_block(
            history, "decision_candidates", "candidates", "plot_prescreening"
        )
    msg = str(excinfo.value)
    assert "decision_candidates" in msg
    assert "candidates" in msg
    assert "set_history" in msg
    assert "history_channels" in msg


def test_require_block_column_missing_without_dense_numeric_view() -> None:
    from saealib.viz._history import _require_block

    history = History(("population",))
    history.append_block("population", {"f": np.zeros((2, 1))})
    with pytest.raises(ValidationError) as excinfo:
        _require_block(history, "population", "x", "plot_population")
    msg = str(excinfo.value)
    assert "population" in msg
    assert "x" in msg
    assert "DenseNumericView" in msg
    assert "set_history" not in msg
    assert "history_channels" not in msg


def test_require_block_present() -> None:
    from saealib.viz._history import _require_block

    history = History(("decision_candidates",))
    history.append_block(
        "decision_candidates",
        {"candidates": np.zeros((3, 2))},
        selected=1.0,
        acquisition_scores=0.5,
    )
    blocks = _require_block(
        history, "decision_candidates", "candidates", "plot_prescreening"
    )
    assert len(blocks) == 1
    assert blocks[0].shape == (3, 2)
