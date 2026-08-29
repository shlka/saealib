"""Tests for shared :mod:`saealib.viz` helpers."""

from __future__ import annotations

import subprocess
import sys

import pytest

from saealib.exceptions import ValidationError


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
    from saealib.viz._matplotlib import _require_matplotlib

    class _MatplotlibBlocker:
        def find_spec(self, name: str, path: object, target: object = None):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    blocker = _MatplotlibBlocker()
    saved = sys.modules.pop("matplotlib", None)
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(ImportError) as excinfo:
            _require_matplotlib()
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
