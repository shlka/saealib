"""Read-only History access helpers for :mod:`saealib.viz`.

Every helper reads History only through the public :meth:`History.channel` and
:meth:`History.blocks` views. Missing data raises
:class:`saealib.exceptions.ValidationError` with both what is missing and how to
enable it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

import numpy as np

from saealib.exceptions import ValidationError

if TYPE_CHECKING:
    from saealib.execution.history import History

    class _HasHistory(Protocol):
        history: History | None


def _require_history(result: _HasHistory, function: str) -> History:
    """Return the recorded :class:`History` or raise a clear error.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result whose ``.history`` attribute is inspected.
    function : str
        Name of the calling plot function, used in the error message.

    Returns
    -------
    saealib.execution.history.History
        The recorded history.

    Raises
    ------
    ValidationError
        If ``result.history`` is ``None``.
    """
    history = result.history
    if history is None:
        raise ValidationError(
            f"{function} requires execution history, but the result has none. "
            "Record it by passing history_channels=[...] to minimize() or by "
            "enabling it on the Optimizer with set_history([...])."
        )
    return history


def _require_channel(
    result: _HasHistory, channel: str, function: str
) -> Mapping[str, np.ndarray]:
    """Return a channel's scalar columns as a read-only mapping.

    Parameters
    ----------
    result : saealib.api.Result
        Optimization result providing the history.
    channel : str
        Required history channel name (e.g. ``"decision_candidates"``).
    function : str
        Name of the calling plot function, used in the error message.

    Returns
    -------
    Mapping[str, numpy.ndarray]
        Read-only per-column row views of the channel.

    Raises
    ------
    ValidationError
        If the channel is not enabled.
    """
    history = _require_history(result, function)
    if not history.is_enabled(channel):
        raise ValidationError(
            f'{function} requires the "{channel}" history channel. '
            "Enable it with Optimizer.set_history([...]) or "
            "minimize(..., history_channels=[...])."
        )
    return history.channel(channel)


def _require_block(
    history: History, channel: str, column: str, function: str
) -> tuple[np.ndarray, ...]:
    """Return a channel's block column as read-only per-record views.

    Parameters
    ----------
    history : saealib.execution.history.History
        Recorded history. Use :func:`_require_history` to obtain it from a
        result.
    channel : str
        History channel name.
    column : str
        Block column name within the channel (e.g. ``"candidates"``).
    function : str
        Name of the calling plot function, used in the error message.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        One read-only two-dimensional view per recorded generation.

    Raises
    ------
    ValidationError
        If the channel is not enabled or the block column is unavailable.
    """
    if not history.is_enabled(channel):
        raise ValidationError(
            f'{function} requires the "{channel}" history channel. '
            "Enable it with Optimizer.set_history([...]) or "
            "minimize(..., history_channels=[...])."
        )
    try:
        return history.blocks(channel, column)
    except ValidationError as exc:
        raise ValidationError(
            f'{function} requires the "{column}" block of the '
            f'"{channel}" history channel. '
            "This block is recorded only when the search space provides "
            "the DenseNumericView service."
        ) from exc
