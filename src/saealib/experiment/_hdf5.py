"""HDF5 persistence for one experiment trial."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from saealib.context import OptimizationState


def _require_h5py() -> ModuleType:
    """Return h5py or raise an actionable import error."""
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ImportError(
            "HDF5 results require h5py. Install it with `pip install saealib[hdf5]`."
        ) from exc
    return h5py


def write_trial(
    path: str | Path,
    state: OptimizationState,
    *,
    seed: int | None,
    wall_time: float,
    labels: Mapping[str, str],
) -> None:
    """Write archive, history, and trial metadata to an HDF5 file.

    ``labels`` is stored as JSON because HDF5 attributes have no portable
    mapping type and JSON keeps the index readable by other HDF5 clients.
    """
    h5py = _require_h5py()
    destination = Path(path)
    from saealib.experiment._trial import (
        _archive_columns,
        _enabled_channels,
        trial_metadata,
    )

    history = state.history
    meta = trial_metadata(state, seed=seed, wall_time=wall_time, labels=labels)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as handle:
        archive = handle.create_group("archive")
        for name, array in _archive_columns(state).items():
            archive.create_dataset(name, data=_hdf5_data(array))
        history_group = handle.create_group("history")
        if history is not None:
            for channel_name in _enabled_channels(history):
                channel_group = history_group.create_group(channel_name)
                for name, array in history.channel(channel_name).items():
                    channel_group.create_dataset(name, data=_hdf5_data(array))
                records = list(history.records(channel_name))
                block_names = {
                    name
                    for record in records
                    for name, value in record.items()
                    if isinstance(value, np.ndarray) and value.ndim == 2
                }
                for name in sorted(block_names):
                    block_group = channel_group.create_group(name)
                    blocks = history.blocks(channel_name, name)
                    width = max(1, len(str(len(blocks) - 1)))
                    for generation, block in enumerate(blocks):
                        block_group.create_dataset(
                            f"gen{generation:0{width}d}", data=_hdf5_data(block)
                        )
        encoded = sorted(
            name
            for name, value in meta.items()
            if isinstance(value, (dict, list, tuple)) or value is None
        )
        for name, value in meta.items():
            handle.attrs[name] = json.dumps(value) if name in encoded else value
        handle.attrs["_json_attrs"] = json.dumps(encoded)


def read_hdf5_trial(path: str | Path) -> dict[str, Any]:
    """Read an HDF5 trial into nested mappings of NumPy arrays."""
    h5py = _require_h5py()
    result: dict[str, Any] = {"archive": {}, "history": {}, "meta": {}}
    with h5py.File(path, "r") as handle:
        for name, dataset in handle["archive"].items():
            result["archive"][name] = np.asarray(dataset)
        for channel_name, channel_group in handle["history"].items():
            result["history"][channel_name] = {}
            for name, item in channel_group.items():
                if isinstance(item, h5py.Group):
                    result["history"][channel_name][name] = tuple(
                        np.asarray(item[generation]) for generation in sorted(item)
                    )
                else:
                    result["history"][channel_name][name] = np.asarray(item)
        encoded = set(json.loads(handle.attrs.get("_json_attrs", "[]")))
        result["meta"] = {
            name: json.loads(value) if name in encoded else value
            for name, value in handle.attrs.items()
            if name != "_json_attrs"
        }
    return result


def _hdf5_data(array: np.ndarray) -> np.ndarray:
    """Convert arrays to dtypes accepted consistently by h5py."""
    value = np.asarray(array)
    if value.dtype.kind == "U":
        return value.astype(h5py_string_dtype())
    return value


def h5py_string_dtype() -> Any:
    """Return the variable-length UTF-8 dtype without importing h5py early."""
    return _require_h5py().string_dtype(encoding="utf-8")
