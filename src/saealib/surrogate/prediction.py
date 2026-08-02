"""SurrogatePrediction: unified return type for all surrogate model predictions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from saealib.exceptions import ValidationError


@dataclass
class PredictionChannel:
    """
    A single named prediction channel.

    Attributes
    ----------
    value : np.ndarray
        Predicted values. shape: (n_samples, n_output)
    std : np.ndarray or None
        Predicted standard deviations (uncertainty). shape: (n_samples,
        n_output). None if this channel provides no uncertainty estimate.
    covariance : np.ndarray or None
        Predicted covariance, when a surrogate provides joint uncertainty
        across samples or outputs. Shape is implementation-specific.
    samples : np.ndarray or None
        Posterior/Monte-Carlo samples backing this channel, when available.
        Shape is implementation-specific.
    metadata : dict
        Implementation-specific additional information for this channel.
    """

    value: np.ndarray
    std: np.ndarray | None = None
    covariance: np.ndarray | None = None
    samples: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize owned arrays and validate channel-local fields."""
        self.value = _owned_matrix(self.value, "value")
        if self.std is not None:
            self.std = _owned_matrix(self.std, "std", self.value.shape)
        for name in ("covariance", "samples"):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value)
                if (
                    arr.dtype == object
                    or arr.ndim == 0
                    or arr.shape[0] != self.value.shape[0]
                ):
                    raise ValidationError(
                        f"{name} must have the same leading dimension as value"
                    )
                setattr(
                    self, name, np.array(arr, dtype=np.float64, order="C", copy=True)
                )


@dataclass
class SurrogatePrediction:
    """
    Unified return type for surrogate model predictions.

    ``channels`` is the canonical representation: one named
    :class:`PredictionChannel` per prediction output (e.g. ``"objective"``,
    ``"win_rate"``). ``value``/``std`` are convenience properties delegating
    to the ``"objective"`` channel and raise ``KeyError`` when it is absent.

    Attributes
    ----------
    channels : dict[str, PredictionChannel]
        Named prediction channels. Channel names are unique, non-empty
        strings.
    x : np.ndarray or None
        The query points passed to ``predict()``. shape: (n_samples,
        n_features). None unless the surrogate populates it. Needed by
        acquisition functions that have no other channel to the points
        being scored (e.g. :class:`~saealib.acquisition.mean.CORSDistance`).
    label : np.ndarray or None
        Predicted class labels. shape: (n_samples,).
        None unless the surrogate is a classification model.
    metadata : dict
        Implementation-specific additional information
        (e.g., SHAP values, gradient estimates).

    """

    channels: dict[str, PredictionChannel]
    x: np.ndarray | None = None
    label: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    # Values that are conventionally used should be implemented
    # as attributes rather than metadata.

    def __post_init__(self) -> None:
        """Validate channel names, row alignment, and optional arrays."""
        if not isinstance(self.channels, dict):
            raise ValidationError("SurrogatePrediction.channels must be a dict")
        for name, channel in self.channels.items():
            if not isinstance(name, str) or not name:
                raise ValidationError(
                    "prediction channel names must be non-empty strings"
                )
            if not isinstance(channel, PredictionChannel):
                raise ValidationError(f"channel {name!r} is not a PredictionChannel")
        n = next(iter(self.channels.values())).value.shape[0] if self.channels else 0
        if any(channel.value.shape[0] != n for channel in self.channels.values()):
            raise ValidationError(
                "all prediction channels must have the same row count"
            )
        if self.x is not None:
            x = np.asarray(self.x)
            if x.ndim != 2 or x.dtype == object or x.shape[0] != n:
                raise ValidationError("prediction x must have shape (n, dim)")
            self.x = np.array(x, dtype=np.float64, order="C", copy=True)
        if self.label is not None:
            label = np.asarray(self.label)
            if label.shape != (n,) or label.dtype == object:
                raise ValidationError("prediction label must have shape (n,)")
            self.label = np.array(label, copy=True)

    @property
    def value(self) -> np.ndarray:
        """Return the ``"objective"`` channel's value. Raises KeyError if absent."""
        return self.channels["objective"].value

    @property
    def std(self) -> np.ndarray | None:
        """Return the ``"objective"`` channel's std. Raises KeyError if absent."""
        return self.channels["objective"].std

    @property
    def has_uncertainty(self) -> bool:
        """Return True if the ``"objective"`` channel provides uncertainty."""
        channel = self.channels.get("objective")
        return channel is not None and channel.std is not None

    @property
    def has_label(self) -> bool:
        """Return True if classification labels are available."""
        return self.label is not None

    def select_channel(
        self, name: str, *, as_objective: bool = True
    ) -> SurrogatePrediction:
        """
        Project a single named channel into a standalone ``SurrogatePrediction``.

        Parameters
        ----------
        name : str
            Channel name to select. Raises ``KeyError`` if not present in
            ``self.channels``.
        as_objective : bool, optional
            If True (default), the result's ``channels`` is
            ``{"objective": self.channels[name]}`` so ``.value``/``.std``
            resolve normally. If False, the result's ``channels`` is
            ``{name: self.channels[name]}``, preserving the original name
            for callers that want to project without claiming the channel
            is the objective.

        Returns
        -------
        SurrogatePrediction
            A new prediction carrying forward ``self.x`` and
            ``self.metadata`` unchanged. The selected ``PredictionChannel``
            (with its own ``covariance``/``samples``) is reused, not copied.
        """
        channel = self.channels[name]
        key = "objective" if as_objective else name
        return SurrogatePrediction(
            channels={key: channel}, x=self.x, metadata=self.metadata
        )

    @classmethod
    def objective(
        cls,
        value: np.ndarray,
        std: np.ndarray | None = None,
        x: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SurrogatePrediction:
        """
        Build a single-``"objective"``-channel ``SurrogatePrediction``.

        Equivalent to constructing
        ``channels={"objective": PredictionChannel(value, std)}`` by hand.
        """
        return cls(
            channels={"objective": PredictionChannel(value=value, std=std)},
            x=x,
            metadata=metadata if metadata is not None else {},
        )


def _owned_matrix(
    value: np.ndarray, name: str, expected: tuple[int, int | None] | None = None
) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.dtype == object:
        raise ValidationError(f"prediction {name} must have shape (n, n_output)")
    if expected is not None and (
        arr.shape[0] != expected[0]
        or (expected[1] is not None and arr.shape != expected)
    ):
        raise ValidationError(
            f"prediction {name} shape {arr.shape} does not match {expected}"
        )
    return np.array(arr, dtype=np.float64, order="C", copy=True)
