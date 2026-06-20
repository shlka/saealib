"""PyTorch surrogate adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

from saealib.surrogate.base import Surrogate
from saealib.surrogate.prediction import SurrogatePrediction


class TorchSurrogate(Surrogate):
    """
    Surrogate adapter for PyTorch ``nn.Module`` models.

    Wraps any ``torch.nn.Module`` that maps ``(n_samples, n_features)`` to
    ``(n_samples, n_obj)``. The model is trained from scratch on each ``fit``
    call by resetting to the weights captured at construction time, so this
    adapter is safe to use with ``LocalSurrogateManager`` (which re-fits per
    candidate).

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model. Its output shape must be ``(n_samples, n_obj)``.
    optimizer_cls : type or None
        Optimizer class (e.g. ``torch.optim.Adam``).
        Defaults to ``torch.optim.Adam``.
    optimizer_kwargs : dict or None
        Keyword arguments forwarded to the optimizer constructor.
        Defaults to ``{}``.
    loss_fn : callable or None
        Loss function. Defaults to ``torch.nn.MSELoss()``.
    epochs : int
        Number of training epochs per ``fit`` call. Default: 100.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cls: type | None = None,
        optimizer_kwargs: dict | None = None,
        loss_fn: Callable[..., Any] | None = None,
        epochs: int = 100,
    ) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for TorchSurrogate. "
                "Install it with: pip install saealib[torch]"
            ) from e
        self.model = model
        self._initial_state = {k: v.clone() for k, v in model.state_dict().items()}
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.n_obj: int | None = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fit the surrogate model.

        Resets model weights to their initial values before training.

        Parameters
        ----------
        train_x : np.ndarray
            Training input data. shape: (n_samples, n_features)
        train_y : np.ndarray
            Training output data. shape: (n_samples, n_obj) or (n_samples,).
        """
        import torch

        self.model.load_state_dict(self._initial_state)
        self.model.train()

        arr = np.asarray(train_y, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.n_obj = arr.shape[1]

        device = next(self.model.parameters()).device
        x_tensor = torch.tensor(train_x, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(arr, dtype=torch.float32).to(device)

        optimizer_cls = self.optimizer_cls or torch.optim.Adam
        optimizer = optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        loss_fn = self.loss_fn or torch.nn.MSELoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(x_tensor)  # type: ignore[operator]  # torch Module.__call__ not typed as callable in stubs
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, test_x: np.ndarray) -> SurrogatePrediction:
        """
        Predict using the surrogate model.

        Parameters
        ----------
        test_x : np.ndarray
            Input data. shape: (n_samples, n_features) or (n_features,)

        Returns
        -------
        SurrogatePrediction
            prediction.value shape: (n_samples, n_obj)
            prediction.std is None
        """
        import torch

        test = np.asarray(test_x)
        if test.ndim == 1:
            test = test.reshape(1, -1)

        self.model.eval()
        with torch.no_grad():
            device = next(self.model.parameters()).device
            x_tensor = torch.tensor(test, dtype=torch.float32).to(device)
            pred = self.model(x_tensor).detach().cpu().numpy()  # type: ignore[operator]  # torch Module.__call__ not typed as callable in stubs

        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return SurrogatePrediction(value=pred)
