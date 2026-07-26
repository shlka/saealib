# Integrating External Libraries

`saealib` provides adapters that thinly wrap external machine learning libraries behind its own abstract base classes.

The adapter only translates `saealib`-side data representations such as `Problem`/`Population`/`ctx`; the learning algorithm itself uses the external library's implementation as-is.

Currently, surrogate model adapters (scikit-learn, XGBoost, LightGBM, PyTorch) are implemented.

## Installation

Each adapter can only be used once the corresponding `extra` is installed.

```bash
pip install "saealib[sklearn]"
```

See [Installation](../getting_started/installation.md) for how to install and the full list of extras.

Importing an adapter without the corresponding `extra` installed raises `ImportError`.

## Surrogate adapters

Each adapter implements `saealib`'s `Surrogate` base class, and can be passed to the `surrogate` argument just like the built-in `RBFSurrogate`.

| Class | Required `extra` | Wrapped model |
|--------|--------|--------|
| `SklearnGPRSurrogate` | `sklearn` | Gaussian Process Regressor |
| `SklearnRFRSurrogate` | `sklearn` | Random Forest Regressor |
| `SklearnSVMSurrogate` | `sklearn` | Support Vector Regression |
| `SklearnNNSurrogate` | `sklearn` | Multi-layer Perceptron |
| `SklearnXGBSurrogate` | `xgboost` | XGBoost regression |
| `SklearnLGBMSurrogate` | `lightgbm` | LightGBM regression |
| `TorchSurrogate` | `torch` | Any PyTorch `nn.Module` |

Keyword arguments to the constructor are passed straight through to the corresponding library's model.

```python
import numpy as np
from saealib import minimize, SklearnGPRSurrogate


def expensive_func(x):
    return np.sum(x**2)


DIM = 10

result = minimize(
    expensive_func,
    dim=DIM,
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
    surrogate=SklearnGPRSurrogate(),
    max_fe=300,
    seed=0,
)
```

Passing a `Surrogate` instance to the `surrogate` argument works the same as the `RBFSurrogate` example in "Switching components" in [Single-Objective Optimization](single_objective.md) — internally it is wrapped in a `LocalSurrogateManager`.

See [Surrogate](../components/surrogate.md) for adapters aimed at classification problems (e.g. feasibility classification) and the detailed arguments of each adapter.

## References

- {py:class}`saealib.Surrogate`
- {py:class}`saealib.SklearnGPRSurrogate` / {py:class}`saealib.SklearnRFRSurrogate` / {py:class}`saealib.SklearnSVMSurrogate` / {py:class}`saealib.SklearnNNSurrogate`
- {py:class}`saealib.SklearnXGBSurrogate` / {py:class}`saealib.SklearnLGBMSurrogate`
- {py:class}`saealib.TorchSurrogate`
- {py:func}`saealib.minimize`
