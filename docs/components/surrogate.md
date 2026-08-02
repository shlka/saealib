# Surrogate

`saealib` restricts the responsibility of the predictive model approximating the objective function to `Surrogate`, a swappable component.
`Surrogate` knows only fit/predict — it knows nothing about how a prediction is converted into a score ([AcquisitionFunction](acquisition_functions.md)), or where training data comes from ([TrainingSet](training_set.md), via [SurrogateManager](surrogate_manager.md)).

## Surrogate's role

`Surrogate` requires two methods to be implemented.

**`fit(train_x, train_y) -> None`**: Trains the model on inputs of shape `(n_samples, n_features)` and outputs of shape `(n_samples, n_obj)` (or `(n_samples,)` for single-objective).

**`predict(test_x) -> SurrogatePrediction`**: Returns predictions for inputs of shape `(n_samples, n_features)`.

The class attribute `provides_uncertainty: bool = False` indicates whether predictions carry uncertainty (standard deviation).
The default is `False`; only Gaussian Process implementations override it to `True`.

There are two marker base classes between `Surrogate` and `predict()`.

**`RegressionSurrogate`**: A marker for regression surrogates where `train_y` is a real-valued objective function output.

**`ComparisonSurrogate`**: For comparison surrogates where `train_y` is a `{0, 1}`-valued binary comparison label.
`predict_proba(test_x) -> SurrogatePrediction` (values are win probabilities in `[0, 1]`) is the primary interface, with `predict()` delegating to `predict_proba()` by default.

## SurrogatePrediction

`predict()`'s return value is `SurrogatePrediction`, a unified dataclass.

| Field | Content |
|---|---|
| `value` | The predicted value. Shape `(n_samples, n_obj)` |
| `std` | Uncertainty (standard deviation). `None` for surrogates that don't provide it |
| `label` | Class labels, only present for classification models |
| `metadata` | A `dict` holding implementation-specific auxiliary information, such as SHAP values |

`value` and `std` are convenience properties for the objective channel.
When a surrogate's prediction represents a quantity that isn't the objective value (such as a novelty score), this mechanism prevents contaminating things like pbest.
[SurrogateManager](surrogate_manager.md)'s `ArchiveBasedManager` family uses this technique.

## Built-in Surrogates

**`RBFSurrogate(kernel, dim)`**: A surrogate using RBF interpolation {cite}`gutmann2001rbf,regis2005cors` (the origin of RBF interpolation itself is Hardy, 1971).
`gaussian_kernel(x1, x2, sigma=2.0)` is the default kernel used, but the `kernel` argument is designed as a public API accepting a kernel function, so users can inject any kernel.
`predict()` explicitly returns `std=None` (RBF interpolation doesn't provide uncertainty).

**`PerObjectiveSurrogate(surrogates)`**: A `RegressionSurrogate` subclass, a composite class that assigns a different surrogate per objective.
Raises `ValueError` at `fit` time if `train_y`'s column count doesn't match `len(surrogates)`.
`provides_uncertainty` is a composite judgment, returning `True` only if every constituent surrogate is `True`.

### External library adapters

Regression surrogates via a scikit-learn-compatible API carry a `Sklearn` prefix.

| Class | Model |
|---|---|
| `SklearnGPRSurrogate` | Gaussian Process {cite}`sacks1989dace,rasmussen2006gpml`. The sole implementation with `provides_uncertainty=True` |
| `SklearnRFRSurrogate` | Random Forest regression |
| `SklearnSVMSurrogate` | SVM |
| `SklearnNNSurrogate` | MLP |
| `SklearnXGBSurrogate` | XGBoost (the `xgboost` extra) |
| `SklearnLGBMSurrogate` | LightGBM (the `lightgbm` extra) |
| `TorchSurrogate` | PyTorch-based models (the `torch` extra) |

`SklearnGPRSurrogate` computes standard deviation from the GP kernel via `return_std=True`, and returns `provides_uncertainty=True`.

The following classes are classification surrogates aimed at feasibility prediction.

| Class | Model |
|---|---|
| `SklearnClassificationSurrogate` | Classification models compatible with scikit-learn in general |
| `SklearnRFCClassificationSurrogate` | Random Forest classification |
| `SklearnSVCClassificationSurrogate` | SVM classification |

See [TrainingSet](training_set.md)'s `FeasibilityClassificationSet` for how training data is extracted for these classification surrogates.
For pairwise comparison, use the dedicated `ComparisonSurrogate`-family implementations rather than these classification surrogates, paired with [SurrogateManager](surrogate_manager.md)'s `PairwiseSurrogateManager` and `PairwiseComparisonSet`.

See [Installation](../getting_started/installation.md) for how to install each extra.

```{note}
Adapters for BoTorch/SMT surrogates, beyond scikit-learn/XGBoost/LightGBM/PyTorch, are not currently implemented in `saealib`.
`pyproject.toml` has no corresponding extra either.
pymoo has adapters, but at the [Problem](problem.md)/[Crossover](crossover.md)/[Mutation](mutation.md)/[Algorithm](algorithm.md) level, not as a `Surrogate` — pymoo doesn't provide surrogate models.
```

## Extension hooks

If you just want to add post-fit processing, you can add it to an existing `Surrogate` instance with `with_post_fit(fn)` instead of creating a new subclass.
`with_post_fit` doesn't modify the original instance — it returns a copy with `fn` added.

```python
from saealib import RBFSurrogate, gaussian_kernel


def log_fit(train_x, train_y, ctx=None):
    print(f"fit on {len(train_x)} samples")


base = RBFSurrogate(gaussian_kernel, dim=2)
logged = base.with_post_fit(log_fit)
```

`fn`'s signature is `fn(train_x, train_y, ctx) -> None`.

## Implementing a custom Surrogate

If you need a custom predictive model, subclass `Surrogate` and implement `fit()`/`predict()`.
Choose `RegressionSurrogate` for regression, or `ComparisonSurrogate` (implementing `predict_proba()`) for comparison, as the class to subclass.

The following example is a simple regression surrogate that returns the nearest training point's objective value directly as the prediction.

```python
import numpy as np
from saealib import RegressionSurrogate, SurrogatePrediction


class NearestNeighborSurrogate(RegressionSurrogate):
    """A simple surrogate that returns the nearest point's objective value directly as the prediction."""

    def fit(self, train_x, train_y):
        self.train_x = np.asarray(train_x, dtype=float)
        self.train_y = np.asarray(train_y, dtype=float)

    def predict(self, test_x):
        test_x = np.atleast_2d(test_x)
        dists = np.linalg.norm(self.train_x[None, :, :] - test_x[:, None, :], axis=2)
        nearest = dists.argmin(axis=1)
        value = self.train_y[nearest]
        return SurrogatePrediction(value=value)
```

## Uncertainty support table

To use an uncertainty-based [AcquisitionFunction](acquisition_functions.md), `Surrogate` needs to return `std`.

| Class | `provides_uncertainty` |
|---|---|
| `SklearnGPRSurrogate` | `True` |
| `RBFSurrogate` / `SklearnRFRSurrogate` / `SklearnSVMSurrogate` / `SklearnNNSurrogate` / `SklearnXGBSurrogate` / `SklearnLGBMSurrogate` / `TorchSurrogate` | `False` |
| `PerObjectiveSurrogate` | `True` only if every constituent surrogate is `True` |

`Optimizer.validate()` detects and warns about a mismatch between `AcquisitionFunction`'s `requires_uncertainty` and `Surrogate`'s `provides_uncertainty`.

## Related components

- [SurrogateManager](surrogate_manager.md): Coordinates `Surrogate`'s fit/predict and combines it with scoring
- [TrainingSet](training_set.md): How training data passed to `Surrogate` is extracted
- [AcquisitionFunction](acquisition_functions.md): Converts `predict()`'s result into a score
- [Surrogate accuracy evaluation and dynamic switching](surrogate_switching.md): Evaluating a surrogate's generalization performance
- [Installation](../getting_started/installation.md): How to install each extra

## References

- {py:class}`saealib.Surrogate`
- {py:class}`saealib.RegressionSurrogate`
- {py:class}`saealib.ComparisonSurrogate`
- {py:class}`saealib.SurrogatePrediction`
- {py:class}`saealib.RBFSurrogate`
- {py:func}`saealib.gaussian_kernel`
- {py:class}`saealib.PerObjectiveSurrogate`
- {py:class}`saealib.SklearnGPRSurrogate`
- {py:class}`saealib.SklearnRFRSurrogate`
- {py:class}`saealib.SklearnSVMSurrogate`
- {py:class}`saealib.SklearnNNSurrogate`
- {py:class}`saealib.SklearnXGBSurrogate`
- {py:class}`saealib.SklearnLGBMSurrogate`
- {py:class}`saealib.TorchSurrogate`
- {py:class}`saealib.SklearnClassificationSurrogate`
- {py:class}`saealib.SklearnRFCClassificationSurrogate`
- {py:class}`saealib.SklearnSVCClassificationSurrogate`
