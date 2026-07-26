# AccuracyBasedSurrogateSwitcher

The mechanism for dynamically switching `SurrogateManager` or `OptimizationStrategy` based on surrogate accuracy is made of three independent concerns, combined.

- **Accuracy metric** (`SurrogateAccuracyMetric`): Computes a scalar value from a single fit/predict pair
- **Accuracy evaluation method** (`AccuracyEvaluator`): Decides how to measure the surrogate's generalization performance using the accuracy metric (cross-validation, held-out, etc.)
- **Switching decision** (`AccuracyBasedSurrogateSwitcher`): Takes the evaluation result and decides which component or parameter to use next

The accuracy evaluation method is injected as the `accuracy_evaluator` argument on things like `GlobalSurrogateManager`, and its result is passed to the switching decision via the `surrogate_manager.last_accuracy` property.
This whole mechanism is meant to be used inside an `iterate()` loop.
Because the high-level API (`minimize`/`maximize`) provides no way to swap components mid-run, these Switchers can't be used there.

## Accuracy metric: SurrogateAccuracyMetric

`SurrogateAccuracyMetric` requires two things to be implemented: `name` (a property) and `compute(y_true, y_pred) -> float`.

| Class | Range | Meaning |
|---|---|---|
| `SpearmanCorrelation` | `[-1, 1]`, higher is better | The average per-objective Spearman rank correlation. Based on the view that rank preservation matters for EA {cite}`yu2019spearman` |
| `RMSE` | `[0, ∞)`, lower is better | Root mean squared error |
| `R2Score` | `(-∞, 1]`, higher is better | Coefficient of determination |

If `metrics` is omitted from `AccuracyEvaluator`'s constructor, all three metrics are used by default.

## Accuracy evaluation method: AccuracyEvaluator

`AccuracyEvaluator` requires only one method, `evaluate(surrogate, train_x, train_y) -> SurrogateAccuracy`, to be implemented.

| Class | Parameters | Evaluation method |
|---|---|---|
| `KFoldAccuracyEvaluator` | `metrics=None, n_splits=5` | k-fold cross-validation, retraining a copy of the surrogate per fold |
| `LOOAccuracyEvaluator` | `metrics=None` | Equivalent to `KFoldAccuracyEvaluator` with `n_splits=n_samples` (leave-one-out cross-validation) |
| `HeldOutAccuracyEvaluator` | `held_x, held_y, metrics=None` | Evaluates an already-fitted surrogate on the given held-out data, without retraining |

`HeldOutAccuracyEvaluator` is used for comparison against recent true evaluation points {cite}`hanawa2025switching`.

`SurrogateAccuracy` (`metrics: dict[str, float]`, `n_samples: int`) is a simple container holding the evaluation result, with `get(name, default=nan)` to retrieve a value by metric name.

## Switching decision: AccuracyBasedSurrogateSwitcher

`AccuracyBasedSurrogateSwitcher` requires only one method, `switch(accuracy: SurrogateAccuracy | None) -> T`, to be implemented.
Called inside an `iterate()` loop, paired with `optimizer.set_*()`.

```python
switcher = ManagerSwitcher(primary, fallback)
for ctx in optimizer.iterate():
    optimizer.set_surrogate_manager(
        switcher.switch(optimizer.surrogate_manager.last_accuracy)
    )
```

| Class | Parameters | What it switches |
|---|---|---|
| `ManagerSwitcher` | `primary, fallback, *, metric="spearman", threshold=0.5` | `SurrogateManager` |
| `StrategySwitcher` | `primary, fallback, *, metric="spearman", threshold=0.56` | `OptimizationStrategy` |
| `GenCtrlSwitcher` | `*, gm_max=5, gm_min=0, update_rate=0.5, metric="spearman", initial_error=0.5` | `gen_ctrl` (an integer) |

`ManagerSwitcher`/`StrategySwitcher` are simple binary switches, returning `primary` if the specified metric is at or above the threshold, and `fallback` otherwise.
`StrategySwitcher`'s default threshold `0.56` is based on the PS/IB-GB switching setting from {cite}`hanawa2025switching`.

Rather than a binary switch, `GenCtrlSwitcher` continuously adjusts `gen_ctrl` via exponential smoothing {cite}`repicky2017genctrl`.
Because it holds state — a smoothed error estimate in the public attribute `smoothed_error` — use one instance per `run`.
The `int` returned by `GenCtrlSwitcher.switch()` is meant to be passed directly to [GenerationBasedStrategy](strategies.md)'s `gen_ctrl` argument.

## Implementing a custom Switcher

If you need custom switching logic, subclass `AccuracyBasedSurrogateSwitcher` and implement only `switch()`.

```python
from saealib import AccuracyBasedSurrogateSwitcher


class ThresholdIntSwitcher(AccuracyBasedSurrogateSwitcher):
    """A custom integer-parameter switch: increase n_neighbors if the Spearman correlation is at or above 0.7."""

    def switch(self, accuracy):
        if accuracy is None:
            return 20
        return 50 if accuracy.get("spearman") >= 0.7 else 20
```

## Related components

- [SurrogateManager](surrogate_manager.md): The `accuracy_evaluator` argument and `last_accuracy` property
- [strategies](strategies.md): What `StrategySwitcher`/`GenCtrlSwitcher` switch

## References

- {py:class}`saealib.AccuracyBasedSurrogateSwitcher`
- {py:class}`saealib.ManagerSwitcher`
- {py:class}`saealib.StrategySwitcher`
- {py:class}`saealib.GenCtrlSwitcher`
- {py:class}`saealib.SurrogateAccuracyMetric`
- {py:class}`saealib.SpearmanCorrelation`
- {py:class}`saealib.RMSE`
- {py:class}`saealib.R2Score`
- {py:class}`saealib.AccuracyEvaluator`
- {py:class}`saealib.KFoldAccuracyEvaluator`
- {py:class}`saealib.LOOAccuracyEvaluator`
- {py:class}`saealib.HeldOutAccuracyEvaluator`
- {py:class}`saealib.SurrogateAccuracy`
