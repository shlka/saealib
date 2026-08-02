# Evaluator

`OptimizationStrategy` and `Initializer` delegate converting a batch of design-variable candidates into objective values, raw constraint values, and constraint violation to `Evaluator`, a swappable execution backend.
Whether evaluation runs sequentially or in parallel can be switched by swapping out only `Evaluator`, without changing the pipeline-side code.

## Evaluator's role

`Evaluator` requires only one method, `evaluate_batch(x, problem) -> EvaluationResult`, to be implemented.
`x` receives the set of design variables to evaluate with shape `(n, dim)`, and it builds and returns an `EvaluationResult` using [Problem](problem.md)'s objective and constraint functions.

`EvaluationResult` is a dataclass holding three arrays.

- **`f`**: Objective values. Shape `(n, n_obj)`
- **`g`**: Raw constraint values. Shape `(n, n_constraints)`. `(n, 0)` if the problem has no constraints
- **`cv`**: Aggregated constraint violation per candidate. Shape `(n,)`. All `0` if the problem has no constraints

## Built-in Evaluators

| Class | Parameters | Characteristics |
|---|---|---|
| `SerialEvaluator` | None | Evaluates candidates one at a time, sequentially. The default |
| `JoblibEvaluator` | `n_jobs=-1, backend="loky", **joblib_kwargs` | Evaluates candidates in parallel via `joblib.Parallel` |

Using `JoblibEvaluator` requires the `parallel` extra (`pip install saealib[parallel]`); if not installed, it raises `ImportError` at construction time.
Besides `backend`'s default `"loky"` (a process pool serializing with cloudpickle), you can switch to third-party backends like `"dask"`/`"ray"` with a single parameter change (the corresponding package and cluster are required separately).
In configurations using multiple `JoblibEvaluator`s at once, such as an island model, CPU cores can end up over-reserved.
Either limit each island's `n_jobs` to `1` and control overall concurrency via the parallelism across islands, or use `joblib.parallel_backend` as a context manager to limit the number of inner workers.
For independent completion times, use `AsyncEvaluator` with
`SteadyStateStrategy`, `AsyncScheduler`, and
`Optimizer.set_async_scheduler()`. Each steady-state candidate can occupy its
own pending request; the scheduler owns those requests and serializes
population and archive updates while `collect(wait=False)` remains
non-blocking.

Swap it via `Optimizer.set_evaluator(evaluator)`.

`EvaluationRequest.metadata` is consumed by evaluators that expose explicit
execution policies. `FidelityEvaluator` reads `fidelity`, while
`RepeatedEvaluationRunner` submits separate requests with stable candidate IDs
and stores every observation in an append history.

## Implementing a custom Evaluator

If you need a custom execution backend, subclass `Evaluator` and implement only `evaluate_batch()`.
`SerialEvaluator`'s implementation serves directly as a template.

```python
import numpy as np
from saealib import EvaluationResult, Evaluator


class ReversedOrderEvaluator(Evaluator):
    """An Evaluator that evaluates candidates in reverse order, from the end."""

    def evaluate_batch(self, x, problem):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(x)
        f = np.empty((n, problem.n_obj), dtype=float)
        g = np.empty((n, problem.n_constraints), dtype=float)
        cv = np.zeros(n, dtype=float)
        for i in reversed(range(n)):
            g_i, cv_i = problem.evaluate_constraints(x[i])
            f[i] = problem.evaluate(x[i], g_i)
            g[i] = g_i
            cv[i] = cv_i
        return EvaluationResult(f=f, g=g, cv=cv)
```

You need to preserve the order of calling `problem.evaluate_constraints(xi)` before `problem.evaluate(xi, g_i)`.
Because [ConstraintHandler](constraints.md)'s `augment_objective` corrects the objective value using the constraint values, reversing this order means the correction isn't applied correctly.

## Related components

- [Problem](problem.md): Where the objective and constraint functions evaluated by `evaluate_batch` are defined
- [ConstraintHandler](constraints.md): Aggregates constraint violation and corrects the objective value inside `problem.evaluate`
- [Initializer](initialization.md): Uses `Evaluator` to evaluate the initial population
- [strategies](strategies.md): Uses `Evaluator` to evaluate candidates each generation

## References

- {py:class}`saealib.Evaluator`
- {py:class}`saealib.SerialEvaluator`
- {py:class}`saealib.JoblibEvaluator`
- {py:class}`saealib.AsyncEvaluator`
- {py:class}`saealib.AsyncScheduler`
- {py:class}`saealib.EvaluationResult`
