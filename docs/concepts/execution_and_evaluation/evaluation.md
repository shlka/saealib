---
primary_layer: layer4
---

# Evaluator

`OptimizationStrategy` は `EvaluationPlan` を作り、Evaluatorまたは非同期Schedulerへ評価Requestを渡します。
Requestは候補IDとGenomeBatchを持ち、EvaluatorはObservationとして目的値、制約値、制約違反を返します。
評価が同期か非同期かは、Pipeline側の候補生成契約を変えずにEvaluatorとRuntime providerで切り替えられます。

## Evaluatorの役割

`Evaluator` の実装境界は `evaluate_batch(payload, problem) -> EvaluationResult` です。
`payload` は通常 `GenomeBatch` であり、旧来のベクトル経路では `numpy.ndarray` の `(n, dim)` ビューを受け取れます。
Genomeを評価関数の入力へ変換する必要がある場合は、Problemの `evaluation_adapter` を使います。

`EvaluationRequest` は、Request ID、候補ID、GenomeBatch、メタデータを保持します。
評価結果は行の順序と候補IDの対応を維持し、目的値、制約値、制約違反に加えて評価コストや出力を持てます。

`EvaluationResult` は、評価結果の配列と候補IDを保持するデータクラスです。

- **`f`**：目的値。形状は `(n, n_obj)`。
- **`g`**：生の制約値。形状は `(n, n_constraints)`。制約がない場合は `(n, 0)`。
- **`cv`**：候補ごとの集約制約違反。形状は `(n,)`。制約がない場合はすべて `0`。
- **`candidate_ids`**：結果の各行に対応する候補ID。

## Built-in Evaluators

| Class | Parameters | Characteristics |
|---|---|---|
| `SerialEvaluator` | None | Evaluates candidates one at a time, sequentially. The default |
| `JoblibEvaluator` | `n_jobs=-1, backend="loky", **joblib_kwargs` | Evaluates candidates in parallel via `joblib.Parallel` |
| `ThreadPoolEvaluator` | Executor options | Uses a thread pool for synchronous batch evaluation |
| `AsyncEvaluator` | Adapter and scheduler options | Submits requests and exposes nonblocking lifecycle operations |

Using `JoblibEvaluator` requires the `parallel` extra (`pip install saealib[parallel]`); if not installed, it raises `ImportError` at construction time.
Besides `backend`'s default `"loky"` (a process pool serializing with cloudpickle), you can switch to third-party backends like `"dask"`/`"ray"` with a single parameter change (the corresponding package and cluster are required separately).
In configurations using multiple `JoblibEvaluator`s at once, such as an island model, CPU cores can end up over-reserved.
Either limit each island's `n_jobs` to `1` and control overall concurrency via the parallelism across islands, or use `joblib.parallel_backend` as a context manager to limit the number of inner workers.
独立した完了時刻を扱う場合は、`AsyncEvaluator`、`SteadyStateStrategy`、`AsyncEvaluationScheduler`、
`Optimizer.set_async_evaluation_scheduler()` を組み合わせます。
各候補は保留中のRequestとして管理され、Schedulerが容量、予算、Requestの状態を管理します。
`collect(wait=False)` は完了を待たずに進捗を返し、PopulationとArchiveの更新はRuntimeが順序を管理します。

同期Evaluatorを差し替える場合は、`Optimizer.set_evaluator(evaluator)` を使います。

`EvaluationRequest.metadata` is consumed by evaluators that expose explicit
execution planners. `RepeatedEvaluation` returns one `EvaluationPlan` request
per replicate with stable candidate IDs. The scheduler owns capacity, budget,
and request lifecycle.

## カスタムEvaluatorの実装

独自の評価バックエンドを追加するときは、`Evaluator` を継承して `evaluate_batch()` を実装します。
次の例は、ベクトルGenomeを受け取る互換性用Evaluatorです。
非ベクトルGenomeでは、評価関数へ渡すpayloadを `EvaluationAdapter` で定義します。

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

`problem.evaluate_constraints(xi)` を `problem.evaluate(xi, g_i)` より先に呼び出す順序を維持します。
[ConstraintHandler](../problem_and_ranking/constraints.md) は制約値を使って目的値を補正するため、この順序を逆にすると補正が正しく適用されません。

## Related components

- [Problem](../problem_and_ranking/problem.md): Where the objective and constraint functions evaluated by `evaluate_batch` are defined
- [ConstraintHandler](../problem_and_ranking/constraints.md): Aggregates constraint violation and corrects the objective value inside `problem.evaluate`
- [Initializer](initialization.md): Uses `Evaluator` to evaluate the initial population
- [strategies](strategies.md): Uses `Evaluator` to evaluate candidates each generation
- [Feedback](../observation_and_state/feedback.md)：評価結果をAlgorithmへ渡すFeedbackの構成

## References

- {py:class}`saealib.Evaluator`
- {py:class}`saealib.SerialEvaluator`
- {py:class}`saealib.JoblibEvaluator`
- {py:class}`saealib.AsyncEvaluator`
- {py:class}`saealib.AsyncEvaluationScheduler`
- {py:class}`saealib.EvaluationResult`
