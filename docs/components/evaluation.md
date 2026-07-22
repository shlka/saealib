# Evaluator

`OptimizationStrategy`や`Initializer`は、設計変数の候補群を目的関数値、生の制約値、制約違反へ変換する処理を、`Evaluator`という差し替え可能な実行バックエンドに委ねています。
評価を逐次実行するか並列実行するかは、パイプライン側のコードを変えずに`Evaluator`だけを差し替えれば切り替えられます。

## Evaluatorの役割

`Evaluator`が実装を要求するメソッドは`evaluate_batch(x, problem) -> EvaluationResult`の1つだけです。
`x`には評価対象の設計変数群が`shape (n, dim)`で渡され、[Problem](problem.md)の目的関数と制約関数を使って`EvaluationResult`を構築して返します。

`EvaluationResult`は3つの配列を持つデータクラスです。

- **`f`**：目的関数値。`shape (n, n_obj)`
- **`g`**：生の制約値。`shape (n, n_constraints)`。問題が制約を持たない場合は`(n, 0)`
- **`cv`**：候補ごとの集約された制約違反量。`shape (n,)`。問題が制約を持たない場合は全て`0`

## 組み込みEvaluator

| クラス | パラメータ | 特徴 |
|---|---|---|
| `SerialEvaluator` | なし | 候補を1件ずつ逐次評価する。既定値 |
| `JoblibEvaluator` | `n_jobs=-1, backend="loky", **joblib_kwargs` | `joblib.Parallel`経由で候補を並列評価する |

`JoblibEvaluator`を使うには`parallel` extra（`pip install saealib[parallel]`）が必要で、未インストールの場合は構築時に`ImportError`になります。
`backend`は`"loky"`（既定、cloudpickleでシリアライズするプロセスプール）のほか、`"dask"`/`"ray"`のようなサードパーティバックエンドにも1パラメータの変更で切り替えられます（対応するパッケージとクラスタが別途必要）。
Islandモデルのように複数の`JoblibEvaluator`を同時に使う構成では、CPUコアの多重予約が起こりえます。
各island側の`n_jobs`を`1`に絞り、island間の並列度で全体の同時実行数を制御するか、`joblib.parallel_backend`をコンテキストマネージャとして使って内側のワーカー数を制限します。
非同期な評価（候補ごとに独立したタイミングで結果を受け取る方式）は、現状`Evaluator`のスコープ外です。

`Optimizer.set_evaluator(evaluator)`で差し替えます。

## 独自Evaluatorの実装方法

独自の実行バックエンドが必要な場合は、`Evaluator`を継承して`evaluate_batch()`だけを実装すればよいです。
`SerialEvaluator`の実装がそのままテンプレートになります。

```python
import numpy as np
from saealib import EvaluationResult, Evaluator


class ReversedOrderEvaluator(Evaluator):
    """候補を末尾から順に評価する Evaluator。"""

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

`problem.evaluate_constraints(xi)`を呼んでから`problem.evaluate(xi, g_i)`を呼ぶ、という順序を保つ必要があります。
[ConstraintHandler](constraints.md)の`augment_objective`が制約値を使って目的関数値を補正するため、この順序を逆にすると補正が正しく反映されません。

## 関連コンポーネント

- [Problem](problem.md) — `evaluate_batch`が評価する目的関数と制約関数の定義元
- [ConstraintHandler](constraints.md) — `problem.evaluate`の内部で制約違反の集約と目的関数の補正を行う
- [Initializer](initialization.md) — 初期集団の評価に`Evaluator`を使う
- [strategies](strategies.md) — 各世代の候補評価に`Evaluator`を使う

## 参照

- {py:class}`saealib.Evaluator`
- {py:class}`saealib.SerialEvaluator`
- {py:class}`saealib.JoblibEvaluator`
- {py:class}`saealib.EvaluationResult`
