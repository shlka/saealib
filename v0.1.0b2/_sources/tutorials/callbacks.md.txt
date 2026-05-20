# コールバックによる進捗の監視

このチュートリアルでは，`CallbackManager` を使って最適化の進捗をログ出力・記録する方法を説明します．

## デフォルトのログ出力

`Optimizer` は初期化時に `logging_generation` を `GenerationStartEvent` へ自動登録します．Python の `logging` モジュールを有効にするだけで，世代ごとの情報が出力されます．

```python
import logging
import numpy as np
from saealib import minimize

logging.basicConfig(level=logging.INFO, format="%(message)s")

def sphere(x):
    return np.sum(x ** 2)

result = minimize(sphere, dim=5, lb=[-5.0]*5, ub=[5.0]*5, max_fe=300, seed=0)
```

出力例（単目的）：

```
Generation 1 started. fe: 25. Best f: [0.42]
Generation 2 started. fe: 27. Best f: [0.31]
...
```

多目的の場合はパレートフロントのサイズと各目的の範囲が出力されます．

---

## カスタムハンドラの登録

低レベルAPIでは `optimizer.cbmanager.register()` でハンドラを追加できます．ハンドラはイベントオブジェクトを受け取る関数です．

```python
from saealib.callback import GenerationEndEvent

def my_handler(event: GenerationEndEvent) -> None:
    ctx = event.ctx
    best_f = ctx.archive.get_array("f").min()
    print(f"gen={ctx.gen}  fe={ctx.fe}  best={best_f:.4f}")

optimizer.cbmanager.register(GenerationEndEvent, my_handler)
```

---

## 収束履歴の記録

クロージャを使うと，外部のリストへ結果を蓄積できます．

```python
import numpy as np
from saealib.problem import Problem
from saealib.optimizer import Optimizer
from saealib.algorithms.ga import GA
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.acquisition.mean import MeanPrediction
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.execution.initializer import LHSInitializer
from saealib.termination import Termination, max_fe
from saealib.callback import GenerationEndEvent

DIM = 5
problem = Problem(
    func=lambda x: np.sum(x ** 2),
    dim=DIM,
    n_obj=1,
    weight=np.array([-1.0]),
    lb=[-5.0] * DIM,
    ub=[ 5.0] * DIM,
)

history: list[tuple[int, float]] = []

def record_history(event: GenerationEndEvent) -> None:
    ctx = event.ctx
    best_f = float(ctx.archive.get_array("f").min())
    history.append((ctx.fe, best_f))

optimizer = (
    Optimizer(problem)
    .set_initializer(LHSInitializer(n_init_archive=5*DIM, n_init_population=4*DIM, seed=0))
    .set_algorithm(GA(
        crossover=CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4),
        mutation=MutationUniform(mutation_rate=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    ))
    .set_surrogate_manager(LocalSurrogateManager(
        RBFsurrogate(gaussian_kernel, dim=DIM),
        MeanPrediction(weights=np.array([-1.0])),
        n_neighbors=20,
    ))
    .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.1))
    .set_termination(Termination(max_fe(300)))
)

optimizer.cbmanager.register(GenerationEndEvent, record_history)
optimizer.run()

for fe, best in history:
    print(f"fe={fe:4d}  best_f={best:.6f}")
```

---

## 多目的問題: ハイパーボリュームの追跡

`logging_generation_hv` はハイパーボリュームを世代ごとにログ出力するファクトリ関数です．基準点（リファレンスポイント）を渡します．

```python
import logging
from saealib.callback import GenerationStartEvent, logging_generation_hv

logging.basicConfig(level=logging.INFO, format="%(message)s")

hv_handler = logging_generation_hv(reference_point=np.array([1.1, 1.1]))
optimizer.cbmanager.register(GenerationStartEvent, hv_handler)
```

出力例：

```
Generation 1. fe: 50. HV: 0.312451
Generation 2. fe: 52. HV: 0.347820
...
```

ハイパーボリュームは最小化規約で計算されます．各目的の真の最良値より大きい値を基準点に設定してください．

---

## デフォルトハンドラの無効化

`logging_generation` が不要な場合は `unregister` で除去できます．

```python
from saealib.callback import logging_generation, GenerationStartEvent

optimizer.cbmanager.unregister(GenerationStartEvent, logging_generation)
```

既存ハンドラを別のものに置き換えるには `replace` を使います．

```python
optimizer.cbmanager.replace(GenerationStartEvent, logging_generation, my_handler)
```

---

## `iterate()` との使い分け

世代ごとの処理は `iterate()` でも書けますが，2 つのアプローチにはそれぞれ適した用途があります．

| 方法 | 適した用途 |
|------|-----------|
| `iterate()` | スクリプト内で完結する一時的な処理，早期終了のカスタム条件 |
| `cbmanager` | 再利用可能なログや記録，複数の Optimizer に共通して使いたい処理 |

---

## 利用可能なイベント一覧

| イベント | 発火タイミング |
|---------|--------------|
| `RunStartEvent` | 最適化の開始時（1回） |
| `RunEndEvent` | 最適化の終了時（1回） |
| `GenerationStartEvent` | 各世代の開始時 |
| `GenerationEndEvent` | 各世代の終了時 |
| `PostAskEvent` | アルゴリズムが候補を生成した直後 |
| `PostSurrogateFitEvent` | サロゲートのフィット直後 |
| `PostEvaluationEvent` | 真の関数評価直後 |

---

## 参照

- {py:class}`saealib.CallbackManager`
- {py:func}`saealib.logging_generation` / {py:func}`saealib.logging_generation_hv`
- {py:class}`saealib.GenerationStartEvent` / {py:class}`saealib.GenerationEndEvent`
- {py:class}`saealib.RunStartEvent` / {py:class}`saealib.RunEndEvent`
- {py:class}`saealib.PostAskEvent` / {py:class}`saealib.PostCrossoverEvent` / {py:class}`saealib.PostMutationEvent`
- {py:class}`saealib.PostSurrogateFitEvent` / {py:class}`saealib.PostEvaluationEvent`
- {py:func}`saealib.hypervolume`
