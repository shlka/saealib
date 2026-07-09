# カスタムコンポーネント

このチュートリアルでは，コールバックによる候補解の書き換えと，`AcquisitionFunction` を継承したカスタム獲得関数の実装方法を説明します．

## コールバックによる候補解の書き換え

`PostCrossoverEvent`・`PostMutationEvent`・`PostAskEvent` は，アルゴリズムが候補解を生成した直後に発火します．これらのイベントオブジェクトが持つ `candidates` フィールドを書き換えると，アルゴリズムはその変更後の候補解を使って処理を続けます．

```{note}
`PostCrossoverEvent` と `PostMutationEvent` は GA 固有のイベントです．PSO を使う場合は `PostAskEvent` を使ってください．
```

### 設計変数の一部を固定する

特定の変数を定数に固定して，低次元の部分空間だけを探索したい場合の例です．

```python
from saealib.callback import PostAskEvent

def fix_x0(event: PostAskEvent) -> None:
    event.candidates[:, 0] = 0.0  # x[0] を常に 0 に固定

optimizer.cbmanager.register(PostAskEvent, fix_x0)
```

### カスタム修復（グリッドへのスナップ）

設計変数が離散グリッド上に制限される問題では，突然変異後に丸め処理を挟めます．

```python
import numpy as np
from saealib.callback import PostMutationEvent

def snap_to_grid(event: PostMutationEvent) -> None:
    event.candidates = np.round(event.candidates, decimals=1)

optimizer.cbmanager.register(PostMutationEvent, snap_to_grid)
```

### 既知の良解を候補に挿入する

ドメイン知識として良いと分かっている解を毎世代 1 つ混ぜることで，収束を促進できます．

```python
from saealib.callback import PostAskEvent

known_good = np.array([0.1, -0.2, 0.3, 0.0, 0.5])

def inject_known_solution(event: PostAskEvent) -> None:
    if len(event.candidates) > 0:
        event.candidates[0] = known_good

optimizer.cbmanager.register(PostAskEvent, inject_known_solution)
```

---

## カスタム獲得関数の実装

`AcquisitionFunction` を継承して `compute_reference` と `score` を実装すると，独自の選択基準を定義できます．

| メソッド | 役割 |
|---------|------|
| `compute_reference(archive)` | アーカイブから参照値を計算して返す |
| `score(prediction, reference)` | `SurrogatePrediction` をスコア（高いほど良い）に変換する |

### 実装例: チェビシェフスカラー化

多目的問題でパレートフロントの特定の点（目標トレードオフ）を狙う獲得関数です．`MeanPrediction` の重み付き和とは異なり，各目的の最大乖離を最小化するため，均等なトレードオフが得やすくなります．

```python
import numpy as np
from saealib.acquisition.base import AcquisitionFunction
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.population import Archive

class ChebyshevScalarization(AcquisitionFunction):
    """
    チェビシェフスカラー化による獲得関数（最小化規約）．

    Parameters
    ----------
    weights : np.ndarray
        各目的の重み（正値）．shape: (n_obj,)
    target : np.ndarray or None
        目標目的値．None のとき，アーカイブの理想点（各目的の最小値）を使用．
    """

    def __init__(self, weights: np.ndarray, target: np.ndarray | None = None):
        self.weights = np.asarray(weights, dtype=float)
        self.target = target

    def compute_reference(self, archive: Archive) -> np.ndarray:
        if self.target is not None:
            return self.target
        return archive.get_array("f").min(axis=0)  # 理想点

    def score(self, prediction: SurrogatePrediction, reference: np.ndarray) -> np.ndarray:
        # weighted Chebyshev distance: max_i{ w_i * (mu_i - z_i) }
        diff = prediction.mean - reference      # (n_samples, n_obj)
        weighted = self.weights * diff          # (n_samples, n_obj)
        # 最大乖離が小さい候補ほど良い → 符号反転でスコア化
        return -np.max(weighted, axis=1)        # (n_samples,)
```

`LocalSurrogateManager` と組み合わせて使います：

```python
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel

DIM = 10
N_OBJ = 2

acquisition = ChebyshevScalarization(
    weights=np.array([1.0, 1.0]),   # 均等な重み
    target=np.array([0.0, 0.0]),    # 理想点を目標に（Noneにするとアーカイブから自動計算）
)

surrogate_manager = LocalSurrogateManager(
    RBFSurrogate(gaussian_kernel, dim=DIM),
    acquisition,
    n_neighbors=30,
)
```

あとは通常の低レベル API と同様に `Optimizer` に渡します：

```python
from saealib.problem import Problem
from saealib.optimizer import Optimizer
from saealib.algorithms.ga import GA
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.execution.initializer import LHSInitializer
from saealib.termination import Termination, max_fe
from saealib.comparators import non_dominated_sort

def zdt1(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.array([f1, f2])

problem = Problem(
    func=zdt1,
    dim=DIM,
    n_obj=N_OBJ,
    weight=np.array([-1.0, -1.0]),
    lb=[0.0] * DIM,
    ub=[1.0] * DIM,
)

ctx = (
    Optimizer(problem)
    .set_initializer(LHSInitializer(n_init_archive=5*DIM, n_init_population=4*DIM, seed=0))
    .set_algorithm(GA(
        crossover=CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4),
        mutation=MutationUniform(mutation_rate=0.3),
        parent_selection=SequentialSelection(),
        survivor_selection=TruncationSelection(),
    ))
    .set_surrogate_manager(surrogate_manager)
    .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.1))
    .set_termination(Termination(max_fe(500)))
    .run()
)

archive_f = ctx.archive.get_array("f")
_, fronts = non_dominated_sort(archive_f)
pareto_f = archive_f[fronts[0]]
print(f"パレートフロント上の解数: {len(fronts[0])}")
```

---

## カスタム評価器（Evaluator）の実装

ライブラリ内で真の（高コストな）目的関数評価を行う処理は，すべて `Evaluator` という単一の入口に集約されています．`Evaluator` を継承して `evaluate_batch` を実装すると，並列実行・リモート実行・キャッシュなどの評価バックエンドを，パイプライン本体に手を入れずに差し替えられます．

| 役割 | 内容 |
|---------|------|
| `evaluate_batch(x, problem)` | 設計変数のバッチ `x`（shape: `(n, dim)`）を評価し，`EvaluationResult` を返す |
| `EvaluationResult` | `f`（目的関数値 `(n, n_obj)`）・`g`（生制約値 `(n, n_constraints)`）・`cv`（制約違反量 `(n,)`）を保持する |

デフォルトは候補を逐次評価する `SerialEvaluator` です．

### 実装例: スレッドプールによる並列評価

```python
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from saealib import Problem
from saealib.execution.evaluator import EvaluationResult, Evaluator


class ParallelEvaluator(Evaluator):
    """スレッドプールで候補バッチを並列評価する評価器．"""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def evaluate_batch(self, x: np.ndarray, problem: Problem) -> EvaluationResult:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(x)
        f = np.empty((n, problem.n_obj), dtype=float)
        g = np.empty((n, problem.n_constraints), dtype=float)
        cv = np.zeros(n, dtype=float)

        def _eval(xi):
            g_i, cv_i = problem.evaluate_constraints(xi)
            return problem.evaluate(xi), g_i, cv_i

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for i, (f_i, g_i, cv_i) in enumerate(pool.map(_eval, x)):
                f[i], g[i], cv[i] = f_i, g_i, cv_i

        return EvaluationResult(f=f, g=g, cv=cv)
```

`Optimizer.set_evaluator()` で差し込みます：

```python
optimizer.set_evaluator(ParallelEvaluator(max_workers=4))
```

```{note}
真の並列性が必要な場合は，`ThreadPoolExecutor` を `ProcessPoolExecutor`・joblib・MPI などに置き換えてください（目的関数が pickle 可能である必要があります）．
```

### 追加の評価出力を持たせたい場合

評価コスト・勾配・ノイズ推定など，`f` / `g` / `cv` 以外の値を扱いたい場合は，`EvaluationResult` を**継承**して必要なフィールドを追加します．コア最適化ループは基底クラスの `f` / `g` / `cv` のみを参照するため，追加フィールドを持つサブクラスを返しても安全に動作します．

```python
from dataclasses import dataclass

import numpy as np
from saealib.execution.evaluator import EvaluationResult


@dataclass
class CostAwareResult(EvaluationResult):
    cost: np.ndarray  # 各候補の評価コスト. shape: (n,)
```

追加フィールドは，独自の `SurrogateManager` やコールバックから参照して活用できます．

---

## 参照

- {py:class}`saealib.AcquisitionFunction` / {py:class}`saealib.SurrogatePrediction`
- {py:class}`saealib.PostAskEvent` / {py:class}`saealib.PostCrossoverEvent` / {py:class}`saealib.PostMutationEvent`
- {py:class}`saealib.CallbackManager`
- {py:class}`saealib.Evaluator` / {py:class}`saealib.SerialEvaluator` / {py:class}`saealib.EvaluationResult`
