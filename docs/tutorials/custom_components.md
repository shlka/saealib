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
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel

DIM = 10
N_OBJ = 2

acquisition = ChebyshevScalarization(
    weights=np.array([1.0, 1.0]),   # 均等な重み
    target=np.array([0.0, 0.0]),    # 理想点を目標に（Noneにするとアーカイブから自動計算）
)

surrogate_manager = LocalSurrogateManager(
    RBFsurrogate(gaussian_kernel, dim=DIM),
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
