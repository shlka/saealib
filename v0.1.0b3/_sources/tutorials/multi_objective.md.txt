# 多目的最適化

このチュートリアルでは，複数の目的関数を同時に最適化する多目的最適化問題の扱い方を説明します．

## 問題設定

多目的最適化では，複数の目的間にトレードオフが存在するため，単一の「最適解」は存在しません．代わりに，どれも他に劣らない**パレートフロント**上の解集合が得られます．

ここでは ZDT1 問題（最小化，2 目的）を例として使います．

```python
import numpy as np

def zdt1(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.array([f1, f2])

DIM = 10
LB = [0.0] * DIM
UB = [1.0] * DIM
```

---

## 高レベルAPI: `minimize`

`n_obj` に目的数を指定します．それ以外の呼び出し方は単目的と同じです．

```python
from saealib import minimize

result = minimize(zdt1, dim=DIM, lb=LB, ub=UB, n_obj=2, max_fe=500, seed=0)
```

### 結果の読み方

多目的の場合，`result.X` と `result.F` はパレートフロント上の解集合になります．

```python
print(result.X.shape)  # (n_pareto, dim)
print(result.F.shape)  # (n_pareto, n_obj)
print(result.fe)       # 真の関数評価回数
```

パレートフロント上の解を目的空間で確認します．

```python
f1_vals = result.F[:, 0]
f2_vals = result.F[:, 1]

for f1, f2 in zip(f1_vals, f2_vals):
    print(f"f1={f1:.4f}  f2={f2:.4f}")
```

---

## アルゴリズムと戦略の選択

単目的と同様に `algorithm` と `strategy` を切り替えられます．

```python
# PSO + Pre-selection 戦略
result = minimize(
    zdt1,
    algorithm='PSO',
    dim=DIM,
    lb=LB,
    ub=UB,
    n_obj=2,
    strategy='ps',
    max_fe=500,
    seed=0,
)
```

---

## 低レベルAPI: `Optimizer`

`Problem` の `weight` に各目的の符号を指定します．最小化なら `-1.0`，最大化なら `1.0` です．

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
from saealib.comparators import non_dominated_sort

N_OBJ = 2
problem = Problem(
    func=zdt1,
    dim=DIM,
    n_obj=N_OBJ,
    weight=np.array([-1.0, -1.0]),  # 両目的を最小化
    lb=LB,
    ub=UB,
)

algorithm = GA(
    crossover=CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4),
    mutation=MutationUniform(mutation_rate=0.3),
    parent_selection=SequentialSelection(),
    survivor_selection=TruncationSelection(),
)

surrogate = RBFsurrogate(gaussian_kernel, dim=DIM)
surrogate_manager = LocalSurrogateManager(
    surrogate,
    MeanPrediction(weights=np.array([-1.0, -1.0])),
    n_neighbors=30,
)

strategy = IndividualBasedStrategy(evaluation_ratio=0.1)

initializer = LHSInitializer(
    n_init_archive=5 * DIM,
    n_init_population=4 * DIM,
    seed=0,
)

termination = Termination(max_fe(500))

ctx = (
    Optimizer(problem)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_strategy(strategy)
    .set_termination(termination)
    .run()
)
```

### パレートフロントの抽出

`non_dominated_sort` でアーカイブからパレートフロントを取り出します．

```python
archive_x = ctx.archive.get_array("x")
archive_f = ctx.archive.get_array("f")

_, fronts = non_dominated_sort(archive_f)
pareto_idx = fronts[0]

pareto_x = archive_x[pareto_idx]
pareto_f = archive_f[pareto_idx]

print(f"パレートフロント上の解数: {len(pareto_idx)}")
print(f"pareto_x.shape: {pareto_x.shape}")  # (n_pareto, dim)
print(f"pareto_f.shape: {pareto_f.shape}")  # (n_pareto, n_obj)
```

---

## `EnsembleSurrogateManager` を使う

多目的問題では，目的ごとに異なるサロゲートを用意して `EnsembleSurrogateManager` で組み合わせることができます．各サブマネージャーのスコアはランク正規化されたうえで加重平均されます．

```python
from saealib.surrogate.manager import LocalSurrogateManager, EnsembleSurrogateManager
from saealib.acquisition.mean import MeanPrediction
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel

manager_f1 = LocalSurrogateManager(
    RBFsurrogate(gaussian_kernel, dim=DIM),
    MeanPrediction(weights=np.array([-1.0, 0.0])),  # f1 のみを対象
    n_neighbors=30,
)
manager_f2 = LocalSurrogateManager(
    RBFsurrogate(gaussian_kernel, dim=DIM),
    MeanPrediction(weights=np.array([0.0, -1.0])),  # f2 のみを対象
    n_neighbors=30,
)

ensemble_manager = EnsembleSurrogateManager(
    managers=[manager_f1, manager_f2],
    weights=np.array([1.0, 1.0]),  # 均等に重み付け
)
```

```python
ctx = (
    Optimizer(problem)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_surrogate_manager(ensemble_manager)
    .set_strategy(strategy)
    .set_termination(termination)
    .run()
)
```

---

## 参照

- {py:func}`saealib.minimize`
- {py:class}`saealib.Optimizer`
- {py:class}`saealib.GA` / {py:class}`saealib.PSO`
- {py:class}`saealib.IndividualBasedStrategy` / {py:class}`saealib.GenerationBasedStrategy` / {py:class}`saealib.PreSelectionStrategy`
- {py:class}`saealib.LocalSurrogateManager` / {py:class}`saealib.EnsembleSurrogateManager`
- {py:class}`saealib.RBFsurrogate`
- {py:class}`saealib.MeanPrediction`
- {py:func}`saealib.non_dominated_sort`
- {py:class}`saealib.LHSInitializer`
- {py:class}`saealib.Termination` / {py:func}`saealib.max_fe`
