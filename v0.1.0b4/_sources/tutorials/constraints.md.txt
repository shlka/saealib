# 制約付き最適化

このチュートリアルでは，設計変数に不等式制約がある問題の扱い方を説明します．

```{warning}
今後制約処理の方法は強化する予定です．インターフェースの変更は未定ですが，変更される可能性があることに留意してください．
```

```{note}
`Constraint` は `InequalityConstraint` へリネームされました．`Constraint` は非推奨エイリアスとして当面残りますが，使用すると `DeprecationWarning` が発生します．新しいコードでは `InequalityConstraint` を使用してください．
```

## 制約の定義

`saealib` の制約は **不等式制約** `g(x) <= threshold` の形式で表現します．`InequalityConstraint` クラスを使って定義します．

```python
from saealib.problem import InequalityConstraint

# g(x) = x[0]^2 + x[1]^2 - 1 <= 0  (原点からの距離が 1 以内)
c1 = InequalityConstraint(func=lambda x: x[0]**2 + x[1]**2 - 1.0)

# g(x) <= threshold を明示する場合
# g(x) = x[0] - 0.5 <= 0
c2 = InequalityConstraint(func=lambda x: x[0], threshold=0.5)
```

`InequalityConstraint` は `violation(x)` で違反量（`max(0, g(x) - threshold)`）を返します．制約を満たしていれば `0.0` になります．

---

## `Problem` への制約の組み込み

`constraints` 引数にリストで渡します．

```python
import numpy as np
from saealib.problem import Problem, InequalityConstraint

def objective(x):
    return np.sum(x ** 2)

# x[0]^2 + x[1]^2 <= 1 の制約
c1 = InequalityConstraint(func=lambda x: x[0]**2 + x[1]**2 - 1.0)

problem = Problem(
    func=objective,
    dim=2,
    n_obj=1,
    weight=np.array([-1.0]),
    lb=[-2.0, -2.0],
    ub=[ 2.0,  2.0],
    constraints=[c1],
)
```

---

## `minimize` で制約付き問題を解く

`Problem` インスタンスを `minimize` の第一引数として直接渡します．`dim`, `lb`, `ub` は `Problem` から取得されるため省略します．

```python
from saealib import minimize

result = minimize(problem, max_fe=300, seed=0)

print(result.X)   # 最適解の設計変数
print(result.F)   # 最適解の目的関数値
print(result.fe)  # 真の関数評価回数
```

制約が複数ある場合もリストに追加するだけです．

```python
c1 = InequalityConstraint(func=lambda x: x[0]**2 + x[1]**2 - 1.0)  # x[0]^2 + x[1]^2 <= 1
c2 = InequalityConstraint(func=lambda x: -x[0])                     # x[0] >= 0

problem = Problem(
    func=objective,
    dim=2,
    n_obj=1,
    weight=np.array([-1.0]),
    lb=[-2.0, -2.0],
    ub=[ 2.0,  2.0],
    constraints=[c1, c2],
)

result = minimize(problem, max_fe=300, seed=0)
```

---

## 不等式制約以外の表現方法

`InequalityConstraint` は `g(x) <= threshold` のみをサポートしていますが，以下の変換で他の制約形式も扱えます．

### `>=` 制約（下界制約）

`g(x) >= c` は両辺を符号反転して `−g(x) <= −c` に変換します．

```python
# x[0] >= 0.5  →  -x[0] <= -0.5
c_ge = InequalityConstraint(func=lambda x: -x[0], threshold=-0.5)
```

### 等式制約

`g(x) == c` は **2 つの不等式制約** に分解します．

- `g(x) <= c + ε`
- `−g(x) <= −c + ε`（すなわち `g(x) >= c − ε`）

ε は許容誤差で，問題スケールに合わせて設定します．

```python
# x[0] + x[1] == 1.0  (許容誤差 ε = 1e-3)
eps = 1e-3
c_eq_upper = InequalityConstraint(func=lambda x:  (x[0] + x[1]),  threshold= 1.0 + eps)
c_eq_lower = InequalityConstraint(func=lambda x: -(x[0] + x[1]),  threshold=-1.0 + eps)

problem = Problem(
    func=objective,
    dim=2,
    n_obj=1,
    weight=np.array([-1.0]),
    lb=[-2.0, -2.0],
    ub=[ 2.0,  2.0],
    constraints=[c_eq_upper, c_eq_lower],
)
```

> **注意**: 等式制約の実行可能領域は `2ε` 幅の細い帯状領域になります．ε が小さいほど SAEA が偶然その帯を踏む確率が低くなり，実行可能解を見つけるために必要な関数評価回数が急増します．不等式制約（`<=`, `>=`）と比べて探索が格段に難しいため，`archive.get_array("cv")` で実行可能解の数を確認しながら ε や `max_fe` を調整してください．

---

## 制約違反量の確認

アーカイブには各解の制約違反量（`cv`）が記録されています．`cv == 0.0` の解が制約を満たしている解です．

```python
from saealib.optimizer import Optimizer
from saealib.execution.initializer import LHSInitializer
from saealib.termination import Termination, max_fe
from saealib.algorithms.ga import GA
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel
from saealib.surrogate.manager import LocalSurrogateManager
from saealib.acquisition.mean import MeanPrediction
from saealib.strategies.ib import IndividualBasedStrategy

algorithm = GA(
    crossover=CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4),
    mutation=MutationUniform(mutation_rate=0.3),
    parent_selection=SequentialSelection(),
    survivor_selection=TruncationSelection(),
)

surrogate_manager = LocalSurrogateManager(
    RBFsurrogate(gaussian_kernel, dim=2),
    MeanPrediction(weights=np.array([-1.0])),
    n_neighbors=20,
)

ctx = (
    Optimizer(problem)
    .set_initializer(LHSInitializer(n_init_archive=10, n_init_population=8, seed=0))
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.1))
    .set_termination(Termination(max_fe(300)))
    .run()
)

archive_x = ctx.archive.get_array("x")
archive_f = ctx.archive.get_array("f")
archive_cv = ctx.archive.get_array("cv")

# 制約を満たす解だけを抽出
feasible_mask = archive_cv == 0.0
feasible_x = archive_x[feasible_mask]
feasible_f = archive_f[feasible_mask]

print(f"実行可能解の数: {feasible_mask.sum()} / {len(archive_cv)}")
if len(feasible_f):
    best_idx = int(np.argmin(feasible_f))
    print(f"最適解 (制約満足): {feasible_x[best_idx]}")
    print(f"目的値:           {feasible_f[best_idx]}")
```

---

## 制約処理の差し替え（`ConstraintHandler`）

制約処理の方法（制約違反量 `cv` の集約方法，目的関数へのペナルティ付与，実行可能判定のしきい値など）は `ConstraintHandler` として `Problem` に注入できます．既定では `StaticToleranceHandler` が使われ，`cv` は各制約違反量の総和 `sum(max(0, g_i - threshold_i))`，実行可能判定のしきい値は固定の `eps_cv` になります（従来の挙動）．

```python
from saealib.problem import Problem, InequalityConstraint, ConstraintHandler

class PenaltyHandler(ConstraintHandler):
    """違反量の総和を cv とし，目的関数にペナルティを加える例．"""

    def __init__(self, coef: float = 1e3):
        self.coef = coef

    def compute_cv(self, constraints, x, g):
        return float(sum(max(0.0, gi - c.threshold) for gi, c in zip(g, constraints)))

    def augment_objective(self, f, constraints, x, g):
        return f + self.coef * self.compute_cv(constraints, x, g)

problem = Problem(
    func=objective,
    dim=2,
    n_obj=1,
    weight=np.array([-1.0]),
    lb=[-2.0, -2.0],
    ub=[ 2.0,  2.0],
    constraints=[c1],
    handler=PenaltyHandler(coef=1e3),
)
```

`ConstraintHandler` は制約処理のライフサイクルをオーバーライド可能なフックとして公開します．

| フック | 役割 | 既定 |
|---|---|---|
| `repair(x, constraints)` | 評価前の解修復 | 無修復（`x` を返す） |
| `compute_cv(constraints, x, g)` | 制約違反量 `cv` の集約（抽象メソッド） | — |
| `augment_objective(f, constraints, x, g)` | 目的関数の変換（ペナルティ等） | 恒等（`f` を返す） |
| `feasibility_threshold` | 実行可能判定のしきい値 `eps_cv` | `1e-6` |
| `on_generation_end(gen, population)` | 世代末の処理 | no-op |

`compute_cv` のみが抽象メソッドで，残りのフックは既定で no-op です．必要なフックだけをオーバーライドしてください．勾配ベースの修復を行う場合は，`InequalityConstraint.gradient(x)` をサブクラスで実装して制約のヤコビアンを提供できます．

---

## 参照

- {py:class}`saealib.InequalityConstraint`
- {py:class}`saealib.ConstraintHandler`
- {py:class}`saealib.StaticToleranceHandler`
- {py:class}`saealib.Problem`
- {py:func}`saealib.minimize`
- {py:class}`saealib.Optimizer`
