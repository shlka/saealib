# 単目的最適化

このチュートリアルでは，高コストな目的関数を持つ単目的最適化問題を例に，`saealib` の使い方を段階的に説明します．

## 問題設定

シミュレーションなど評価コストが高い関数を想定します．ここでは Sphere 関数を例として使います．

```python
import numpy as np

def expensive_func(x):
    # 実際には呼び出しに時間がかかる関数を想定
    return np.sum(x ** 2)

DIM = 10
LB = [-5.0] * DIM
UB = [ 5.0] * DIM
```

---

## 高レベルAPI: `minimize`

最もシンプルな呼び出し方です．`dim`, `lb`, `ub` を指定するだけで実行できます．

```python
from saealib import minimize

result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, seed=0)

print(result.X)   # 最適解の設計変数  shape: (dim,)
print(result.F)   # 最適解の目的関数値  shape: (1,)
print(result.fe)  # 真の関数評価回数
print(result.gen) # 完了した世代数
```

`max_fe` を省略すると `200 * dim` が上限として使われます．評価回数を明示的に制限するには：

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, max_fe=500, seed=0)
```

---

## アルゴリズムの選択

`algorithm` 引数で進化的アルゴリズムを切り替えられます．

| 文字列 | クラス | 特徴 |
|--------|--------|------|
| `'GA'` | `GA` | 交叉・突然変異による探索（デフォルト） |
| `'PSO'` | `PSO` | 粒子の速度更新による探索 |

```python
# GA (デフォルト)
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, algorithm='GA', seed=0)

# PSO
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, algorithm='PSO', seed=0)
```

GA のパラメータ（交叉率など）を細かく調整したい場合は，インスタンスを直接渡します．

```python
from saealib.algorithms.ga import GA
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection

ga = GA(
    crossover=CrossoverBLXAlpha(crossover_rate=0.9, alpha=0.5),
    mutation=MutationUniform(mutation_rate=0.1),
    parent_selection=SequentialSelection(),
    survivor_selection=TruncationSelection(),
)

result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, algorithm=ga, seed=0)
```

---

## サロゲート支援戦略の選択

`strategy` 引数で，サロゲートモデルをどのように使うか（どの候補を真に評価するか）を制御します．

| 文字列 | クラス | 動作 |
|--------|--------|------|
| `'ib'` | `IndividualBasedStrategy` | 各世代の候補を個別にサロゲートで評価し，上位 `evaluation_ratio` 割のみを真に評価（デフォルト） |
| `'gb'` | `GenerationBasedStrategy` | `gen_ctrl` 世代分をサロゲートのみで回し，1世代だけ真に評価 |
| `'ps'` | `PreSelectionStrategy` | 大量の候補を生成してサロゲートで絞り込み，上位 `n_select` 個だけを真に評価 |

```python
# Individual-based (デフォルト): 評価コストが非常に高い場合に向く
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, strategy='ib', seed=0)

# Generation-based: サロゲートの信頼性が高いときに向く
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, strategy='gb', seed=0)

# Pre-selection: 候補数を大きく増やして探索したいときに向く
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, strategy='ps', seed=0)
```

---

## 低レベルAPI: `Optimizer`

コンポーネントを個別にインスタンス化して `Optimizer` に組み込む方法です．`minimize` では調整できない細かい設定が可能です．

### 基本的な組み立て

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

DIM = 10
problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    weight=np.array([-1.0]),  # -1: 最小化
    lb=[-5.0] * DIM,
    ub=[ 5.0] * DIM,
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
    MeanPrediction(weights=np.array([-1.0])),
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

archive_x = ctx.archive.get_array("x")
archive_f = ctx.archive.get_array("f")
best_idx = int(np.argmin(archive_f))
print("最適解:", archive_x[best_idx])
print("目的値:", archive_f[best_idx])
print("評価回数:", ctx.fe)
```

### 複数の終了条件

`Termination` には複数の条件を渡せます．いずれかが満たされた時点で終了します．

```python
from saealib.termination import Termination, max_fe, max_gen

termination = Termination(max_fe(500), max_gen(200))
```

カスタム条件を Lambda で追加することもできます．

```python
termination = Termination(
    max_fe(500),
    lambda ctx: ctx.archive.get_array("f").min() < 1e-4,
)
```

### `GlobalSurrogateManager` を使う

`LocalSurrogateManager` はデフォルトで近傍 k 点だけを使ってサロゲートを局所フィットします．アーカイブ全体を使ってグローバルなフィットを行うには `GlobalSurrogateManager` を使います．

```python
from saealib.surrogate.manager import GlobalSurrogateManager
from saealib.acquisition.mean import MeanPrediction

surrogate_manager = GlobalSurrogateManager(
    surrogate=RBFsurrogate(gaussian_kernel, dim=DIM),
    acquisition=MeanPrediction(weights=np.array([-1.0])),
)
```

### `PreSelectionStrategy` を使う

大量の候補をサロゲートで絞り込む戦略です．

```python
from saealib.strategies.ps import PreSelectionStrategy

strategy = PreSelectionStrategy(
    n_candidates=100,  # サロゲートで評価する候補数
    n_select=5,        # 真に評価する候補数
)
```

---

## 世代ごとのアクセス: `Optimizer.iterate()`

`run()` の代わりに `iterate()` を使うと，世代単位でコンテキストを取得できます．進捗の記録やカスタムな早期終了に使えます．

```python
optimizer = (
    Optimizer(problem)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_strategy(strategy)
    .set_termination(termination)
)

history = []
for ctx in optimizer.iterate():
    best_f = ctx.archive.get_array("f").min()
    history.append((ctx.fe, best_f))
    print(f"gen={ctx.gen:4d}  fe={ctx.fe:4d}  best_f={best_f:.6f}")

print("終了 — 評価回数:", ctx.fe)
```

---

## 参照

- {py:func}`saealib.minimize` / {py:func}`saealib.maximize`
- {py:class}`saealib.Optimizer`
- {py:class}`saealib.GA` / {py:class}`saealib.PSO`
- {py:class}`saealib.IndividualBasedStrategy` / {py:class}`saealib.GenerationBasedStrategy` / {py:class}`saealib.PreSelectionStrategy`
- {py:class}`saealib.LocalSurrogateManager` / {py:class}`saealib.GlobalSurrogateManager`
- {py:class}`saealib.RBFsurrogate`
- {py:class}`saealib.MeanPrediction`
- {py:class}`saealib.LHSInitializer`
- {py:class}`saealib.Termination` / {py:func}`saealib.max_fe` / {py:func}`saealib.max_gen`

