# 単目的最適化

評価コストの高い目的関数を持つ単目的最適化問題を、`saealib`で解きます。

まず問題を定義し、高レベルAPIの`minimize`で解いたあと、`Optimizer`による低レベルAPIへ進みます。

各コンポーネントの詳しい仕様やカスタマイズ方法は、この後の節からリンクする[コンポーネント](../components/index.md)配下の各ページを参照してください。

## 問題設定

シミュレーションのように、1回の呼び出しに時間がかかる目的関数を想定します。

ここでは例として、評価コストの高さを模したSphere関数を最小化します。

```python
import numpy as np


def expensive_func(x):
    # assume a function that is expensive to call in practice
    return np.sum(x**2)


DIM = 10
LB = [-5.0] * DIM
UB = [5.0] * DIM
```

`DIM`は設計変数の次元数、`LB`と`UB`はその下界と上界を与える`DIM`次元のリストです。

目的関数は、`DIM`次元の設計変数を受け取り、目的関数値を返す`Callable`として定義します。

## 高レベルAPI: minimize / maximize

`minimize`は、`dim`、`lb`、`ub`を指定するだけで最適化を実行できる高レベルAPIです。

```python
from saealib import minimize

result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, seed=0)

print(result.x)   # optimal design variables  shape: (dim,)
print(result.f)   # optimal objective value  shape: (n_obj,)
print(result.fe)  # true function evaluations
print(result.gen) # completed generations
```

最大評価回数`max_fe`を省略すると、`200 * dim`が既定値として使われます。

評価回数を明示的に制限するには、次のように指定します。

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, max_fe=500, seed=0)
```

## コンポーネントの切り替え

`minimize`は、進化的アルゴリズム、サロゲートモデル、評価戦略という3つのコンポーネントを、それぞれ`algorithm`、`surrogate`、`strategy`引数の文字列で切り替えられます。

3つとも、文字列の代わりにインスタンスを直接渡すこともできます。

各コンポーネントの内部動作やカスタマイズ方法は、[Algorithm](../components/algorithm.md)、[Surrogate](../components/surrogate.md)、[OptimizationStrategy](../components/strategies.md)のページで扱います。

### アルゴリズム

`algorithm`引数は、候補解を生成する進化的アルゴリズムを選びます。

| 文字列 | クラス | 特徴 |
|--------|--------|------|
| `'GA'` | `GA` | 交叉・突然変異による探索（既定） |
| `'PSO'` | `PSO` | 粒子の速度更新による探索 |

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, algorithm="PSO", seed=0)
```

### サロゲート

`surrogate`引数は、目的関数を近似するサロゲートモデルを選びます。

| 文字列 | 解決される構成 | 説明 |
|--------|--------|------|
| `'rbf'` | `RBFSurrogate` + `LocalSurrogateManager`（既定） | ガウスRBFカーネルによる近傍点の局所フィット |

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, surrogate="rbf", seed=0)
```

### 評価戦略

`strategy`引数は、生成した候補解のうちどれに真の（高コストな）評価を行うかを決める評価戦略を選びます。

| 文字列 | クラス | 動作 |
|--------|--------|------|
| `'ib'` | `IndividualBasedStrategy` | 候補を個別にサロゲートで評価し、上位`evaluation_ratio`割だけを真に評価する（既定） |
| `'gb'` | `GenerationBasedStrategy` | `gen_ctrl`世代分をサロゲートのみで進め、1世代だけ真に評価する |
| `'ps'` | `PreSelectionStrategy` | 大量の候補をサロゲートで絞り込み、上位`n_select`個だけを真に評価する |

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, strategy="ib", seed=0)
```

## 低レベルAPI: Optimizer

`minimize`は各コンポーネントを既定の組み合わせで結び付けますが、個々のパラメータまでは調整できません。

コンポーネントを個別にインスタンス化し、`Optimizer`に組み込めば、この制約を外せます。

```python
import numpy as np
from saealib import (
    Problem,
    Optimizer,
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
    RBFSurrogate,
    gaussian_kernel,
    LocalSurrogateManager,
    MeanPrediction,
    IndividualBasedStrategy,
    LHSInitializer,
    Termination,
    max_fe,
)

DIM = 10
SEED = 0

problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),  # -1: minimize
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
)

algorithm = GA(
    crossover=CrossoverBLXAlpha(0.7, 0.4),
    mutation=MutationUniform(0.3),
    parent_selection=SequentialSelection(),
    survivor_selection=TruncationSelection(),
)

surrogate_manager = LocalSurrogateManager(
    RBFSurrogate(gaussian_kernel, dim=DIM),
    MeanPrediction(),  # direction is auto-injected from problem.direction
)

strategy = IndividualBasedStrategy(evaluation_ratio=0.1)

initializer = LHSInitializer(
    n_init_archive=5 * DIM,
    n_init_population=4 * DIM,
    seed=SEED,
)

termination = Termination(max_fe(500))

ctx = (
    Optimizer(problem, seed=SEED)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_strategy(strategy)
    .set_termination(termination)
    .run()
)

archive_x = ctx.archive.get_array("x")
archive_f = ctx.archive.get_array("f")[:, 0]
best_idx = int(np.argmin(archive_f))
print("最適解:", archive_x[best_idx])
print("目的値:", archive_f[best_idx])
print("評価回数:", ctx.fe)
```

乱数シードは、`Optimizer(problem, seed=SEED)`と`LHSInitializer(..., seed=SEED)`の両方に同じ値を渡してください。

`Optimizer`の`seed`は、`set_initializer()`を呼ばずに済ませたとき（`minimize`/`maximize`など）だけ、既定の`LHSInitializer`へ自動的に伝播します。

`Initializer`を自分で組み立てる場合は、明示的に渡す必要があります。

`Termination`には複数の条件を渡せます。

列挙した条件は、いずれか一つが満たされた時点で終了します（OR結合）。

```python
from saealib import Termination, max_fe, max_gen

termination = Termination(max_fe(500), max_gen(200))
```

カスタム条件をlambdaで追加することもできます。

```python
termination = Termination(
    max_fe(500),
    lambda ctx: ctx.archive.get_array("f")[:, 0].min() < 1e-4,
)
```

`run()`の代わりに`iterate()`を使うと、世代単位でコンテキストを取得できます。

進捗の記録や、カスタムな早期終了の実装に使えます。

```python
optimizer = (
    Optimizer(problem, seed=SEED)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_strategy(strategy)
    .set_termination(termination)
)

history = []
for ctx in optimizer.iterate():
    best_f = ctx.archive.get_array("f")[:, 0].min()
    history.append((ctx.fe, best_f))
    print(f"gen={ctx.gen:4d}  fe={ctx.fe:4d}  best_f={best_f:.6f}")

print("評価回数:", ctx.fe)
```

## 参照

- {py:func}`saealib.minimize` / {py:func}`saealib.maximize`
- {py:class}`saealib.Optimizer`
- {py:class}`saealib.GA` / {py:class}`saealib.PSO`
- {py:class}`saealib.IndividualBasedStrategy` / {py:class}`saealib.GenerationBasedStrategy` / {py:class}`saealib.PreSelectionStrategy`
- {py:class}`saealib.LocalSurrogateManager`
- {py:class}`saealib.RBFSurrogate`
- {py:class}`saealib.MeanPrediction`
- {py:class}`saealib.LHSInitializer`
- {py:class}`saealib.Termination` / {py:func}`saealib.max_fe` / {py:func}`saealib.max_gen`
