---
primary_layer: layer1
related_layers: [layer2]
page_type: tutorial
---

# 単目的最適化

評価コストの高い目的関数を持つ単目的最適化問題を、`saealib`で解きます。

まず問題を定義し、高レベルAPIの `minimize` で単目的最適化を実行します。

各コンポーネントの詳しい仕様やカスタマイズ方法は、この後の節からリンクする[コンポーネント](../concepts/index.md)配下の各ページを参照してください。

:::{admonition} このページでできるようになること
:class: tip

このページを終えると、高コストな単目的関数を `minimize()` で最適化できます。
:::

目的関数を渡してすぐ実行したいだけなら[高レベルAPI](highlevel_api.md)、コンポーネントを責務から選びたい場合は[ビルトインコンポーネントの差し替え](component_swap.md)を参照してください。このページでは`minimize()`による単目的最適化を実行し、文字列引数でアルゴリズム、Surrogate、評価Strategyを選びます。

## 問題を設定する

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

`DIM`は設計変数の次元数、`LB`と`UB`はその下限と上限を与える`DIM`次元のリストです。

目的関数は、`DIM`次元の設計変数を受け取り、目的関数値を返す`Callable`として定義します。

## 高レベルAPI：minimize / maximize

`minimize`は、`dim`、`lb`、`ub`を指定するだけで最適化を実行できる高レベルAPIです。

```python
from saealib import minimize

result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, seed=0)

print(result.x)  # optimal design variables  shape: (dim,)
print(result.f)  # optimal objective value  shape: (n_obj,)
print(result.fe)  # true function evaluations
print(result.gen)  # completed generations
```

最大評価回数`max_fe`を省略すると、`200 * dim`が既定値として使われます。

評価回数を明示的に制限するには、次のように指定します。

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, max_fe=500, seed=0)
```

## コンポーネントの切り替え

`minimize`は、進化的アルゴリズム、サロゲートモデル、評価戦略という3つのコンポーネントを、それぞれ`algorithm`、`surrogate`、`strategy`引数の文字列で切り替えられます。

3つとも、文字列の代わりにインスタンスを直接渡すこともできます。

各コンポーネントの内部動作やカスタマイズ方法は、[Algorithm](../concepts/search_algorithms/algorithm.md)、[Surrogate](../concepts/surrogate_modeling/surrogate.md)、[OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md)のページで扱います。

### アルゴリズム

`algorithm`引数は、候補解を生成する進化的アルゴリズムを選びます。

| 文字列 | クラス | 特徴 |
|--------|--------|------|
| `'GA'` | `GA` | 交叉と突然変異による探索（既定） |
| `'PSO'` | `PSO` | 粒子の速度更新による探索 |

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, algorithm="PSO", seed=0)
```

### Surrogate

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
| `'ib'` | `IndividualBasedStrategy` | 候補を個別にサロゲートで評価し、上位`evaluation_ratio`の割合だけを真に評価する（既定） |
| `'gb'` | `GenerationBasedStrategy` | `gen_ctrl`世代分をサロゲートのみで進め、1世代だけ真に評価する |
| `'ps'` | `PreSelectionStrategy` | 大量の候補をサロゲートで絞り込み、上位`n_select`個だけを真に評価する |

```python
result = minimize(expensive_func, dim=DIM, lb=LB, ub=UB, strategy="ib", seed=0)
```

`minimize()` の既定構成では足りない場合は、各コンポーネントを個別に構成して `Optimizer` に登録できます。[Optimizerを低レベルAPIで組み立てる](lowlevel_api.md) では、Terminationの複数条件や `iterate()` による世代ごとの進捗取得も扱います。

## 関連するConceptとReference

- {py:func}`saealib.minimize` / {py:func}`saealib.maximize`
- {py:class}`saealib.GA` / {py:class}`saealib.PSO`
- {py:class}`saealib.IndividualBasedStrategy` / {py:class}`saealib.GenerationBasedStrategy` / {py:class}`saealib.PreSelectionStrategy`
