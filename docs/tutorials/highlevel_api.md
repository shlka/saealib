---
primary_layer: layer1
related_layers: []
page_type: tutorial
---

# 高レベルAPIで最適化を実行する

前提は、目的関数と設計変数の範囲が決まっていることです。
:::{admonition} このページでできるようになること
:class: tip

このページを終えると、最小化または最大化を実行し、`Result` から解と実行状況を取得できます。
:::

## Problemを準備する

目的関数を受け取る高レベルAPIでは、`dim`、`lb`、`ub` を同時に渡します。
目的関数は、設計変数の配列を受け取り、スカラーまたは目的数に対応する配列を返します。

```python
import numpy as np
from saealib import minimize


def objective(x):
    return np.sum(x**2)


result = minimize(
    objective,
    dim=3,
    lb=[-5.0, -5.0, -5.0],
    ub=[5.0, 5.0, 5.0],
    max_fe=100,
    seed=0,
    verbose=False,
)
```

`Problem` をすでに構築している場合は、そのインスタンスを第一引数に渡します。
`Problem` の目的方向、変数、制約の設計は [Problemの概念](../concepts/problem_and_ranking/problem.md) を参照してください。
importの使い分けは [Canonical Imports](../api/imports.md) にまとめています。

## minimizeとmaximizeを選ぶ

目的方向が一つなら、関数名で実行の意図を表せます。
`minimize` は既定で全目的を最小化し、`maximize` は既定で全目的を最大化します。

| 要件 | 呼び出し | 追加の指定 |
|---|---|---|
| すべて最小化 | `minimize(...)` | なし |
| すべて最大化 | `maximize(...)` | なし |
| Problemで方向を定義済み | `minimize(problem)` または `maximize(problem)` | `Problem` の方向が使われる |
| 複数目的の方向を混在させる | `minimize(..., direction=[...])` | 目的ごとに `minimize` または `maximize` を指定 |

多目的の目的関数や方向の定義は [多目的最適化](multi_objective.md) に、制約は [制約付き最適化](constraints.md) に、混合変数は [混合変数](mixed_variable.md) に分けています。

## 実行を制御する

`max_fe` は真の目的関数評価の上限です。
省略時は `200 * dim` が使われます。
`pop_size` は集団サイズ、`seed` は初期化に使う乱数シード、`verbose=False` は世代ログの抑制に使います。

アルゴリズム、Surrogate、Strategyをビルトインコンポーネントから選ぶ場合は、文字列またはインスタンスを `algorithm`、`surrogate`、`strategy` に渡せます。
選択と差替えの手順は [ビルトインコンポーネントを差し替える](component_swap.md) に分けています。

## Resultを読む

`minimize()` と `maximize()` は `Result` を返します。
単目的では `result.x` が最良設計変数、`result.f` が要素数1の配列として対応する目的値を保持します。
`result.fe` は真の評価回数、`result.gen` は完了した世代数です。
多目的では `x` と `f` がPareto解の行列になります。

```python
print(result.x)
print(result.f)
print(result.fe, result.gen)

archive = result.ctx.archive
print(archive.get_array("x"))
```

`result.ctx` は完全な `OptimizationState` です。
世代ごとの状態取得、途中の停止、コンポーネントの組み立てが必要になったら [低レベルAPI](lowlevel_api.md) に進みます。

## 関連するConceptとReference

- [Problem](../concepts/problem_and_ranking/problem.md)
- [高レベルAPIリファレンス](../api/highlevel.md)
