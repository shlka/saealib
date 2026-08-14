---
primary_layer: cross
related_layers: []
page_type: guide
---

# ビルトインコンポーネントを選んで差し替える

前提は、高レベルAPIまたは `Optimizer` の実行経路を使うことです。
:::{admonition} このページでできるようになること
:class: tip

このページを終えると、変更したい責務に対応するビルトインコンポーネントを選び、適切な差し替え点へ渡せます。
:::

構成要素の索引は [最適化の構成要素](../concepts/index.md) で、独自コンポーネントが必要な場合は [独自Component](custom_components.md) を参照します。

importの方針は [Canonical Imports](../api/imports.md) にまとめています。

## 変更したい責務から選ぶ

コンポーネントは、同じ処理の別実装を選ぶための差し替え点です。
`AcquisitionFunction` と `SurrogateManager` は独立した差し替え点であり、一方が他方を包含する関係ではありません。

| 何を変えたいか | 交換するコンポーネント | 確認先 |
|---|---|---|
| 候補を生成する探索法 | `Algorithm` | [Algorithmの概念](../concepts/search_algorithms/algorithm.md)、[Algorithmリファレンス](../api/algorithms.md) |
| 交叉、突然変異、選択 | `Operator` | [Crossoverの概念](../concepts/search_algorithms/crossover.md)、[Mutationの概念](../concepts/search_algorithms/mutation.md)、[Operatorリファレンス](../api/operators.md) |
| 予測モデル | `Surrogate` | [Surrogateの概念](../concepts/surrogate_modeling/surrogate.md)、[Surrogateリファレンス](../api/surrogate.md) |
| 予測の学習データと予測の進め方 | `SurrogateManager` | [SurrogateManagerの概念](../concepts/surrogate_modeling/surrogate_manager.md)、[リファレンス](../api/surrogate.md) |
| 予測を候補スコアへ変換する基準 | `AcquisitionFunction` | [AcquisitionFunctionの概念](../concepts/surrogate_modeling/acquisition_functions.md)、[リファレンス](../api/acquisition.md) |
| Surrogate候補を真に評価する割合や順序 | `Strategy` | [Strategyの概念](../concepts/execution_and_evaluation/strategies.md)、[リファレンス](../api/strategies.md) |
| 解の順位付け | `Comparator` | [Comparatorの概念](../concepts/problem_and_ranking/comparators.md)、[リファレンス](../api/comparators.md) |

## 高レベルAPIで交換する

高レベルAPIでは、`algorithm`、`surrogate`、`strategy` に文字列またはインスタンスを渡します。
現在確認できる文字列は、アルゴリズムの `"GA"` と `"PSO"`、Surrogateの `"rbf"`、Strategyの `"ib"`、`"gb"`、`"ps"` です。

```python
import numpy as np
from saealib import minimize


def objective(x):
    return np.sum(x**2)

result = minimize(
    objective,
    dim=3,
    lb=[-5.0] * 3,
    ub=[5.0] * 3,
    algorithm="PSO",
    strategy="ib",
    max_fe=100,
    seed=0,
    verbose=False,
)
```

Surrogate単体のインスタンスは、内部で `LocalSurrogateManager` に組み込まれます。
ManagerやAcquisitionFunctionを個別に構成する場合は `Optimizer` を使います。

## Optimizerで独立した差し替え点を構成する

`Optimizer.set_*()` は、コンポーネントを独立して設定するためのビルダーです。
たとえば `set_surrogate_manager()` と `set_acquisition()` は別の呼び出しです。

```python
import numpy as np
from saealib import MeanPrediction, Optimizer, Problem
from saealib.surrogate import GlobalSurrogateManager, RBFSurrogate, gaussian_kernel


def objective(x):
    return np.sum(x**2)

problem = Problem(
    objective,
    dim=3,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0] * 3,
    ub=[5.0] * 3,
)
manager = GlobalSurrogateManager(RBFSurrogate(gaussian_kernel, dim=3))

optimizer = (
    Optimizer(problem, seed=0)
    .set_surrogate_manager(manager)
    .set_acquisition(MeanPrediction())
)
```

`Optimizer` に必要な他のコンポーネントは既定値で解決できます。
`set_*()` で設定したコンポーネントと省略したコンポーネントは、`run()` または `iterate()` が既定値を解決した後に検証されます。
必要なコンポーネントをすべて明示設定した構成では、実行前に `validate()` を直接呼び出して検証することもできます。
ビルトインコンポーネントで責務を満たせないときだけ、[独自Componentを実装する](custom_components.md) に進みます。

## 関連するConceptとReference

- [構成要素の概要](../concepts/index.md)
- [Optimizerリファレンス](../api/optimizer.md)
- [Registryリファレンス](../api/registry.md)
