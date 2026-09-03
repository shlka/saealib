---
primary_layer: layer2
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

コンポーネントは、同じ処理の別実装を選ぶための差し替え点です。`AcquisitionFunction`と`SurrogateManager`は独立した差し替え点であり、一方が他方を内包することはありません。

| このページでできるようになること | このページでできるようになること | 参照先 |
|---|---|---|
| このページでできるようになること | `Algorithm` | [Algorithm concept](../concepts/search_algorithms/algorithm.md)、[Algorithm reference](../api/algorithms.md) |
| このページでできるようになること | `Operator` | [交叉の概念](../concepts/search_algorithms/crossover.md)、[突然変異の概念](../concepts/search_algorithms/mutation.md)、[演算子リファレンス](../api/operators.md) |
| このページでできるようになること | `Surrogate` | [サロゲートモデルの概念](../concepts/surrogate_modeling/surrogate.md)、[サロゲートモデルの参照](../api/surrogate.md) |
| このページでできるようになること | `SurrogateManager` | [SurrogateManager concept](../concepts/surrogate_modeling/surrogate_manager.md)、[Reference](../api/surrogate.md) |
| このページでできるようになること | `AcquisitionFunction` | [AcquisitionFunction concept](../concepts/surrogate_modeling/acquisition_functions.md)、[Reference](../api/acquisition.md) |
| 真の評価に送るサロゲート候補の割合または順序 | `Strategy` | [Strategyの概念](../concepts/execution_and_evaluation/strategies.md)、[参照](../api/strategies.md) |
| このページでできるようになること | `Comparator` | [Comparatorの概念](../concepts/problem_and_ranking/comparators.md)、[参照](../api/comparators.md) |

## 高レベルAPIで交換する

高レベルAPIでは、`algorithm`、`surrogate`、`strategy`に文字列またはインスタンスを渡します。利用できる文字列は、アルゴリズムが`"GA"`と`"PSO"`、Surrogateが`"rbf"`、Strategyが`"ib"`、`"gb"`、`"ps"`です。

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

Surrogate単体のインスタンスは内部で`LocalSurrogateManager`にラップされます。ManagerやAcquisitionFunctionを個別に構成する場合は`Optimizer`を使います。

## Optimizerで独立した差し替え点を構成する

`Optimizer.set_*()`はコンポーネントを独立して構成するためのビルダーです。たとえば`set_surrogate_manager()`と`set_acquisition()`は別々に呼び出します。

```python
import numpy as np
from saealib import MeanPrediction, Optimizer, Problem
from saealib.surrogate import GaussianKernel, GlobalSurrogateManager, RBFSurrogate


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
manager = GlobalSurrogateManager(RBFSurrogate(kernel=GaussianKernel()))

optimizer = (
    Optimizer(problem, seed=0)
    .set_surrogate_manager(manager)
    .set_acquisition(MeanPrediction())
)
```

`Optimizer`に必要な他のコンポーネントは既定値から解決できます。`set_*()`で構成したコンポーネントと省略したコンポーネントは、`run()`または`iterate()`が既定値を解決した後に検証されます。必要なコンポーネントをすべて明示した場合は、実行前に`validate()`を直接呼び出すこともできます。ビルトインコンポーネントで責務を満たせない場合に限り、[独自コンポーネントの実装](custom_components.md)へ進みます。

## 関連するConceptとReference

- [構成要素の概要](../concepts/index.md)
- [Optimizerリファレンス](../api/optimizer.md)
- [Registryリファレンス](../api/registry.md)
