---
primary_layer: layer2
page_type: guide
---

# サロゲート精度に応じた動的な切り替え

サロゲートモデルの予測精度は、世代が進むにつれて変化します。

学習データが少ない序盤は精度が低く、アーカイブが充実してくると精度が上がり、探索が収束して目的値の差が縮まると再び精度が落ちることもあります。

この変化に応じて、評価戦略や`SurrogateManager`を実行中に切り替える方法を扱います。

:::{admonition} このページでできるようになること
:class: tip

このページを終えると、Surrogateの精度を世代ごとに追跡し、その値に応じてStrategyやSurrogateManagerを実行中に切り替えられます。
:::

評価戦略を固定する場合は[低レベルAPI](lowlevel_api.md)を使い、独自の切り替え条件を実装する場合は[独自コンポーネント](custom_components.md)を参照してください。このページでは、サロゲートの精度を読み取りながら`iterate()`中に実行構成を切り替える方法を示します。

## Problemを準備する

ここでは単純なSphere関数を使います。

```python
import numpy as np


def expensive_func(x):
    return np.sum(x**2)


DIM = 10
SEED = 0
```

## サロゲートの精度を追跡する

`SurrogateManager`に`accuracy_evaluator`を渡すと、フィットのたびに精度が計算され、`surrogate_manager.last_accuracy`に記録されます。

`LOOAccuracyEvaluator`は、追加の検証用データを用意せず、現在の学習データに対する一個抜き交差検証で精度を求めます。

`LocalSurrogateManager`のコンストラクタにはSurrogateと精度評価器を渡します。 AcquisitionFunctionは `Optimizer.set_acquisition()` で別に設定します。

```python
from saealib import (
    Problem,
    Optimizer,
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
    RBFSurrogate,
    GaussianKernel,
    LocalSurrogateManager,
    MeanPrediction,
    LHSInitializer,
    Termination,
    max_fe,
    LOOAccuracyEvaluator,
)

problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
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
    RBFSurrogate(kernel=GaussianKernel()),
    accuracy_evaluator=LOOAccuracyEvaluator(),
)

initializer = LHSInitializer(
    n_init_archive=5 * DIM, n_init_population=4 * DIM, seed=SEED
)
```

`last_accuracy`は`SurrogateAccuracy`インスタンスで、`.get("spearman")`のように指標名を指定して値を取り出せます。

最初の世代では`last_accuracy`が`None`である点に注意してください。

## StrategySwitcherで評価戦略を切り替える

`StrategySwitcher(primary, fallback, metric="spearman", threshold=0.56)`は、精度が閾値以上なら`primary`を、そうでなければ`fallback`を返します。

次のコード例は、`iterate()` の各ステップで `set_strategy()` を呼び出して構成を変更する手順です。実行環境が変化を検出してplanを再コンパイルし、`switch()` が返すStrategyは次世代から反映されます。 この手順はStage互換経路とgraph-native経路のどちらでも利用できます。

```python
from saealib import PreSelectionStrategy, IndividualBasedStrategy, StrategySwitcher

ps_strategy = PreSelectionStrategy(n_candidates=40, n_select=4)
ib_strategy = IndividualBasedStrategy(evaluation_ratio=0.1)
switcher = StrategySwitcher(primary=ps_strategy, fallback=ib_strategy)

optimizer = (
    Optimizer(problem, seed=SEED)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_acquisition(MeanPrediction())
    .set_strategy(ib_strategy)
    .set_termination(Termination(max_fe(600)))
)

for ctx in optimizer.iterate():
    accuracy = optimizer.surrogate_manager.last_accuracy
    optimizer.set_strategy(switcher.switch(accuracy))

print(ctx.fe)
```

実行すると、序盤は精度が低く`IndividualBasedStrategy`のまま進み、精度が閾値を超えた世代から`PreSelectionStrategy`へ切り替わります。

探索が進んで目的値の差が縮まり精度が再び落ちれば、`IndividualBasedStrategy`へ戻ります。

Component側から再コンパイルを要求する経路については [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md) の「ランタイム差し替えの挙動」を参照してください。

## 別のSwitcherを選ぶ

同じ`switch()`インターフェースを持つ切り替え器が、`StrategySwitcher`以外にも用意されています。

| クラス | 切り替える対象 |
|--------|--------|
| `StrategySwitcher` | 2つの`OptimizationStrategy`（精度が閾値以上か否か） |
| `ManagerSwitcher` | 2つの`SurrogateManager`（精度が閾値以上か否か） |
| `GenCtrlSwitcher` | `GenerationBasedStrategy`の`gen_ctrl`（精度を指数移動平均で平滑化して連続値へ写像） |

`GenCtrlSwitcher`は数値を返すため、`GenerationBasedStrategy`インスタンスの`gen_ctrl`属性に直接割り当てます。次の例でも`iterate()`中に構成を更新します。

```python
from saealib import GenerationBasedStrategy, GenCtrlSwitcher

gen_ctrl_switcher = GenCtrlSwitcher(gm_max=5, gm_min=0)
strategy = GenerationBasedStrategy(gen_ctrl=gen_ctrl_switcher.switch(None))

optimizer.set_strategy(strategy)
for ctx in optimizer.iterate():
    accuracy = optimizer.surrogate_manager.last_accuracy
    strategy.gen_ctrl = gen_ctrl_switcher.switch(accuracy)
```

## 関連するConceptとReference

- {py:class}`saealib.StrategySwitcher` / {py:class}`saealib.ManagerSwitcher` / {py:class}`saealib.GenCtrlSwitcher`
- {py:class}`saealib.LOOAccuracyEvaluator` / {py:class}`saealib.HeldOutAccuracyEvaluator` / {py:class}`saealib.KFoldAccuracyEvaluator`
- {py:class}`saealib.SurrogateAccuracy`
- {py:class}`saealib.Optimizer`

