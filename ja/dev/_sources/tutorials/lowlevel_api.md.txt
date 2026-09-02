---
primary_layer: layer2
page_type: guide
---

# Optimizerを低レベルAPIで組み立てる

前提は、目的関数を `Problem` として表現できることです。

:::{admonition} このページでできるようになること
:class: tip

このページを終えると、既定値では足りないコンポーネントを `Optimizer` に登録し、`run()` または `iterate()` で実行できます。
:::

高レベルAPIの一回実行で足りない場合に低レベルAPIへ移ります。 ビルトインコンポーネントの個別選択だけなら [ビルトインコンポーネントの差し替え](component_swap.md) を先に使い、世代ごとの状態取得や実行中の交換が必要ならこのページの構成にします。 importの方針は [Canonical Imports](../api/imports.md) にまとめています。

## Optimizerを構成する

`Optimizer(problem)` を作り、`set_*()` をチェーンして実行コンポーネントを設定します。 `set_*()` は連結可能な設定APIであり、設定だけを行います。 `run()` または `iterate()` が既定値解決、実行計画の構築、構成の検証を行ってから実行を始めます。

```python
import numpy as np
from saealib import Optimizer, Problem, Termination, max_fe


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

optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(100)))
```

必要なコンポーネントだけを `set_initializer()`、`set_algorithm()`、`set_surrogate_manager()`、`set_acquisition()`、`set_strategy()`、`set_evaluator()` などで追加します。 `set_*()` は `Optimizer` 自身を返すため、設定を連結できます。

## サロゲート込みで完全に構成する

`minimize` は各コンポーネントを既定の組み合わせで結線しますが、個別のパラメータは調整できません。 コンポーネントを個別に生成して `Optimizer` へ組み立てると、この制約がなくなります。

`LocalSurrogateManager` にはSurrogateと学習データの作り方を渡し、AcquisitionFunctionは `Optimizer.set_acquisition()` で別に設定します。 この二つを同じコンストラクタ引数として扱わないでください。

```python
import numpy as np

from saealib import (
    GA,
    IndividualBasedStrategy,
    LHSInitializer,
    Optimizer,
    Problem,
    Termination,
    max_fe,
)
from saealib.acquisition import MeanPrediction
from saealib.operators import (
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.surrogate import GaussianKernel, LocalSurrogateManager, RBFSurrogate

DIM = 10
SEED = 0

problem = Problem(
    objective,
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
    RBFSurrogate(kernel=GaussianKernel()),
)
strategy = IndividualBasedStrategy(evaluation_ratio=0.1)
initializer = LHSInitializer(
    n_init_archive=5 * DIM,
    n_init_population=4 * DIM,
    seed=SEED,
)
termination = Termination(max_fe(500))

optimizer = (
    Optimizer(problem, seed=SEED)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_acquisition(MeanPrediction())
    .set_strategy(strategy)
    .set_termination(termination)
)

ctx = optimizer.run()
archive_x = ctx.archive.get_array("x")
archive_f = ctx.archive.get_array("f")[:, 0]
best_idx = int(np.argmin(archive_f))
print("Best solution:", archive_x[best_idx])
print("Objective value:", archive_f[best_idx])
print("Evaluations:", ctx.fe)
```

乱数シードは、`Optimizer(problem, seed=SEED)` と `LHSInitializer(..., seed=SEED)` の両方へ同じ値を渡します。

`Optimizer` の `seed` が既定の `LHSInitializer` へ自動的に伝播するのは、`set_initializer()` を自分で呼ばない場合（`minimize`/`maximize` 経由など）だけです。

`Initializer` を自分で組み立てる場合は、明示的に渡す必要があります。

## 終了条件を組み合わせる

`Termination` は複数の条件を受け取ります。

列挙した条件のいずれかが満たされた時点で実行を終了します（論理OR）。

```python
from saealib import Termination, max_fe, max_gen

termination = Termination(max_fe(500), max_gen(200))
```

lambdaで独自の条件を追加することもできます。

```python
termination = Termination(
    max_fe(500),
    lambda ctx: ctx.archive.get_array("f")[:, 0].min() < 1e-4,
)
```

## runとiterateを使い分ける

`run()` は終了まで実行し、最後の `OptimizationState` を返します。 `iterate()` は世代ごとの状態を生成するため、履歴の記録や条件付きのコンポーネント交換、独自の早期打ち切りをループの中で行えます。

```{mermaid}
flowchart LR
    A["Configure the Optimizer"] --> B["run or iterate"]
    B --> C["Resolve defaults, build plan, validate"]
    C --> D["Run a generation"]
    D --> E{"Termination met?"}
    E -- No --> D
    E -- Yes --> F["OptimizationState"]
```

```python
ctx = optimizer.run()
print(ctx.fe, ctx.gen)
```

世代ごとの状態が必要な場合は `run()` の代わりに `iterate()` を使います。

```python
history = []
for ctx in optimizer.iterate():
    best_f = ctx.archive.get_array("f")[:, 0].min()
    history.append((ctx.gen, ctx.fe, float(best_f)))
    print(f"gen={ctx.gen:4d}  fe={ctx.fe:4d}  best_f={best_f:.6f}")

print("Evaluations:", ctx.fe)
```

`ctx.archive` は評価済み解を保持し、`ctx.fe` と `ctx.gen` は評価回数と完了世代を示します。 途中でStrategyやSurrogateManagerを交換する場合は、各世代の `ctx` を読んで `optimizer.set_*()` を呼び出します。実行環境が変化を検出してplanを再コンパイルし、次世代から反映します。 この手順はStage互換経路とgraph-native経路のどちらでも利用できます。 Component側から再コンパイルを要求する経路については [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md) の「Behavior of runtime swapping」を参照してください。 checkpointの保存と再開は [Checkpointing](checkpoint.md) に委ねます。

## 関連するConceptとReference

- [Stage](../concepts/observation_and_state/stage.md)
- [OptimizationState](../concepts/observation_and_state/optimization_state.md)
- [Optimizerリファレンス](../api/optimizer.md)
- [Terminationリファレンス](../api/termination.md)
- {py:class}`saealib.LocalSurrogateManager`
- {py:class}`saealib.RBFSurrogate`
- {py:class}`saealib.MeanPrediction`
- {py:class}`saealib.LHSInitializer`
- {py:class}`saealib.IndividualBasedStrategy`
- {py:class}`saealib.Termination` / {py:func}`saealib.max_fe` / {py:func}`saealib.max_gen`
