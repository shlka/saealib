---
primary_layer: cross
related_layers: []
page_type: guide
---

# Optimizerを低レベルAPIで組み立てる

前提は、目的関数を `Problem` として表現できることです。
:::{admonition} このページでできるようになること
:class: tip

このページを終えると、既定値では足りないコンポーネントを `Optimizer` に登録し、`run()` または `iterate()` で実行できます。
:::

高レベルAPIの一回実行で足りない場合に低レベルAPIへ移ります。
ビルトインコンポーネントの個別選択だけなら [ビルトインコンポーネントの差し替え](component_swap.md) を先に使い、世代ごとの状態取得や実行中の交換が必要ならこのページの構成にします。
importの方針は [Canonical Imports](../api/imports.md) にまとめています。

## Optimizerを構成する

`Optimizer(problem)` を作り、`set_*()` をチェーンして実行コンポーネントを設定します。
`set_*()` は連結可能な設定APIであり、設定だけを行います。
`run()` または `iterate()` が既定値解決、実行計画の構築、構成の検証を行ってから実行を始めます。

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

optimizer = (
    Optimizer(problem, seed=0)
    .set_termination(Termination(max_fe(100)))
)
```

必要なコンポーネントだけを `set_initializer()`、`set_algorithm()`、`set_surrogate_manager()`、`set_acquisition()`、`set_strategy()`、`set_evaluator()` などで追加します。
`set_*()` は `Optimizer` 自身を返すため、設定を連結できます。

## runとiterateを使い分ける

`run()` は終了まで実行し、最後の `OptimizationState` を返します。
`iterate()` は世代ごとの状態を生成するため、履歴の記録や条件付きのコンポーネント交換をループの中で行えます。

```{mermaid}
flowchart LR
    A[Optimizerを構成] --> B[runまたはiterate]
    B --> C[既定値解決、計画構築、検証]
    C --> D[世代を実行]
    D --> E{終了条件}
    E -- いいえ --> D
    E -- はい --> F[OptimizationState]
```

```python
ctx = optimizer.run()
print(ctx.fe, ctx.gen)
```

世代ごとの状態が必要な場合は `run()` の代わりに `iterate()` を使います。

```python
history = []
for ctx in optimizer.iterate():
    values = ctx.archive.get_array("f")[:, 0]
    history.append((ctx.gen, float(values.min())))

print(history[-1])
```

`ctx.archive` は評価済み解を保持し、`ctx.fe` と `ctx.gen` は評価回数と完了世代を示します。
途中でStrategyやSurrogateManagerを交換する場合は、各世代の `ctx` を読んで `optimizer.set_*()` を呼び出します。
checkpointの保存と再開は [Checkpointing](checkpoint.md) に委ねます。

## 関連するConceptとReference

- [Stage](../concepts/observation_and_state/stage.md)
- [OptimizationState](../concepts/observation_and_state/optimization_state.md)
- [Optimizerリファレンス](../api/optimizer.md)
- [Terminationリファレンス](../api/termination.md)
