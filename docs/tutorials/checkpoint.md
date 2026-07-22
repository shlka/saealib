# 再現性とチェックポイント

長時間実行する最適化を再現可能にし、途中から再開できるようにします。

チェックポイント機能は低レベルAPIの`Optimizer`でのみ使えます。

`Optimizer`の組み立て方は[単目的最適化](single_objective.md)の低レベルAPI節を参照してください。

## 乱数シードによる再現性

`Optimizer(problem, seed=...)`に同じ`seed`を渡すと、乱数を使う処理が同じ手順で初期化され、同一の結果が得られます。

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


def expensive_func(x):
    return np.sum(x**2)


DIM = 10
SEED = 0

problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
)


def build_optimizer(max_fe_value):
    return (
        Optimizer(problem, seed=SEED)
        .set_initializer(
            LHSInitializer(n_init_archive=5 * DIM, n_init_population=4 * DIM, seed=SEED)
        )
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(0.7, 0.4),
                mutation=MutationUniform(0.3),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate_manager(
            LocalSurrogateManager(RBFSurrogate(gaussian_kernel, dim=DIM), MeanPrediction())
        )
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.1))
        .set_termination(Termination(max_fe(max_fe_value)))
    )


ctx1 = build_optimizer(300).run()
ctx2 = build_optimizer(300).run()

print(np.allclose(ctx1.archive.get_array("f"), ctx2.archive.get_array("f")))  # True
```

`build_optimizer`は、以降の節でも同じコンポーネント構成のまま`Optimizer`を作り直すために使います。

## チェックポイントの保存と再開

`run()`が返す`ctx`は、`ctx.save(path)`でnpz形式の単一ファイルに保存できます。

```python
ctx = build_optimizer(200).run()
ctx.save("checkpoint.npz")
```

保存したチェックポイントは`OptimizationState.load(path, problem)`で読み込み、`Optimizer.run_from(ctx)`に渡すと続きから再開できます。

```python
from saealib.context import OptimizationState

loaded_ctx = OptimizationState.load("checkpoint.npz", problem)

resumed_ctx = build_optimizer(600).run_from(loaded_ctx)
print(resumed_ctx.fe)             # includes the evaluations from before saving
print(resumed_ctx.data["resumed"])  # True
```

`ctx.data["resumed"]`は、`run_from()`で再開したコンテキストにだけ`True`が設定されるフラグです。

`RunStartEvent`などのコールバックからは、`event.ctx.data["resumed"]`として参照できます。

## 自動チェックポイント

`run()`/`iterate()`に`checkpoint_path`を渡すと、`checkpoint_interval`世代ごとに自動保存されます。

`checkpoint_path`は単一ファイルではなくディレクトリとして扱われ、`checkpoint_{gen:06d}.npz`という名前で世代ごとのスナップショットが作られます。

```python
ctx = build_optimizer(300).run(checkpoint_path="checkpoints", checkpoint_interval=5)
```

再開するときは、ディレクトリ内の最新のスナップショットを読み込みます。

```python
from pathlib import Path

latest = sorted(Path("checkpoints").glob("checkpoint_*.npz"))[-1]
loaded_ctx = OptimizationState.load(latest, problem)
```

正常終了後にスナップショットを残したくない場合は、`checkpoint_delete_on_success=True`を指定します（ディレクトリ自体は残り、中のファイルだけが削除されます）。

```python
ctx = build_optimizer(300).run(
    checkpoint_path="checkpoints",
    checkpoint_interval=5,
    checkpoint_delete_on_success=True,
)
```

## pickle形式での保存

npzは`ctx`のみを保存しますが、pickle形式では学習済みのサロゲートパラメータを含めて`Optimizer`ごと保存できます。

```python
optimizer = build_optimizer(200)
ctx = optimizer.run()
optimizer.save_pickle(ctx, "checkpoint.pkl")

loaded_optimizer, loaded_ctx = Optimizer.load_pickle("checkpoint.pkl")
```

実行時にPythonやライブラリのバージョンに関する`UserWarning`が出ることがあります。

`Termination`にlambdaを使うなど、標準の`pickle`で直列化できないオブジェクトを含む`Optimizer`はpickle保存できません。

## CheckpointCallbackを直接使う

`run()`の`checkpoint_path`は、内部で`CheckpointCallback`を登録しているだけです。

同じ処理を明示的に組み込むには、`CheckpointCallback`を`cbmanager`に登録します。

```python
from saealib import CheckpointCallback

optimizer = build_optimizer(300)
callback = CheckpointCallback("checkpoints", interval=5, optimizer=optimizer)
callback.register(optimizer.cbmanager)

ctx = optimizer.run()
```

`format="pickle"`または`format="both"`を指定する場合は、`optimizer`引数が必須です。

## 参照

- {py:class}`saealib.Optimizer`
- {py:class}`saealib.CheckpointCallback`
- {py:func}`saealib.minimize`
