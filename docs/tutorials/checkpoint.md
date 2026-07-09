# 再現性とチェックポイント

このチュートリアルでは，乱数シードによる再現性の確保と，長時間最適化の中断・再開（チェックポイント）の方法を説明します．

## シードによる再現性

`Optimizer` にシードを渡すと，初期個体生成から各世代のランダム操作まで，すべての乱数を一元管理します．

```python
import numpy as np
from saealib import (
    GA, CrossoverBLXAlpha, IndividualBasedStrategy, LHSInitializer,
    MutationUniform, Optimizer, RBFSurrogate, SequentialSelection,
    Termination, TruncationSelection, gaussian_kernel, max_fe,
)
from saealib.comparators import SingleObjectiveComparator
from saealib.problem import Problem

def sphere(x):
    return np.array([np.sum(x ** 2)])

problem = Problem(
    func=sphere,
    dim=5,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0] * 5,
    ub=[5.0] * 5,
    comparator=SingleObjectiveComparator(),
)

def make_optimizer(seed=None):
    return (
        Optimizer(problem, seed=seed)         # seed を Optimizer に渡す
        .set_initializer(LHSInitializer(20, 10))
        .set_algorithm(GA(
            crossover=CrossoverBLXAlpha(crossover_rate=0.9, alpha=0.5),
            mutation=MutationUniform(mutation_rate=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        ))
        .set_surrogate(RBFSurrogate(gaussian_kernel, 5), n_neighbors=10)
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.5))
        .set_termination(Termination(max_fe(200)))
    )

# 同じシードで2回実行 → 結果は完全一致
ctx1 = make_optimizer(seed=42).run()
ctx2 = make_optimizer(seed=42).run()

assert np.array_equal(ctx1.archive.x, ctx2.archive.x)  # 一致
```

`seed=None`（デフォルト）では非決定的に動作します．

---

## チェックポイントの保存と再開

### npz 形式（ポータブル）

`ctx.save(path)` でアーカイブ・個体群・乱数状態を npz ファイルに保存します．再開は `OptimizationContext.load` でコンテキストを復元し，`optimizer.run_from` に渡します．

```python
from saealib.context import OptimizationContext

# 前半を実行
opt_first = make_optimizer(seed=42)
opt_first.set_termination(Termination(max_fe(100)))  # 100 FE まで
ctx_mid = opt_first.run()

# チェックポイントを保存
ctx_mid.save("checkpoint.npz")
print(f"saved: gen={ctx_mid.gen}, fe={ctx_mid.fe}")

# ─────── ここで中断・再起動 ───────

# コンテキストを復元
ctx_loaded = OptimizationContext.load("checkpoint.npz", problem)
print(f"resumed={ctx_loaded.resumed}")  # True

# 最終目標まで再開
opt_resume = make_optimizer(seed=42)   # 同じ構成で再作成
opt_resume.set_termination(Termination(max_fe(200)))  # 最終目標を設定
ctx_final = opt_resume.run_from(ctx_loaded)

print(f"final: gen={ctx_final.gen}, fe={ctx_final.fe}")
```

> **再現性について**  
> `ctx.rng` の内部状態ごと保存するため，同一バージョン・同一環境では中断なし実行と完全一致します．バージョンをまたぐ場合の一致は保証されません．

---

### pickle 形式（より完全な復元）

pickle を使うと，サロゲートの学習済みパラメータなど，Optimizer の全コンポーネントをそのまま保存・復元できます．

```python
import warnings

# Optimizer ごと保存
with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # バージョン依存の警告を抑制
    opt_mid.save_pickle(ctx_mid, "checkpoint.pkl")

# Optimizer ごと復元
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    opt_resume, ctx_loaded = Optimizer.load_pickle("checkpoint.pkl")

opt_resume.set_termination(Termination(max_fe(200)))
ctx_final = opt_resume.run_from(ctx_loaded)
```

> **制約**  
> pickle ファイルは Python・ライブラリのバージョンに依存します．lambda 関数を `Problem.func` に使用している場合は pickle できません（モジュールレベルの関数を使用してください）．

---

## 自動チェックポイント

`run()` / `iterate()` の `checkpoint_path` パラメータで，N 世代ごとに自動保存できます．

```python
ctx = opt.run(
    checkpoint_path="checkpoints/",   # 保存先ディレクトリ
    checkpoint_interval=5,             # 5 世代ごとに保存
    checkpoint_format="npz",           # "npz" | "pickle" | "both"
    checkpoint_delete_on_success=True, # 正常終了後に削除
)
```

ファイルは `checkpoints/checkpoint_000005.npz`, `checkpoints/checkpoint_000010.npz`, … のように命名されます．

### CheckpointCallback を直接使う

より細かい制御が必要な場合は `CheckpointCallback` を直接登録します．

```python
from saealib import CheckpointCallback

cb = CheckpointCallback(
    path="checkpoints/",
    interval=5,
    format="both",             # npz と pkl を両方保存
    delete_on_success=False,
    optimizer=opt,             # pickle 形式に必要
)
cb.register(opt.cbmanager)

ctx = opt.run()
```

---

## resumed フラグ

チェックポイントから復元したコンテキストには `resumed=True` がセットされます．コールバックや戦略でこのフラグを参照できます．

```python
from saealib.callback import RunStartEvent

def on_start(event):
    if event.ctx.resumed:
        print("チェックポイントから再開しました")
    else:
        print("新規実行を開始します")

opt.cbmanager.register(RunStartEvent, on_start)
```
