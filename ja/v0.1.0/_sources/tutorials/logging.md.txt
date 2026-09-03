---
primary_layer: layer2
page_type: guide
---

# 進捗のログ記録

標準の `logging` モジュールで最適化の進捗を記録します。

:::{admonition} このページでできるようになること
:class: tip

このページを終えると、Python標準の `logging` モジュールで進捗、警告、独自イベントを記録できるようになります。
:::

基本的な最適化の実行は [高レベルAPI](highlevel_api.md) を使い、世代単位の構成まで調整する場合は [低レベルAPI](lowlevel_api.md) を参照してください。
このページでは、標準loggingの出力先とログレベルを設定し、組み込み・独自のイベントを記録する方法を扱います。

## 既定のログ動作を使う

`Optimizer` は各世代の開始時に進捗を記録するハンドラ（`logging_generation`）を登録しますが、`minimize`/`maximize` の `verbose=True`（既定値）の場合に限られます。

ただし、このハンドラは `logging.getLogger(__name__).info(...)` を呼ぶだけなので、Pythonの `logging` モジュールでINFOレベルの出力を有効にしない限り何も表示されません。

```python
import numpy as np
from saealib import minimize


def expensive_func(x):
    return np.sum(x**2)


DIM = 5

# nothing is printed here since logging.basicConfig has not been called yet
result = minimize(
    expensive_func, dim=DIM, lb=[-5.0] * DIM, ub=[5.0] * DIM, max_fe=100, seed=0
)
```

進捗を表示するには、`logging.basicConfig` でINFOレベルを有効にします。

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

result = minimize(
    expensive_func, dim=DIM, lb=[-5.0] * DIM, ub=[5.0] * DIM, max_fe=100, seed=0
)
# Generation 0 started. fe: 25. Best f: [14.04274116]
# Generation 1 started. fe: 27. Best f: [14.04274116]
# ...
```

`logging_generation` は目的数に応じて記録内容を切り替えます。単目的では最良の目的値を記録し、多目的では最初の非劣フロントのサイズと目的ごとの値の範囲を記録します。

進捗をまったく記録しない場合は、`verbose=False` を指定してハンドラの登録を停止します。

```python
result = minimize(
    expensive_func,
    dim=DIM,
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
    max_fe=100,
    seed=0,
    verbose=False,
)
```

## 進捗をファイルへ書き込む

`saealib`ロガーに`FileHandler`を追加すると、進捗をファイルへ書き出せます。

```python
import logging

file_handler = logging.FileHandler("optimization.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

saealib_logger = logging.getLogger("saealib")
saealib_logger.addHandler(file_handler)
saealib_logger.setLevel(logging.INFO)

result = minimize(
    expensive_func, dim=DIM, lb=[-5.0] * DIM, ub=[5.0] * DIM, max_fe=100, seed=0
)
```

## 多目的問題のハイパーボリュームを記録する

多目的問題では、`logging_generation_hv(reference_point)` が返すハンドラを登録すると、各世代のハイパーボリュームを記録します。

```python
import numpy as np
from saealib import (
    Optimizer,
    Termination,
    max_fe,
    GenerationStartEvent,
    logging_generation,
    logging_generation_hv,
)
from saealib.benchmarks import zdt1

problem = zdt1(n_var=5)
optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(200)))

# remove the default logging_generation and swap in the HV-based one
optimizer.cbmanager.unregister(GenerationStartEvent, logging_generation)
optimizer.cbmanager.register(
    GenerationStartEvent, logging_generation_hv(reference_point=np.array([1.1, 1.1]))
)

ctx = optimizer.run()
# Generation 0. fe: 25. HV: 0.612345
# ...
```

最小化の規約では、`reference_point` は各目的で達成可能な最良値より大きい値にします。

## warningレベルのログを扱う

一部のコンポーネントは `logger.warning(...)` で数値上の問題を記録します。

たとえば `RBFSurrogate` は、カーネル行列の条件が悪くなったときに警告を発します。

saealibは独自の出力ハンドラを構成しません。出力を生成しない`NullHandler`だけをインストールします。

警告などの`saealib`ログを表示したい場合は、Pythonの`logging`モジュールを自分で構成します。たとえばWARNINGレベルのログを有効にすると、`RBFSurrogate`によるカーネル行列の悪条件警告が表示されます。

```python
import logging
import numpy as np
from saealib import GaussianKernel, RBFSurrogate

logging.basicConfig(level=logging.WARNING)

surrogate = RBFSurrogate(kernel=GaussianKernel(length_scale=1.0), solver="solve")
surrogate.fit(np.array([[0.0], [0.0], [0.0]]), np.array([1.0, 2.0, 3.0]))
```

## 独自のログ記録を追加する

`logging_generation`/`logging_generation_hv` と異なる内容を記録したい場合は、`CallbackManager` に独自のハンドラを登録します。

仕組みの詳細は [CallbackManager](../concepts/observation_and_state/callbacks.md) を参照してください。

## 関連コンセプトと参考情報

- {py:func}`saealib.logging_generation` / {py:func}`saealib.logging_generation_hv`
- {py:class}`saealib.CallbackManager` / {py:class}`saealib.GenerationStartEvent`
- {py:func}`saealib.minimize`
- [診断と観測](../concepts/observation_and_state/diagnostics.md)
