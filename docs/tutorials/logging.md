# 進捗のログ記録

最適化の進捗を、標準の`logging`モジュールで記録します。

## デフォルトの挙動

`Optimizer`は、世代開始のたびに進捗を記録するハンドラ(`logging_generation`)を、`minimize`/`maximize`の`verbose=True`（デフォルト）のときだけ登録します。

ただし、このハンドラは`logging.getLogger(__name__).info(...)`を呼ぶだけなので、Pythonの`logging`モジュール側でINFOレベルの出力を有効にしないかぎり、何も表示されません。

```python
import numpy as np
from saealib import minimize


def expensive_func(x):
    return np.sum(x**2)


DIM = 5

# nothing is printed here since logging.basicConfig has not been called yet
result = minimize(expensive_func, dim=DIM, lb=[-5.0] * DIM, ub=[5.0] * DIM, max_fe=100, seed=0)
```

進捗を表示するには、`logging.basicConfig`でINFOレベルを有効にします。

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

result = minimize(expensive_func, dim=DIM, lb=[-5.0] * DIM, ub=[5.0] * DIM, max_fe=100, seed=0)
# Generation 0 started. fe: 25. Best f: [14.04274116]
# Generation 1 started. fe: 27. Best f: [14.04274116]
# ...
```

`logging_generation`は目的数を見て記録内容を切り替え、単目的では最良の目的値を、多目的では第一非優越フロントのサイズと目的ごとの値の範囲を記録します。

進捗の記録自体が不要な場合は、`verbose=False`を指定してハンドラの登録を止めます。

```python
result = minimize(expensive_func, dim=DIM, lb=[-5.0] * DIM, ub=[5.0] * DIM, max_fe=100, seed=0, verbose=False)
```

## ファイルへの出力

`saealib.callback.handlers`ロガーに`FileHandler`を追加すると、進捗をファイルに書き出せます。

```python
import logging

file_handler = logging.FileHandler("optimization.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

saealib_logger = logging.getLogger("saealib.callback.handlers")
saealib_logger.addHandler(file_handler)
saealib_logger.setLevel(logging.INFO)

result = minimize(expensive_func, dim=DIM, lb=[-5.0] * DIM, ub=[5.0] * DIM, max_fe=100, seed=0)
```

## 多目的でのハイパーボリュームログ

多目的問題では、`logging_generation_hv(reference_point)`が返すハンドラを登録すると、世代ごとのハイパーボリュームを記録できます。

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

`reference_point`は最小化の慣例で、各目的の達成可能な最良値より大きい値を指定します。

## 警告レベルのログ

一部のコンポーネントは、数値的な問題を`logger.warning(...)`で記録します。

例えば`RBFSurrogate`は、カーネル行列が悪条件になったときに警告を出します。

`logging.basicConfig`を呼んでいなくても、WARNING以上のログはPythonの「handler of last resort」によって標準エラー出力に表示されます。

INFOレベルの進捗ログとは異なり、この種の警告は設定なしでも見える点に注意してください。

## 独自のログ処理

`logging_generation`/`logging_generation_hv`が記録する内容以外を記録したい場合は、`CallbackManager`に独自のハンドラを登録します。

詳しい仕組みは[CallbackManager](../components/callbacks.md)を参照してください。

## 参照

- {py:func}`saealib.logging_generation` / {py:func}`saealib.logging_generation_hv`
- {py:class}`saealib.CallbackManager` / {py:class}`saealib.GenerationStartEvent`
- {py:func}`saealib.minimize`
