---
primary_layer: cross
related_layers: [layer1, layer2, layer3, layer4]
page_type: entry
---

# APIリファレンス

APIリファレンスは、公開ファサードを中心に構成します。
実装モジュールの深いパスはcanonical importとして案内しません。

## APIの公開層

import経路を選ぶときは、次の公開層を使います。

- **ルートの便利API**：日常的に使う安定したAPIです。

  ```python
  from saealib import minimize, Problem, GA
  ```

- **公開名前空間**：分野ごとにまとめた公開コンポーネントです。

  ```python
  from saealib.surrogate import RBFSurrogate
  from saealib.operators import MutationPolynomial
  ```

探索空間、実行、Feedbackの名前は、それぞれ `saealib.space`、`saealib.execution`、`saealib.policies` から取得します。

- **フレームワーク拡張API**：拡張に使う契約と構成要素です。

  ```python
  from saealib.core import Component, ComponentGraph, ComponentContract
  ```

- 実装モジュールは、公開ファサードの代わりに使うものではありません。

名前空間の公開名は、自動的にルートへ再公開されません。
既存のルートimportを維持することと、新しい名前をルートへ追加することは別の判断です。

新しいルート公開名は、広く使われ、安定性が見込まれ、一般的なimportを明確に改善する場合に限ります。
それ以外の名前は、対応する公開名前空間またはフレームワーク拡張APIに置きます。

```{toctree}
:maxdepth: 2

imports
core
execution
space
feedback
highlevel
optimizer
exceptions
registry
problem
variables
comparators
decomposition
population
algorithms
operators
surrogate
acquisition
strategies
initialization
evaluation
termination
callbacks
pipeline
stages
utils
../references
```
