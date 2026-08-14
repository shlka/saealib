---
primary_layer: layer4
related_layers: [layer3]
page_type: reference
---

# Core API

コアAPIは、コンポーネント契約、グラフ、状態、コンパイル済み計画の公開ファサードです。
通常の拡張では `saealib.core` から名前を取得し、実装モジュールのパスを直接参照しません。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.core.Component
   saealib.core.ComponentContract
   saealib.core.PartSpec
   saealib.core.AssumptionSet
   saealib.core.DataSpec
   saealib.core.PortContract
   saealib.core.PortSpec
   saealib.core.StateContract
   saealib.core.LifecycleContract
   saealib.core.ExecutionContract
   saealib.core.ComponentGraph
   saealib.core.StructuredGraph
   saealib.core.StructuredRegion
   saealib.core.CompilationRule
   saealib.core.StateStore
   saealib.core.StateView
   saealib.core.StatePatch
   saealib.core.ExecutablePlan
   saealib.core.ExecutionRuntime
```

`Compiler` の公開 import 方針は現在整理中です。
そのため、このページでは `saealib.core` と `saealib.core.compiler` のどちらかを標準のimport pathとして断定しません。
コンパイラを使うコードでは、対象リリースの公開APIとリリースノートを確認してください。

Graphの構成値は `saealib.core.compiler` から参照できます。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.core.compiler.ComponentNode
   saealib.core.compiler.NodeRef
   saealib.core.compiler.DataEdge
   saealib.core.compiler.ControlEdge
   saealib.core.compiler.StateBinding
   saealib.core.compiler.Diagnostic
   saealib.core.compiler.ContractPath
```
