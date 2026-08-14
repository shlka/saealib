---
primary_layer: layer4
related_layers: [layer3]
page_type: reference
---

# Execution API

実行APIは、評価器、初期化器、非同期評価スケジューラー、ランタイムの登録機構を公開します。
`Runner` は内部実装であるため、この公開一覧には含めません。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.execution.AsyncEvaluationScheduler
   saealib.execution.PollResult
   saealib.execution.RuntimeFactory
   saealib.execution.RuntimeRegistration
   saealib.execution.RuntimeRegistry
   saealib.execution.create_runtime
   saealib.execution.default_runtime_registry
```

評価器、評価Request、初期化器の詳細は [Evaluation](evaluation.md) と [Initialization](initialization.md) を参照してください。
