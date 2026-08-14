---
primary_layer: layer2
related_layers: [layer3]
page_type: reference
---

# Stages

既定の世代ループを構成する公開Stageです。
`Stage` は `OptimizationState` を受け取る互換性用の実行面であり、graph-native Componentの契約とは異なります。
構造化Pipelineへ接続する場合は `stage_component(stage)` を使います。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.CountGenerationStage
   saealib.AskStage
   saealib.AcquisitionStage
   saealib.SurrogatePredictStage
   saealib.EvaluationPlanStage
   saealib.AsyncEvaluationSubmitStage
   saealib.EvaluationSubmitStage
   saealib.EvaluationCollectStage
   saealib.EvaluationApplyStage
   saealib.EvaluationAcknowledgeStage
   saealib.SurrogateFitStage
   saealib.ArchiveUpdateStage
   saealib.FeedbackStage
   saealib.TellStage
   saealib.SurrogateOnlyLoopStage
   saealib.InitializationStage
```
