---
primary_layer: layer2
related_layers: [layer3]
page_type: reference
---

# ステージ

これらの公開Stageは既定の世代ループを構成します。
`Stage`は`OptimizationState`を受け取る互換実行面であり、グラフネイティブなComponentの契約とは異なります。
Stageを構造化Pipelineに接続するには`stage_component(stage)`を使います。

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
