---
primary_layer: layer2
related_layers: [layer3]
page_type: reference
---

# Stages

These public Stages compose the default generation loop.
`Stage` is a compatibility execution surface that receives `OptimizationState`, distinct from a graph-native Component's contract.
Use `stage_component(stage)` to connect a Stage to a structured Pipeline.

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
