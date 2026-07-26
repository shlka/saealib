# Stages

Concrete `Stage` implementations that make up the default generation-loop pipeline (see [Pipeline](pipeline.md)).

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.CountGenerationStage
   saealib.AskStage
   saealib.SurrogateScoreStage
   saealib.SurrogateFitStage
   saealib.TopKSelectionStage
   saealib.SortByScoreStage
   saealib.TrueEvaluationStage
   saealib.ArchiveUpdateStage
   saealib.TellStage
   saealib.SurrogateOnlyLoopStage
   saealib.InitializationStage
```
