---
primary_layer: layer2
---

# コールバック

## イベント

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Event
   saealib.RunStartEvent
   saealib.RunEndEvent
   saealib.InitialEvaluationStartEvent
   saealib.InitialEvaluationEndEvent
   saealib.GenerationStartEvent
   saealib.GenerationEndEvent
   saealib.SurrogateStartEvent
   saealib.SurrogateEndEvent
   saealib.AcquisitionStartEvent
   saealib.AcquisitionEndEvent
   saealib.PostCrossoverEvent
   saealib.PostMutationEvent
   saealib.PostAskEvent
   saealib.PostSurrogateFitEvent
   saealib.PostEvaluationEvent
```

## コールバックマネージャー

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.CallbackManager
```

## 組み込みハンドラー

```{eval-rst}
.. autofunction:: saealib.logging_generation
```

```{eval-rst}
.. autofunction:: saealib.logging_generation_hv
```

## チェックポイント保存

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.CheckpointCallback
```

