---
primary_layer: layer2
---

# Callbacks

## Events

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

## Callback Manager

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.CallbackManager
```

## Built-in Handlers

```{eval-rst}
.. autofunction:: saealib.logging_generation
```

```{eval-rst}
.. autofunction:: saealib.logging_generation_hv
```

## Checkpointing

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.CheckpointCallback
```
