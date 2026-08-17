---
primary_layer: layer2
related_layers: [layer3, layer4]
---

# Evaluation

## Evaluator and graph-native contracts

The graph-native path is `GenomeBatch → EvaluationAdapter → EvaluationPayload → Evaluator → ObservationBatch`.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Evaluator
   saealib.SerialEvaluator
   saealib.JoblibEvaluator
   saealib.ThreadPoolEvaluator
   saealib.AsyncEvaluator
   saealib.EvaluationAdapter
   saealib.EvaluationRequest
   saealib.EvaluationHandle
   saealib.EvaluationQuery
   saealib.EvaluationUpdate
   saealib.PendingEvaluation
```

## Evaluation policies and plans

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.EvaluationPlanner
   saealib.EvaluationPlan
   saealib.EvaluateAll
   saealib.TopKEvaluation
   saealib.RatioEvaluation
   saealib.RepeatedEvaluation
   saealib.FidelityEvaluation
   saealib.FidelityPromotion
   saealib.ReplicateSummary
   saealib.aggregate_replicates
```

## Stage compatibility type

On the Stage compatibility path, use `Evaluator.evaluate_batch(...) -> EvaluationResult`. At this compatibility boundary, `EvaluationResult` requires shape and row-count consistency for `f`, `g`, and `cv`; when `candidate_ids` is provided, it must be unique and match the result row count. When `Evaluator.submit` returns `candidate_ids`, they must exactly match the request's candidate IDs.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.EvaluationResult
```
