---
primary_layer: layer4
related_layers: [layer2, layer3]
page_type: reference
---

# Feedback API

The Feedback API exposes candidate proposals, observations, and Feedback policies.
Proposal and observation contract types belong to the public vocabulary of `saealib.core.contracts`, while builders and policies are available from `saealib.policies`.

## Contracts on the graph-native path

`ProposalBatch`, `ObservationBatch`, `FeedbackBatch`, and `FeedbackContract` are contracts on the graph-native path. They handle subject, proposal relation, sequence, status, source, completion semantics, and the consumer's ordering and completion declarations.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.core.contracts.ProposalBatch
   saealib.core.contracts.ObservationBatch
   saealib.core.contracts.FeedbackBatch
   saealib.core.contracts.FeedbackContract
```

## Policies and builders

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.FeedbackBuilder
   saealib.TrueOnlyFeedback
   saealib.PredictedFeedback
   saealib.MixedFeedback
   saealib.NoFeedback
```

## Stage compatibility type

`FeedbackResult` is the dense compatibility data type used on the Stage compatibility path. Array row consistency and candidate-ID uniqueness belong to this type's boundary.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.FeedbackResult
```
