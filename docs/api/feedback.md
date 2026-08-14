---
primary_layer: layer4
related_layers: [layer3]
page_type: reference
---

# Feedback API

フィードバックAPIは、候補への提案、観測、フィードバック方針を公開します。
提案と観測の契約型は `saealib.core.contracts` の公開語彙に属し、ビルダーと方針は `saealib.policies` から利用します。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.core.contracts.ProposalBatch
   saealib.core.contracts.ObservationBatch
   saealib.core.contracts.FeedbackBatch
   saealib.core.contracts.FeedbackContract
   saealib.FeedbackBuilder
   saealib.FeedbackResult
   saealib.TrueOnlyFeedback
   saealib.PredictedFeedback
   saealib.MixedFeedback
   saealib.NoFeedback
```
