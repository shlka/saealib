---
primary_layer: layer4
related_layers: [layer2, layer3]
page_type: reference
---

# フィードバックAPI

フィードバックAPIは、候補への提案、観測、フィードバック方針を公開します。 提案と観測の契約型は `saealib.core.contracts` の公開語彙に属し、ビルダーと方針は `saealib.policies` から利用します。

## グラフネイティブ経路の契約

`ProposalBatch`、`ObservationBatch`、`FeedbackBatch`、`FeedbackContract`はグラフネイティブ経路の契約です。対象、提案関係、順序、状態、出所、完了の意味論、および利用側が宣言する順序と完了を扱います。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.core.contracts.ProposalBatch
   saealib.core.contracts.ObservationBatch
   saealib.core.contracts.FeedbackBatch
   saealib.core.contracts.FeedbackContract
```

## 方針・ビルダー

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

## Stage互換型

`FeedbackResult` はStage互換経路で使う密な互換データ型です。配列の行整合性と候補IDの一意性はこの型の境界に属します。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.FeedbackResult
```

