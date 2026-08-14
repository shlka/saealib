---
primary_layer: layer4
related_layers: [layer3]
page_type: reference
---

# 探索空間API

探索空間APIは、候補表現と空間固有サービスの公開ファサードです。
空間の実装モジュールではなく `saealib.space` の名前を利用します。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.SearchSpace
   saealib.space.ServiceRegistry
   saealib.space.ValidationResult
   saealib.core.contracts.RepresentationSpec
   saealib.space.VectorSpace
   saealib.space.ObjectSpace
   saealib.space.SequenceSpace
   saealib.space.PermutationSpace
   saealib.space.GenomeCodec
   saealib.space.FeatureEncoder
   saealib.space.SamplingService
   saealib.space.ValidationService
```
