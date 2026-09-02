---
primary_layer: layer4
related_layers: [layer3]
page_type: reference
---

# 探索空間API

SearchSpace APIは、候補表現と空間固有サービスの公開ファサードです。空間の実装モジュールではなく`saealib.space`の名前を使います。

## 空間、サービス、アダプター

### 探索空間

`GenomeBatch`表現を提供する探索空間です。

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
```

### SearchSpaceのサービス

サンプリング・検証などの探索空間のサービスは、空間が提供する能力です。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.GenomeCodec
   saealib.space.SamplingService
   saealib.space.ValidationService
```

### アダプター

`EvaluationAdapter`は評価境界に置かれ、`GenomeBatch`を`EvaluationPayload`に変換するアダプターです。`FeatureEncoder`は`saealib.space.FeatureEncoder`として公開されるアダプターのサブタイプで、`GenomeBatch`を`FeatureBatch`に変換し、サロゲートが学習できる特徴量を決めます。`SamplingService`などの空間能力とは種類が異なります。現在の実装では`SurrogateManager`契約が`ServiceRequirement("FeatureEncoder")`を宣言し、`VectorSpace`が既定のエンコーダーをサービスとして登録するため、数値ベクトル空間では追加設定なしに解決されます。`ObjectSpace`、`PermutationSpace`、`SequenceSpace`では、利用者が`FeatureEncoder`を提供しない限り解決に失敗します。サロゲートに渡す入力は利用者が決めます。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.FeatureEncoder
```

