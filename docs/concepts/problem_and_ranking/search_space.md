---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# 探索空間（SearchSpace）

`SearchSpace` は候補の表現と、その表現を扱うためのサービスを定義します。
アルゴリズムや代理モデルは空間の内部表現を直接変更せず、空間が提供する `GenomeBatch` とサービスを利用します。

## SearchSpaceとGenomeBatch

`SearchSpace` は `representation`、`services`、`sample()`、`validate()` を提供します。
`GenomeBatch` は候補を行方向にまとめた値であり、候補IDや目的値を直接所有するデータ構造ではありません。
`validate()` は各行の妥当性とバッチ全体のエラーを `ValidationResult` で返します。

`ServiceRegistry` は空間が提供するサービスを名前で登録します。
サンプリング、検証、複製、同値性、距離、特徴量エンコード、ゲノムのシリアライズなどの処理は、必要な空間が対応するサービスとして公開します。

## ビルトイン探索空間

組み込み空間は表現の制約に応じて次のように分かれます。

- **`VectorSpace`**：固定幅の密な数値ベクトルを扱います。
- **`ObjectSpace`**：任意のオブジェクト表現を保持する最小の空間です。
- **`SequenceSpace`**：長さが変化する系列を扱います。
- **`PermutationSpace`**：順列制約を満たす表現を扱います。

各空間は共通の `SearchSpace` 契約を満たしますが、サービスの集合と `RepresentationSpec` の内容は異なります。
たとえば `ObjectSpace` は既定のサービスを登録しないため、アルゴリズムが要求するサービスを別途提供できる構成にする必要があります。

## 表現とサービス

`RepresentationSpec` はパラメータの種類、形状、表現種別を記述します。
表現の記述は、候補をどのように保存するかを示すものであり、サンプリングや特徴量化の実装そのものではありません。

サービスはこの差を埋める実行可能な境界です。
代理モデルが数値特徴を要求する場合、空間の `FeatureEncoder` が `GenomeBatch` を特徴行列へ変換し、コンパイラは要求されたサービスの有無と互換性を検査します。

## 責務と責務外

SearchSpaceの責務は、候補の表現、候補を生成するサービス、表現の妥当性を所有することです。
候補の目的値、観測の出所、Componentの状態、Graphの実行順序はSearchSpaceが所有しません。
`GenomeBatch`は空間が所有する候補データであり、候補IDやFeedback結果を暗黙に追加する値ではありません。

## 不変条件と生成時点

RepresentationSpecとサービスの集合は空間の生成時に確定し、CompilerはComponentのServiceRequirementと照合します。
`sample()`が返すGenomeBatchは空間のRepresentationSpecに適合し、`validate()`は不適合を黙って修復せずValidationResultで報告します。
サービスの実装状態はSearchSpaceが所有し、Componentは登録されたサービスを要求して利用します。

## 拡張と失敗

新しい候補表現を追加する場合はRepresentationSpec、サンプリング、検証、必要なサービスを一つのSearchSpace境界で定義します。
RepresentationSpecをComponentContractへ移す設計にはせず、ポートのDataSpecやrequired serviceとの接続をCompilerで宣言します。
kindや形状の不一致、必要サービスの不足、無効なGenomeBatchはCompilerまたはValidationResultのDiagnosticsになります。

候補表現の拡張は[フレームワーク拡張](../../framework/extensions.md)を、ポートとDataSpecの境界は[宣言要素（Specs）](../../framework/specs.md)を、候補に対する観測の対応は[Feedback](../observation_and_state/feedback.md)を参照してください。
公開型のimport経路は[APIリファレンス](../../api/index.md)で確認してください。
