---
primary_layer: layer4
related_layers: [layer2, layer3]
page_type: concept
---

# フィードバック

フィードバックは、候補に対応する観測をアルゴリズムが利用できる形で渡す契約です。行位置ではなく、対象、候補ID、提案関係、順序、状態、出所、完了の意味論でレコードを対応付けます。同じ候補に対する部分的な観測、順不同の観測、繰り返しの観測も有効な入力です。

## Feedbackの役割

候補に対応する観測値を、アルゴリズムが利用できるフィードバックへ渡します。候補の表現を生成したり、観測値の正しさを再計算したりする責務は持ちません。

## 提案と観測

`ProposalBatch` は候補と候補間の関係、必要なフィードバック量をまとめます。 `ObservationBatch` は観測スキーマと観測レコードを持ち、目的、制約、特徴、コストなどの量を観測の出所と状態とともに表します。 提案が要求する量と、観測バッチが提供する量が揃ったとき、提案は後続のフィードバック処理へ進めます。

## graph-native経路のFeedback

`FeedbackBatch`は一つの提案に対して届けられた観測バッチです。`ObservationBatch`は観測レコードの対象、量、状態、出所を保持し、`FeedbackBatch`は提案ID、チャネル、順序、最終フラグを加えます。利用側は`FeedbackContract`で順序、完了、多重度、グループ化を宣言します。`FeedbackBuilder`は観測を`FeedbackResult`の密な形へ投影する境界です。

## Stage互換経路のFeedbackResult

`FeedbackResult`はAlgorithmの`tell`へ渡す密な互換データ型です。この型では`candidate_ids`が一意で、`f`、`g`、`cv`、`evaluated_mask`、`source`、artifactsの各配列が同じ行数・形状規則に従います。これは`FeedbackResult`内の行の整合性を制約しますが、外部の`ObservationBatch`と行順を一致させる必要はありません。

## Feedbackが保持するもの

| 型 | 保持する情報 | 所有する境界 |
|---|---|---|
| `ProposalBatch` | 提案ID、候補、候補間の関係、Feedback要件、メタデータ | 提案側 |
| `ObservationBatch` | 観測スキーマと観測レコード | 評価または観測側 |
| `FeedbackBatch` | 提案ID、観測、チャネル、完了状態、順序 | 配信側 |
| `FeedbackResult` | 候補ID、目的値、制約値、評価済みマスク、値の出所 | Algorithmのtell境界 |

`FeedbackResult`の各行は候補IDで識別し、`evaluated_mask`と`source`を同じ行に保持します。これにより、予測値で補われた行と直接評価された行を、後続のAlgorithmや学習データが区別できます。

## 真値、予測値、混合フィードバック

`TrueOnlyFeedback` は真の評価で完了した行だけを返します。 `PredictedFeedback` は目的の予測チャンネルを候補全体に返し、真の評価がない場合の近似的な更新に使います。 `MixedFeedback` は真の評価を優先し、残りの行を目的予測で補います。

この三つは「どの値が正しいか」を判定する機能ではなく、利用可能な観測を選択して一つのフィードバック結果に配置する方針です。 `true`、`surrogate`、`human`、`simulator` などの出所は観測側に残るため、後続の契約や学習データが利用条件を検証できます。

## 不変条件

Feedbackの責務は、対象、候補ID、提案関係、順序、状態、出所、完了の意味論を対応付けることです。候補の表現を生成したり、観測値の正しさを再計算したりする責務は持ちません。同じ候補IDの繰り返し観測を一般Feedbackの不変条件で禁止せず、提案が要求する量と観測が提供する量を区別し、値の出所を失わないことが重要です。

`ProposalBatch`と候補は提案時に生成され、`ObservationBatch`は評価、予測、外部観測の完了時に生成されます。 `FeedbackBuilder`は後続アルゴリズムが利用する直前に両者を照合し、Runtimeは非同期評価の回収後にその結果を次の状態境界へ渡します。

## Feedbackの拡張

コンパイラはフィードバックに関係する`PortContract`、スキーマバインディング、必須サービス、ライフサイクルのフィードバックを検証します。ランタイムは候補IDの一致を確認し、未完了の観測を待ち、同じ結果が二度適用されるのを防ぎます。対象または提案との関係を解決できない観測、宣言した順序や完了の契約に違反する配信、要求量の不足、真の評価を予測で上書きする選択は、診断またはフィードバック構築エラーになります。

Feedback方針を追加する場合は、観測の出所と評価済みマスクを保持したままBuilder境界を拡張します。提案または観測の契約を変更する場合は[Specs](../../framework/specs.md)と[Compiler](../../framework/compiler.md)を、非同期の適用境界については[Runtime](../../framework/runtime.md)を参照してください。公開インポートパスは[APIリファレンス](../../api/index.md)を参照してください。

## 関連コンポーネント

- [Algorithm](../search_algorithms/algorithm.md)：フィードバックを `tell()` で消費します。
- [Evaluator](../execution_and_evaluation/evaluation.md)：評価結果を生成します。
- [OptimizationState](optimization_state.md)：実行状態を保持します。

## 参照

- {py:class}`saealib.FeedbackResult`
- {py:class}`saealib.FeedbackBuilder`
- {py:class}`saealib.core.StateView`
