---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# フィードバック（Feedback）

フィードバックは、候補に対応する観測値をアルゴリズムが利用できる形へ揃えた結果です。
真の評価結果と予測結果は同じ候補IDに対応しますが、観測の出所を区別して保持します。

## 提案と観測

`ProposalBatch` は候補と候補間の関係、必要なフィードバック量をまとめます。
`ObservationBatch` は観測スキーマと観測レコードを持ち、目的、制約、特徴、コストなどの量を観測の出所と状態とともに表します。
提案が要求する量と、観測バッチが提供する量が揃ったとき、提案は後続のフィードバック処理へ進めます。

## FeedbackとFeedbackBuilder

`FeedbackBatch` は一つの提案に対して届けられた観測バッチです。
`FeedbackBuilder` は候補、予測、評価結果を受け取り、アルゴリズムが読む `FeedbackResult` を作ります。
ビルダーは候補IDに沿って行を揃え、目的値、制約値、評価済みマスク、値の出所を返します。

## 保持する情報

| 型 | 保持する情報 | 所有する境界 |
|---|---|---|
| `ProposalBatch` | proposal ID、候補、候補間の関係、Feedback要件、メタデータ | 提案側 |
| `ObservationBatch` | 観測スキーマと観測レコード | 評価または観測側 |
| `FeedbackBatch` | proposal ID、観測、channel、完了状態、sequence | 配信側 |
| `FeedbackResult` | candidate ID、目的値、制約値、評価済みマスク、値の出所 | Algorithmのtell境界 |

`FeedbackResult`の各行はcandidate IDで識別し、`evaluated_mask`と`source`を同じ行に保持します。
これにより、予測値で補われた行と直接評価された行を、後続のAlgorithmや学習データが区別できます。

## 真値、予測値、混合フィードバック

`TrueOnlyFeedback` は真の評価で完了した行だけを返します。
`PredictedFeedback` は目的の予測チャンネルを候補全体に返し、真の評価がない場合の近似的な更新に使います。
`MixedFeedback` は真の評価を優先し、残りの行を目的予測で補います。

この三つは「どの値が正しいか」を判定する機能ではなく、利用可能な観測を選択して一つのフィードバック結果に配置する方針です。
`true`、`surrogate`、`human`、`simulator` などの出所は観測側に残るため、後続の契約や学習データが利用条件を検証できます。

## 責務と不変条件

Feedbackの責務は、候補ID、観測行、値の出所、評価済み状態を対応付けることです。
候補の表現を生成したり、観測値の正しさを再計算したりする責務は持ちません。
同じ候補IDの行を重複させず、提案が要求する量と観測が提供する量を区別し、真の評価と予測の出所を失わないことが不変条件です。

`ProposalBatch`と候補は提案時に生成され、`ObservationBatch`は評価、予測、外部観測の完了時に生成されます。
`FeedbackBuilder`は後続アルゴリズムが利用する直前に両者を照合し、Runtimeは非同期評価の回収後にその結果を次の状態境界へ渡します。

## 検証と拡張

CompilerはFeedback関連のPortContract、schema binding、required service、lifecycle feedbackを検証します。
Runtimeは候補IDの対応、未完了観測の待機、同じ結果の二重適用を検査します。
候補IDの欠落、行数や順序の不一致、要求量の不足、真の評価を予測で上書きする選択はDiagnosticsまたはFeedback構築エラーになります。

新しいFeedback方針を追加する場合は、観測の出所と評価済みマスクを保持するBuilder境界を拡張します。
提案や観測の契約を変更する場合は[宣言要素（Specs）](../../framework/specs.md)と[Compiler](../../framework/compiler.md)を、非同期の適用境界は[Runtime](../../framework/runtime.md)を参照してください。
公開型のimport経路は[APIリファレンス](../../api/index.md)で確認してください。
