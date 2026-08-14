---
primary_layer: layer4
related_layers: [layer3]
page_type: guide
---

# フレームワークを拡張する

既存コンポーネントの差し替えではなく、コンポーネントを検査し実行するフレームワークの契約を拡張します。

## 拡張対象

| 対象 | 役割 | 利用する公開API |
|---|---|---|
| Contract | ポート、状態、ライフサイクル、実行能力を宣言する | `ComponentContract` |
| SearchSpace | Genomeの表現、サンプリング、検証、空間固有サービスを提供する | `SearchSpace` |
| Graph | コンポーネントとデータ、制御、状態の関係を記述する | `ComponentGraph` |
| Compiler rule | 契約の互換性や変換を検査する | `CompilationRule` |
| Feedback | 候補IDと観測を対応付ける | `FeedbackBatch`、`FeedbackBuilder` |
| Runtime | 計画を進め、状態Patchと待機を管理する | `ExecutionRuntime` |

各対象の設計は、[フレームワークの概要](index.md)から[Component](component.md)、[ComponentContract](contract.md)、[宣言要素（Specs）](specs.md)、[Graph](graph.md)、[Compiler](compiler.md)へ進みます。
SearchSpace、Feedback、Runtimeの拡張境界は、それぞれ[探索空間（SearchSpace）](../concepts/problem_and_ranking/search_space.md)、[Feedback](../concepts/observation_and_state/feedback.md)、[Runtime](runtime.md)を参照してください。

## 実装時の原則

契約に宣言していない状態やサービスへ、コンポーネントから直接アクセスしません。
状態更新は `StatePatch` で返し、Runtimeに適用を委譲します。
表現と意味の変換は明示的なAdapterとして登録し、ポート互換性の検査を迂回しません。
コンパイル時の検査と実行時の処理を同じ責務へまとめません。

## 公開APIの境界

フレームワーク拡張の公開語彙は `saealib.core`、`saealib.space`、`saealib.execution`、`saealib.policies` のファサードから取得します。
個別の実装モジュールや内部Runtimeクラスを、一般利用者向けの拡張APIとして案内しません。
Compiler本体やCompiler ruleの詳細なimportは、対象リリースのAPIリファレンスで公開経路を確認してください。

## 検証の観点

新しい拡張では、少なくとも次を検証します。

- 契約の入力と出力が、実際のGraph接続と一致すること。
- 宣言したStateKey以外を読み書きしないこと。
- Feedbackの候補IDと観測行の順序が一致すること。
- 同期評価、非同期評価、チェックポイント再開で状態境界が壊れないこと。
- 通常の利用者向け公開APIが、フレームワーク内部の型に依存しないこと。
