---
primary_layer: layer4
related_layers: [layer3]
page_type: guide
---

# フレームワークを拡張する

既存コンポーネントの差し替えではなく、コンポーネントを検査し実行するフレームワークの契約を拡張します。

## 拡張対象

| 対象 | 役割 | 公開API |
|---|---|---|
| 契約 | ポート、状態、ライフサイクル、実行能力を宣言する | `ComponentContract` |
| SearchSpace | Genomeの表現、サンプリング、検証、空間固有サービスを提供する | `SearchSpace` |
| Graph | コンポーネント、データ、制御、状態の関係を記述する | `ComponentGraph` |
| Compilerルール | 契約の互換性と変換を検査する | `CompilationRule` |
| フィードバック | 候補IDと観測を対応付ける | `FeedbackBatch`、`FeedbackBuilder` |
| Runtime | 計画を進め、状態パッチと待機を管理する | `ExecutionRuntime` |

各対象の設計は、[フレームワークの概要](index.md)から[Component](component.md)、[ComponentContract](contract.md)、[Specs](specs.md)、[Graph](graph.md)、[Compiler](compiler.md)へ進みます。SearchSpace、Feedback、Runtimeの拡張境界については、それぞれ[SearchSpace](../concepts/problem_and_ranking/search_space.md)、[Feedback](../concepts/observation_and_state/feedback.md)、[Runtime](runtime.md)を参照してください。

## 実装時の原則

コンポーネントの契約に宣言されていない状態やサービスへアクセスさせません。状態更新は`StatePatch`値として返し、その適用をRuntimeに委譲します。表現変換と意味変換は明示的なアダプターとして登録し、ポート互換性の検査を迂回しません。コンパイル時の検査と実行時の処理を分離します。

## 公開APIの境界

`saealib.core`、`saealib.space`、`saealib.execution`、`saealib.policies`のファサードからフレームワーク拡張の公開語彙を取得します。個別の実装モジュールや内部Runtimeクラスを一般利用者向けの拡張APIとして示しません。CompilerとCompilerルールの公開import経路は、リリース固有のAPIリファレンスで確認します。

## 検証チェックリスト

新しい拡張では、少なくとも次を検証します。

- 契約の入力と出力が、実際のGraph接続と一致すること。
- 宣言したStateKey以外を読み書きしないこと。
- 各Observationのsubjectが有効なcandidateまたはproposal relationに解決され、consumerが宣言したordering/completion契約を満たすことを確認します。検証は元の位置に依存しません。
- 同期評価、非同期評価、チェックポイント再開で状態境界が壊れないこと。
- 通常の利用者向け公開APIが、フレームワーク内部の型に依存しないこと。
