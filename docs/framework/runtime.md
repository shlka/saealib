---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# ランタイム（Runtime）

ランタイムは、コンパイル済みの計画と状態を受け取り、観測可能な状態境界まで処理を進めます。
計画を作るコンパイラと、計画を進めるランタイムは別の責務を持ちます。

## 責務と所有者

Runtimeの責務は、ExecutablePlanを実行位置と状態へ結び付け、StatePatch、イベント、待機を順序付けることです。
契約の互換性やサービスの解決はCompilerの責務であり、Runtimeが未検証のGraphを修正することはありません。
現在の状態と実行位置の所有者はRuntimeSessionで、永続的な値の所有者はStateStoreです。

## ExecutablePlanとExecutionRuntime

`ExecutablePlan` は `ComponentGraph` の検証済み表現であり、必要なランタイム能力と診断情報を持ちます。
`ExecutionRuntime` は計画を初期化し、状態を一段進め、終了や再コンパイルなどの要求を処理するプロトコルです。
ランタイムはノードの結果に含まれる `StatePatch`、イベント、コマンドを順に適用します。

## RuntimeSessionとRuntimeStep

`RuntimeSession` は計画、現在の状態、実行位置、完了状態をまとめた再開可能なセッションです。
`RuntimeStep` は一回の進行結果であり、次のセッション、状態、観測可能性を返します。
この分離により、チェックポイントや非同期評価の待機を、同じ実行モデルで扱えます。

RuntimeSessionは不変の実行スナップショットであり、Runtimeは次のSessionを返すことで再開位置を明示します。
StatePatchの適用、イベントの配送、コマンドの処理が完了した境界だけを次のStepとして公開します。

| 値 | 保持する情報 |
|---|---|
| `RuntimeSession` | `ExecutablePlan`、現在の`OptimizationState`、実行位置、完了状態、構造化領域のFrame |
| `RuntimeStep` | 更新後の状態、実行したノード、`NodeResult`、待機状態、次のSession |

`ExecutionRuntime.initialize(plan, state)`はSessionを作り、`advance(session)`は一回分のRuntimeStepを返します。
アプリケーションはStepの`finished`、`observable`、`session`を使って、次の処理または再開位置を判断します。

## PipelineRuntimeと非同期待機

`PipelineRuntime` は通常のパイプラインを順に実行し、`AsyncPipelineRuntime` は非同期評価の提出と回収を含む待機状態を扱います。
非同期評価では、提出した要求がすぐに完了するとは限らないため、ランタイムは状態を保持したまま次のポーリングへ戻ります。
評価の完了時に得た更新をパッチとして適用してから、後続ステージを再開します。

`RuntimeSession` はこの待機位置を保持し、`RuntimeStep` は進捗があったかどうかと次のセッションを返します。
非同期評価の実装は `AsyncEvaluator` とスケジューラーが担い、`AsyncPipelineRuntime` はそれを計画の実行境界へ接続します。

Runtimeは、Planが要求するRuntime capabilityが実行環境にあること、NodeResultのStatePatchが宣言された状態効果に収まること、待機中の要求を重複適用しないことを確認します。
計画の診断、未宣言の状態書き込み、無効な実行位置、回収できない非同期要求は実行を進めず、Diagnosticsまたは失敗結果として扱います。

## 実行器（Runner）

`Runner` は `Optimizer` から計画と初期状態を取得し、ランタイムを反復して状態の境界だけを公開する薄い内部実装です。
アプリケーションが依存する安定した構成APIではなく、実行ランタイムを駆動するための内部コンポーネントとして扱います。
通常の利用では `minimize()`、`maximize()`、`Optimizer.run()`、`Optimizer.iterate()` を使います。

## 拡張点と関連ページ

新しい実行能力や待機状態を追加する場合は、ComponentContractのexecutionとRuntimeSessionの遷移を同時に定義します。
既存のStageを組み込む場合は、StageのOptimizationState境界をアダプターで接続し、graph-native ComponentのStateView境界と混同しません。
実行計画の構成は[Compiler](compiler.md)を、状態の所有と変更は[OptimizationState](../concepts/observation_and_state/optimization_state.md)を、拡張手順は[フレームワーク拡張](extensions.md)を参照してください。
具体的なRuntime型の公開import経路は[APIリファレンス](../api/index.md)で確認してください。
