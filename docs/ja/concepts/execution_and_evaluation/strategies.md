---
primary_layer: layer3
related_layers: [layer2]
page_type: concept
---

# OptimizationStrategy

`saealib` は、高コストな真の評価へ送る候補の選択を `OptimizationStrategy` に委譲します。 Strategyは、候補生成、予測、評価計画、Feedback、Population更新を一つのグラフとして記述します。

## OptimizationStrategyの役割

graph-native経路の拡張点は `build_graph(provider) -> ComponentGraph` です。 Optimizerはこのグラフを実行前にCompilerへ渡し、`ExecutablePlan`を作成します。

`build_pipeline(provider) -> Pipeline` はStage互換経路で、既存のPipeline DSLを利用するStrategyが提供できる互換性用の記述形式です。 この形式を使う場合も、構造化Pipelineへ入れるStageは `stage_component(stage)` で明示的に包みます。

`step(ctx, provider)` はStage互換経路に残る境界です。 新しいグラフネイティブコンポーネントは `StateView` を読み、`StatePatch` または `NodeResult` を返します。

クラス属性`requires_surrogate: bool`は、この戦略が`SurrogateManager`を必要とするかを示します。 `Optimizer.validate()`がこの属性を見て、`surrogate_manager`が未設定のまま`requires_surrogate=True`の戦略を使おうとしていないかを確認します。

## 組み込みStrategy

| クラス | パラメータ | 方式 |
|---|---|---|
| `DirectStrategy` | なし | サロゲートを使わず、生成した候補を全件真に評価する |
| `SteadyStateStrategy` | なし | 1ステップにつき1候補を生成し、非同期補充に対応する |
| `IndividualBasedStrategy` | `evaluation_ratio: float = 0.1` | 全候補をサロゲートでスコアリングし、上位`evaluation_ratio`の割合だけ真に評価する |
| `PreSelectionStrategy` | `n_candidates: int, n_select: int`（共に必須） | `n_candidates`件を生成してスコアリングし、上位`n_select`件だけ真に評価する |
| `GenerationBasedStrategy` | `gen_ctrl: int`（必須） | `gen_ctrl`世代分をサロゲートのみで進め、その後1世代だけ真に評価する |

`IndividualBasedStrategy`は個体の割合、`PreSelectionStrategy`は個体の件数で選抜する点が異なります。 `GenerationBasedStrategy`は個体単位ではなく世代単位でサロゲートと真の評価を切り替えます。 `DirectStrategy`はサロゲートを一切使わない比較対象で、`requires_surrogate=False`です。

定常状態実行では、`DirectStrategy`と`AsyncEvaluationScheduler`を組み合わせます。スケジューラは利用可能なワーカースロットを埋め、ブロックせずにポーリングし、完了した更新をライフサイクル順にコミットします。

### 各Strategyのパイプライン構成

| クラス | パイプライン |
|---|---|
| `DirectStrategy` | 世代数のカウント → 要求 → 真の評価 → アーカイブ更新 → 反映 |
| `IndividualBasedStrategy` | CountGeneration → Ask → SurrogatePredictStage → AcquisitionStage → SortByScore → TrueEvaluation（比率指定）→ ArchiveUpdate → Tell |
| `PreSelectionStrategy` | CountGeneration → Ask（n_candidates）→ SurrogatePredictStage → AcquisitionStage → TopKSelection（k=n_select）→ TrueEvaluation → ArchiveUpdate → Tell |
| `GenerationBasedStrategy` | SurrogateOnlyLoop(gen_ctrl回) → CountGeneration → Ask → TrueEvaluation → ArchiveUpdate → Tell |

各Strategyの構成要素には、`EvaluationPlan`、非同期評価の送信と収集、Feedback、Tellが含まれます。Stageはその一部を担う互換性用の実行単位であり、構造化されたパイプライン全体と同じ実行契約ではありません。詳細は[Stage](../observation_and_state/stage.md)と[Framework](../../framework/index.md)を参照してください。

## どのStrategyを選ぶか

評価コストが極めて高い問題では、サロゲートで大半の候補を足切りする`IndividualBasedStrategy`や`PreSelectionStrategy`が有効です。 サロゲートの信頼度がまだ低い探索初期や、サロゲートの学習コスト自体を頻繁に払いたくない場合は、複数世代をまとめてサロゲートのみで進める`GenerationBasedStrategy`が適しています。 サロゲートの近似誤差そのものが許容できない、あるいは評価コストが十分低い問題では、`DirectStrategy`で真の評価だけに頼るのが妥当です。

## 実行時差し替えの挙動

`iterate()` / `run()` のステップ境界で `provider.algorithm` や `provider.surrogate_manager` などの構成を変更すると、実行環境が変化を検出してplanを再コンパイルし、次世代から反映します。 この手順はStage互換経路とgraph-native経路のどちらでも利用できます。 Component側から再コンパイルを要求するときは、plan nodeが `NodeResult.commands` で `RuntimeCommand` / `RequestRecompile` を返します。 要求はRuntime環境が `recompile()` を提供する場合に受理され、再コンパイルはステップ間で行われます。受理されないことも正常な結果です。

## 独自Strategyの実装方法

独自の候補選択方式を実装するときは、graph-nativeの経路とStage互換の経路を分けて選びます。

**グラフネイティブ経路**：`build_graph()`を実装して`ComponentGraph`を返します。パイプラインDSLを使って段階的に移行する場合は、各Stageを`stage_component()`でラップしてから`lower_pipeline()`に渡します。

```python
from saealib import OptimizationStrategy, Pipeline
from saealib.core import lower_pipeline
from saealib.stages import stage_component


class CustomStrategy(OptimizationStrategy):
    """Skeleton for building a Strategy ComponentGraph from the Pipeline DSL."""

    requires_surrogate = False

    def build_graph(self, provider):
        pipeline = Pipeline(
            steps=[
                stage_component(first_stage),
                stage_component(second_stage),
            ]
        )
        return lower_pipeline(pipeline)
```

`first_stage` と `second_stage` は、実際に組み合わせるStageインスタンスを表します。 この例はPipelineからComponentGraphを作る境界だけを示しており、評価計画やFeedbackの契約は各Stageの構成に応じて追加します。

既存のStrategyの一部だけを調整する場合は、新しいStrategyを作るよりも [Pipeline.replace/find](../extension_guidelines.md) で互換性用パイプラインを変更する方が適しています。

## 関連コンポーネント

- [Stage](../observation_and_state/stage.md)：各Strategyが組み合わせるパイプラインステージ単体の契約
- [SurrogateManager](../surrogate_modeling/surrogate_manager.md)：`requires_surrogate=True`の戦略が使う予測機構
- [拡張のガイドライン](../extension_guidelines.md)：`Pipeline.replace`/`find`によるステージの並べ替え
- [コンポーネント概要](../index.md)：パイプライン全体の構成図

## 参照

- {py:class}`saealib.OptimizationStrategy`
- {py:class}`saealib.IndividualBasedStrategy`
- {py:class}`saealib.GenerationBasedStrategy`
- {py:class}`saealib.PreSelectionStrategy`
- {py:class}`saealib.DirectStrategy`
- {py:class}`saealib.AsyncEvaluationScheduler`
