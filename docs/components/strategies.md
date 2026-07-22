# OptimizationStrategy

`saealib`は、どの候補解に高コストな真の評価を割り当てるかという判断を、`OptimizationStrategy`という差し替え可能なコンポーネントに委ねている。
`step()`が1世代分の「生成、スコアリング、評価、更新」を実行する。

## OptimizationStrategyの役割

`OptimizationStrategy`が実装を要求するメソッドは`step(ctx, provider) -> OptimizationState | None`の1つだけである。
`None`を返すのは、`ctx`をin-place更新するレガシースタイルの実装向けの取り決めであり、組み込みの4種は全て更新後の`OptimizationState`を返す。

クラス属性`requires_surrogate: bool`は、この戦略が`SurrogateManager`を必要とするかを示す。
`Optimizer.validate()`がこの属性を見て、`surrogate_manager`が未設定のまま`requires_surrogate=True`の戦略を使おうとしていないかを確認する。

## 組み込みStrategy

| クラス | パラメータ | 方式 |
|---|---|---|
| `DirectStrategy` | なし | サロゲートを使わず、生成した候補を全件真に評価する |
| `IndividualBasedStrategy` | `evaluation_ratio: float = 0.1` | 全候補をサロゲートでスコアリングし、上位`evaluation_ratio`の割合だけ真に評価する |
| `PreSelectionStrategy` | `n_candidates: int, n_select: int`（共に必須） | `n_candidates`件を生成してスコアリングし、上位`n_select`件だけ真に評価する |
| `GenerationBasedStrategy` | `gen_ctrl: int`（必須） | `gen_ctrl`世代分をサロゲートのみで進め、その後1世代だけ真に評価する |

`IndividualBasedStrategy`は個体の割合、`PreSelectionStrategy`は個体の件数で選抜する点が異なる。
`GenerationBasedStrategy`は個体単位ではなく世代単位でサロゲートと真の評価を切り替える。
`DirectStrategy`はサロゲートを一切使わない比較対象で、`requires_surrogate=False`である。

### 各Strategyのパイプライン構成

| クラス | パイプライン |
|---|---|
| `DirectStrategy` | CountGeneration → Ask → TrueEvaluation → ArchiveUpdate → Tell |
| `IndividualBasedStrategy` | CountGeneration → Ask → SurrogateScore → SortByScore → TrueEvaluation(比率指定) → ArchiveUpdate → Tell |
| `PreSelectionStrategy` | CountGeneration → Ask(n_candidates件) → SurrogateScore → TopKSelection(k=n_select) → TrueEvaluation → ArchiveUpdate → Tell |
| `GenerationBasedStrategy` | SurrogateOnlyLoop(gen_ctrl回) → CountGeneration → Ask → TrueEvaluation → ArchiveUpdate → Tell |

各[Stage](stage.md)単体の契約は、そちらのページを参照する。
全体のパイプライン図は[コンポーネント概要](index.md)を参照する。

## どのStrategyを選ぶか

評価コストが極めて高い問題では、サロゲートで大半の候補を足切りする`IndividualBasedStrategy`や`PreSelectionStrategy`が有効である。
サロゲートの信頼度がまだ低い探索初期や、サロゲートの学習コスト自体を頻繁に払いたくない場合は、複数世代をまとめてサロゲートのみで進める`GenerationBasedStrategy`が適している。
サロゲートの近似誤差そのものが許容できない、あるいは評価コストが十分低い問題では、`DirectStrategy`で真の評価だけに頼るのが妥当である。

## 実行時差し替えの挙動

各Strategyの`step()`は、呼ばれるたびに無条件で`self.pipeline = self._build_pipeline(provider)`を実行してからパイプラインを実行する。
パイプラインをキャッシュしないため、`provider.algorithm`や`provider.surrogate_manager`を実行中に差し替えても、次の世代から確実に反映される。

## 独自Strategyの実装方法

独自の候補選抜方式が必要な場合、2つのアプローチがある。

**`OptimizationStrategy`を直接継承する**：`step()`を自分で実装する。
[Pipeline/Stage](extension_guidelines.md)を組み合わせて新しいパイプラインを構築する場合も、この形になる。

```python
from saealib import OptimizationStrategy, Pipeline
from saealib.stages import (
    CountGenerationStage, AskStage, TrueEvaluationStage,
    ArchiveUpdateStage, TellStage,
)


class SimpleDirectStrategy(OptimizationStrategy):
    """DirectStrategyと同じ内容を、Pipelineを自分で組み立てて再現する例。"""

    requires_surrogate = False

    def step(self, ctx, provider):
        cbmanager = getattr(provider, "cbmanager", None)
        pipeline = Pipeline([
            CountGenerationStage(),
            AskStage(provider.algorithm, cbmanager=cbmanager),
            TrueEvaluationStage(provider.evaluator, cbmanager=cbmanager),
            ArchiveUpdateStage(),
            TellStage(provider.algorithm),
        ])
        return pipeline.execute(ctx)
```

既存戦略のパイプラインを微調整したいだけであれば、`OptimizationStrategy`を新しく書くのではなく、[Pipeline.replace/find](extension_guidelines.md)で組み込みパイプラインの一部だけを差し替えるほうが軽量である。

## 関連コンポーネント

- [Stage](stage.md) — 各Strategyが組み合わせるパイプラインステージ単体の契約
- [SurrogateManager](surrogate_manager.md) — `requires_surrogate=True`の戦略が使うスコアリング機構
- [拡張のガイドライン](extension_guidelines.md) — `Pipeline.replace`/`find`によるステージの並べ替え
- [コンポーネント概要](index.md) — パイプライン全体の構成図

## 参照

- {py:class}`saealib.OptimizationStrategy`
- {py:class}`saealib.IndividualBasedStrategy`
- {py:class}`saealib.GenerationBasedStrategy`
- {py:class}`saealib.PreSelectionStrategy`
- {py:class}`saealib.DirectStrategy`
