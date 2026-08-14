---
primary_layer: layer3
related_layers: [layer2, layer4]
page_type: guide
---

# 既存の契約に独自Componentを追加する

前提は、既存のComponentの契約に収まる差し替えを実装することです。
:::{admonition} このページでできるようになること
:class: tip

このページを終えると、拡張点を選び、最小の契約を確認し、登録してテストできます。
:::

拡張点の分類は [拡張ガイドライン](../concepts/extension_guidelines.md) で、契約や実行基盤自体を変える場合は [フレームワーク拡張](../framework/extensions.md) に進みます。

importの方針は [Canonical Imports](../api/imports.md) にまとめています。

## 拡張点を選ぶ

ビルトインコンポーネントの設定変更で足りるなら [ビルトインコンポーネントの差し替え](component_swap.md) または [Hook](interface_hooks.md) を使います。
新しい探索、予測、評価、世代処理が必要なら、対応する契約を持つComponentを選びます。

| 変更内容 | 拡張点 | 最小の実装境界 |
|---|---|---|
| 候補生成と評価結果の反映 | `Algorithm` | `ask()` と `tell()`、必要な属性と契約 |
| 予測モデル | `Surrogate` | `fit()` と `predict()` |
| 予測から候補スコアを作る | `AcquisitionFunction` | `evaluate()`、または対応するPointwiseの `compute_reference()` と `score()` |
| 評価バックエンドを差し替える | `Evaluator` | `evaluate_batch()` と `EvaluationResult` |
| 世代の実行方針 | `Strategy` | `build_graph()` または既存のStrategy経路 |
| 互換実行単位 | `Stage` | `execute(state) -> state` |

各契約の正確な必須メソッドは、[Algorithmリファレンス](../api/algorithms.md)、[Surrogateリファレンス](../api/surrogate.md)、[Acquisitionリファレンス](../api/acquisition.md)、[Stageリファレンス](../api/stages.md) で確認します。
名前が似ていても契約は同一ではないため、別のComponentのメソッドを流用するわけにはいきません。

## 最小の契約を確認する

実装前に、既存の抽象基底クラスの抽象メソッド、`contract()`、実行時に参照される属性を確認します。
`Algorithm` の `ask()` と `tell()` は `StateView` を読み、`ProposalBatch` または `StatePatch` を返します。
`Surrogate` は配列を受け取り、`SurrogatePrediction` を返します。
`Stage` は `OptimizationState` を受け取り、更新後のStateを返します。

## 一種類を実装する

ここでは、評価処理だけを差し替える `Evaluator` の最小例を示します。
`evaluate_batch(x, problem) -> EvaluationResult` が実装境界であり、`EvaluationResult` は `f`、`g`、`cv` の2次元または1次元配列を受け取ります。

```python
import numpy as np
from saealib import EvaluationResult, Evaluator


class BatchSphereEvaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        values = np.sum(x**2, axis=1)
        return EvaluationResult(
            f=values[:, None],
            g=np.empty((len(x), 0)),
            cv=np.zeros(len(x)),
        )
```

独自Evaluatorを `Optimizer.set_evaluator()` に渡すと、組み込みの同期評価アダプターから呼び出されます。
評価結果の行数と配列形状は `EvaluationResult` が検証します。

## 登録または組み込む

直接使う場合は、公開された `Optimizer.set_evaluator(evaluator)` にインスタンスを渡します。
設定ファイルやPresetから名前で構築する場合だけ、`@register()` を付けてRegistryへ登録します。

```python
import numpy as np
from saealib import Optimizer, Problem, register


@register()
class NamedBatchSphereEvaluator(BatchSphereEvaluator):
    pass


problem = Problem(
    lambda x: np.sum(x**2),
    dim=3,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0] * 3,
    ub=[5.0] * 3,
)
optimizer = Optimizer(problem).set_evaluator(NamedBatchSphereEvaluator())
```

`Problem` の評価関数を常にこのEvaluatorで置き換えるとは限らないため、`set_evaluator()` を呼んだ構成を実際に使って検証します。

## 最小テストを置く

まず `evaluate_batch()` の形状と値を固定した小さな単体テストを書きます。
次に `Optimizer(problem).set_evaluator(...)` から短い `max_fe` で実行し、契約検証と実行経路を確認します。
抽象基底クラスの一覧化や実行基盤の変更はこのガイドの範囲ではなく、対応するAPIリファレンスまたはフレームワーク拡張の文書を参照します。

## 関連するConceptとReference

- [Evaluatorの概念](../concepts/execution_and_evaluation/evaluation.md)
- [拡張方針](../concepts/extension_guidelines.md)
- [Evaluationリファレンス](../api/evaluation.md)
- [Coreリファレンス](../api/core.md)

## コンポーネントの責務を分ける

既存の実行契約を保ったまま拡張する場合は、担当する責務と状態の境界を先に決めます。

| コンポーネント | 主な責務 | 状態の境界 |
|---|---|---|
| `Algorithm` | 候補の提案とFeedbackの消費 | `ProposalBatch`、`FeedbackBatch`、`StatePatch` |
| Operator | 交叉、突然変異、親選択、生存者選択 | Algorithmから渡された候補と乱数 |
| `Surrogate` | 予測モデルのfitとpredict | Archiveの学習データと予測結果 |
| `OptimizationStrategy` | 評価計画、候補選択、Feedbackの流れ | `build_graph()`またはStage互換経路 |
| `Stage` | 既存sequential経路の一処理 | `OptimizationState`と`replace()` |

各コンポーネントは、対応するConceptページの契約と組み込み実装を確認してから実装します。
`Algorithm`の`ask()`と`tell()`、Strategyの`build_graph()`、Stageの`execute()`は同じ境界ではありません。

## 実装順とStageの移行

1. `Problem`と`SearchSpace`が返すデータ形式を確認します。
2. 公開名前空間から依存コンポーネントを取得します。
3. コンポーネント単体の契約と状態アクセスを実装します。
4. `Optimizer`の差し替えAPIで組み合わせます。
5. Feedbackの候補ID、評価の出所、チェックポイント動作を確認します。

既存Stageを構造化Pipelineへ接続する場合は、`stage_component(stage)`を明示します。
新しいgraph-native componentでは、Stageの`OptimizationState`境界を再利用せず、`ComponentContract`、`StateView`、`StatePatch`を実装します。

詳細は[Algorithm](../concepts/search_algorithms/algorithm.md)、[OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md)、[Stage](../concepts/observation_and_state/stage.md)を参照してください。
