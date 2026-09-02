---
primary_layer: cross
related_layers: [layer2, layer3]
page_type: entry
---

# 実行と評価

既存のStageベース経路では、Initializerが初期集団を作り、OptimizationStrategyが世代ごとの処理を進めます。 Algorithmが候補を生成し、SurrogateManagerとAcquisitionFunctionが予測とスコア計算を行った後、Evaluatorが高コストな真の評価を実行します。 評価結果はAlgorithmの集団とArchiveへ反映され、Terminationが停止条件を満たすまで世代処理を繰り返します。

## パイプラインの流れ

次の図は、初期化から終了判定までのデータの流れを示します。 Surrogateを使わない構成では、予測とスコア計算の部分を省略できます。

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Create initial population"] --> STEP
    subgraph STEP["OptimizationStrategy.step()<br/>Process one generation"]
        direction TB
        ASK["Algorithm.ask()<br/>Generate candidates"] --> PREDICT["SurrogateManager.predict()<br/>Predict"]
        PREDICT --> SCORE["AcquisitionFunction<br/>Score candidates"]
        SCORE --> SEL["Select candidates for true evaluation"]
        SEL --> EVAL["Evaluator → Problem<br/>Expensive evaluation"]
        EVAL --> TELL["Algorithm.tell()<br/>Update population"]
    end
    STEP --> TERM{"Termination?"}
    TERM -- "No" --> STEP
    TERM -- "Yes" --> RESULT([Result])
    EVAL -- "Evaluated points" --> ARC[("Archive")]
    ARC -. "Training data" .-> PREDICT
```

各StageはCallbackManagerを通じて型付きイベントを発行します。 そのため、Stageをサブクラス化せずに進行状況を観測したり、実行時の差し替えを行ったりできます。

## コンポーネントの役割

| コンポーネント | 役割 |
|---|---|
| [Problem](problem_and_ranking/problem.md) | 目的関数、設計変数、探索範囲、制約、最適化方向を定義します。 |
| [Initializer](execution_and_evaluation/initialization.md) | ループ開始前の初期集団とArchiveを生成します。 |
| [Algorithm](search_algorithms/algorithm.md) | `ask()`で候補を生成し、`tell()`で評価結果を集団へ反映します。 |
| [OptimizationStrategy](execution_and_evaluation/strategies.md) | 一世代の処理を組み立て、真の評価へ送る候補を決めます。 |
| [SurrogateManager](surrogate_modeling/surrogate_manager.md) | Surrogateの学習と予測を調整します。 |
| [Surrogate](surrogate_modeling/surrogate.md) | 学習データから予測します。スコア計算の方法は持ちません。 |
| [AcquisitionFunction](surrogate_modeling/acquisition_functions.md) | 予測結果を候補選択用のスコアへ変換します。 |
| [Evaluator](execution_and_evaluation/evaluation.md) | 目的関数を逐次または並列に評価します。 |
| [Population](observation_and_state/population.md) | 評価済みの個体と集団のデータを保持します。 |
| [Termination](execution_and_evaluation/termination.md) | 世代処理を終了する条件を判定します。 |
| [CallbackManager](observation_and_state/callbacks.md) | パイプライン全体のイベントを観測し、記録します。 |

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`play;sd-mr-1` 初期化（Initializer）
:link: execution_and_evaluation/initialization
:link-type: doc
初期集団とArchiveを生成する方法を選びます。
:::

:::{grid-item-card} {fa}`microchip;sd-mr-1` Evaluator
:link: execution_and_evaluation/evaluation
:link-type: doc
目的関数の逐次評価と並列評価を実行します。
:::

:::{grid-item-card} {fa}`chess-knight;sd-mr-1` OptimizationStrategy
:link: execution_and_evaluation/strategies
:link-type: doc
候補生成、予測、評価計画、状態更新の流れを組み立てます。
:::

:::{grid-item-card} {fa}`stop;sd-mr-1` Termination
:link: execution_and_evaluation/termination
:link-type: doc
評価回数や世代数などの終了条件を組み合わせます。
:::

::::

```{toctree}
:hidden:

execution_and_evaluation/initialization
execution_and_evaluation/evaluation
execution_and_evaluation/strategies
execution_and_evaluation/termination
```
