# コンポーネント

各コンポーネントの詳しい使い方と拡張ガイドラインです。
まず全体のパイプラインでコンポーネント同士がどう組み合わさるかを示し、そのあとページ単位の詳細に入ります。

## パイプライン全体の構成

`Optimizer`は以下のコンポーネントを束ね、`Termination`が終了と判定するまで世代ループを駆動します。
1世代分の処理は`OptimizationStrategy`が統括します。
`Algorithm`が候補解を生成し（ask）、`SurrogateManager`がそれを安価にスコアリングし、戦略がどの候補解に高コストな真の評価を割り当てるかを決め、結果は`Algorithm`側の母集団（tell）と`Archive`の両方に反映されます。
`Archive`はサロゲートの学習データも兼ねています。

```{mermaid}
flowchart TD
    INIT["Initializer<br/>(初期集団の生成)"] --> STEP
    subgraph STEP["OptimizationStrategy.step() — 1世代分"]
        direction TB
        ASK["Algorithm.ask()<br/>候補解を生成"] --> SCORE["SurrogateManager<br/>score_candidates()"]
        SCORE --> SEL["真の評価を行う<br/>候補解を選択"]
        SEL --> EVAL["Evaluator →<br/>Problem(高コスト)"]
        EVAL --> TELL["Algorithm.tell()<br/>母集団を更新"]
    end
    STEP --> TERM{"Termination?"}
    TERM -- "継続" --> STEP
    TERM -- "終了" --> RESULT([Result])
    subgraph SM["SurrogateManager"]
        direction TB
        SUR["Surrogate<br/>fit / predict"] --> ACQ["AcquisitionFunction<br/>予測値→スカラースコア"]
    end
    SCORE -.-> SM
    EVAL -- "評価済みの点" --> ARC[("Archive")]
    ARC -. "学習データ" .-> SUR
```

各ステージは`CallbackManager`経由で型付きイベントを発火するため、サブクラス化せずにパイプラインの途中経過を観察したり操作したりできます（[CallbackManager](callbacks.md)を参照）。

各コンポーネントの役割は次の通りです。

| コンポーネント | 役割 |
|---|---|
| [Problem](problem.md) | 目的関数・設計変数・探索範囲・制約・最適化の方向(direction)を定義する |
| [Initializer](initialization.md) | ループ開始前に初期集団とアーカイブを生成する |
| [Algorithm](algorithm.md) | 進化的探索本体（GA/PSO）。`ask()`が候補解を生成し、`tell()`が母集団を更新する |
| [OptimizationStrategy](strategies.md) | 1世代分のパイプラインを統括し、どの候補解が真の評価を受けるかを決める |
| [SurrogateManager](surrogate_manager.md) | サロゲートのフィットとスコアリングを橋渡しし、`score_candidates()`を公開する |
| [Surrogate](surrogate.md) | アーカイブのデータでフィットし予測する。スコアリングの方法は関知しない |
| [AcquisitionFunction](acquisition_functions.md) | 予測値を（高いほど良い）スカラースコアへ変換する。モデルの詳細は関知しない |
| [Evaluator](evaluation.md) | 真の評価を実行する（逐次、または`parallel`エクストラによる並列） |
| [Archive](population.md) | 真に評価済みの点をすべて蓄積する。サロゲートの学習データセットも兼ねる |
| [Termination](termination.md) | ループの終了条件を判定する（既定は最大評価回数） |
| [CallbackManager](callbacks.md) | パイプライン各所のイベントを観察・記録し、実行時のコンポーネント差し替えにも使う |

## 低レベルAPIでの組み立て

`Optimizer`を直接使うと、各コンポーネントをメソッドチェーンで個別に差し替えられます。

```python
from saealib import GA, Optimizer, Problem, IndividualBasedStrategy
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.surrogate import GlobalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel
from saealib.acquisition import MeanPrediction
from saealib.termination import Termination, max_fe

problem = Problem(func, dim=5, lb=[-5] * 5, ub=[5] * 5, n_obj=1, direction=[-1])

opt = (
    Optimizer(problem)
    .set_algorithm(GA(
        CrossoverBLXAlpha(prob=0.7, alpha=0.4),
        MutationUniform(prob_var=0.3),
        SequentialSelection(),
        TruncationSelection(),
    ))
    .set_surrogate_manager(
        GlobalSurrogateManager(RBFSurrogate(gaussian_kernel, dim=5), MeanPrediction())
    )
    .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.1))
    .set_termination(Termination(max_fe(100)))
)

ctx = opt.run()
```

高レベルAPI（`minimize()`/`maximize()`）は、このパイプラインをsensible defaultsで自動構成したものです。
世代ごとの検査やカスタムループ制御が必要な研究用途では、`.run()`の代わりに`.iterate()`を使います。
`.iterate()`を使った実例は[単目的最適化](../tutorials/single_objective.md)などのチュートリアルにあります。

各Strategyが内部でこのパイプラインをどう構成するかは[OptimizationStrategy](strategies.md)を、パイプラインを構成するステージ単体の契約は[Stage](stage.md)を参照してください。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 拡張のガイドライン
:link: extension_guidelines
:link-type: doc

サブクラス化では重すぎる場合に: `with_post`/`with_post_fit`、`Pipeline`/`Stage`、`CallbackManager`、`Registry`。
:::

::::

## 問題定義とランキング

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Problem
:link: problem
:link-type: doc

目的関数、変数、方向(direction)、制約を定義します。
:::

:::{grid-item-card} ConstraintHandler
:link: constraints
:link-type: doc

制約違反の集約、実行可能性判定、修復戦略を独自実装する方法。
:::

:::{grid-item-card} Comparator
:link: comparators
:link-type: doc

解の順位付け。NSGA2/SPEA2/HVなどの使い分け。
:::

:::{grid-item-card} Dominator
:link: dominance
:link-type: doc

Pareto支配やε支配などの支配関係を独自実装する方法。
:::

:::{grid-item-card} NonDominatedSorter
:link: nondominated_sorting
:link-type: doc

非優越ソートアルゴリズムの差し替え。crowding distanceやSPEA2 fitnessの計算も。
:::

:::{grid-item-card} Decomposition
:link: decomposition
:link-type: doc

MOEA/D風のスカラー化（分解）関数と`DecompositionComparator`。
:::

::::

## 探索アルゴリズム

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Algorithm
:link: algorithm
:link-type: doc

GAとPSO、カスタム`Algorithm`の実装方法。
:::

:::{grid-item-card} Crossover
:link: crossover
:link-type: doc

BLX-α/SBX/一様交叉など。混合変数問題での使い分け。
:::

:::{grid-item-card} Mutation
:link: mutation
:link-type: doc

一様/ガウス/多項式突然変異など。混合変数問題での使い分け。
:::

:::{grid-item-card} ParentSelection
:link: parent_selection
:link-type: doc

トーナメント選択やルーレット選択などの親選択方式。
:::

:::{grid-item-card} SurvivorSelection
:link: survivor_selection
:link-type: doc

打ち切り選択などの生存選択（世代交代）方式。
:::

::::

## サロゲートモデリング

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Surrogate
:link: surrogate
:link-type: doc

組み込みサロゲート、外部ライブラリアダプタ、独自Surrogateの実装方法。
:::

:::{grid-item-card} SurrogateManager
:link: surrogate_manager
:link-type: doc

サロゲートの予測と獲得関数のスコアリングを橋渡しします。
:::

:::{grid-item-card} TrainingSet
:link: training_set
:link-type: doc

サロゲートの学習データをどこから、どのラベルで抽出するか。
:::

:::{grid-item-card} AcquisitionFunction
:link: acquisition_functions
:link-type: doc

サロゲートの予測値から候補解をスコアリングします。
:::

:::{grid-item-card} AccuracyBasedSurrogateSwitcher
:link: surrogate_switching
:link-type: doc

精度指標、評価方法、`iterate()`ループでの動的なコンポーネント切り替え。
:::

::::

## 実行基盤と評価戦略

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Initializer
:link: initialization
:link-type: doc

初期集団とアーカイブの生成方法（LHS/Random/Sobol）と独自実装。
:::

:::{grid-item-card} Evaluator
:link: evaluation
:link-type: doc

目的関数評価の逐次実行と並列実行のバックエンド。
:::

:::{grid-item-card} OptimizationStrategy
:link: strategies
:link-type: doc

どの候補解に真の評価を行うかを決定します。
:::

:::{grid-item-card} Termination
:link: termination
:link-type: doc

終了条件の組み合わせ方（`|`/`&`/`~`）と独自条件の書き方。
:::

::::

## 観察とコアデータ構造

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} CallbackManager
:link: callbacks
:link-type: doc

最適化パイプラインを観察し、ログ記録や内部へのアクセスを行います。
:::

:::{grid-item-card} Stage
:link: stage
:link-type: doc

OptimizationStrategy内部の世代ループを構成するステージ。独自Stageの実装方法。
:::

:::{grid-item-card} OptimizationState
:link: optimization_state
:link-type: doc

パイプラインを貫通する状態(ctx)。主要フィールドとチェックポイント。
:::

:::{grid-item-card} Population
:link: population
:link-type: doc

個体群とアーカイブのデータ構造とAPI。Archive/ParetoArchiveの仕組み。
:::

::::

```{toctree}
:hidden:

extension_guidelines
problem
constraints
comparators
dominance
nondominated_sorting
decomposition
algorithm
crossover
mutation
parent_selection
survivor_selection
surrogate
surrogate_manager
training_set
acquisition_functions
surrogate_switching
initialization
evaluation
strategies
termination
callbacks
stage
optimization_state
population
```
