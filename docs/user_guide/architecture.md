# アーキテクチャ

saealib のパイプラインは独立したコンポーネントで構成されており，自由に組み合わせて使用できます．

## コンポーネントパイプライン

```{mermaid}
flowchart TD
    PR[Problem] --> IN[Initializer]
    IN -- 初期集団を真の関数で評価 --> AR[(Archive)]
    AR --> AL[Algorithm]

    subgraph 世代ループ
        AL -- 候補解を生成 --> SM
        SM -- スコア付き候補解 --> OS[OptimizationStrategy]
        OS -- 有望な候補解を真の関数で評価 --> AR
    end

    subgraph SM[SurrogateManager]
        S[Surrogate] -- フィット --> AF[AcquisitionFunction]
    end

    AR --> T{Termination}
    T -- 継続 --> AL
    T -- 終了 --> R[Result]
```

## 各コンポーネントの役割

| コンポーネント | 役割 |
|---|---|
| {py:class}`~saealib.Problem` | 目的関数・変数次元・探索範囲・制約を定義する |
| {py:class}`~saealib.Initializer` | 初期集団を生成する（デフォルト: LHS） |
| {py:class}`~saealib.Algorithm` | 候補解（子個体）を生成する（GA / PSO） |
| {py:class}`~saealib.SurrogateManager` | Archive を使ってサロゲートをフィットし，候補解をスコアリングする |
| {py:class}`~saealib.Surrogate` | 目的関数を近似する機械学習モデル（デフォルト: RBF） |
| {py:class}`~saealib.AcquisitionFunction` | サロゲートの予測値から各候補解の有望度を計算する |
| {py:class}`~saealib.OptimizationStrategy` | 有望な候補解を選択し，真の関数で評価する |
| {py:class}`~saealib.Archive` | 真の評価済み解を蓄積する |
| {py:class}`~saealib.Termination` | 終了条件を判定する |
| {py:class}`~saealib.CallbackManager` | イベントフックで進捗の監視や挙動のカスタマイズを行う |

## 低レベル API

`Optimizer` を使うと各コンポーネントを個別に差し替えられます．

```python
from saealib import (
    GA, Optimizer, Problem,
    IndividualBasedStrategy,
)
from saealib.surrogate import GlobalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel
from saealib.acquisition import MeanPrediction
from saealib.termination import max_fe

problem = Problem(func, dim=5, lb=-5, ub=5, n_obj=1, direction=[-1])

opt = (
    Optimizer(problem)
    .set_algorithm(GA())
    .set_surrogate_manager(
        GlobalSurrogateManager(RBFSurrogate(gaussian_kernel, dim=5), MeanPrediction())
    )
    .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.1))
    .set_termination(max_fe(100))
)

ctx = opt.run()
```

高レベル API (`minimize` / `maximize`) は，このパイプラインを sensible defaults で自動構成したものです．

## ステージパイプライン

`OptimizationStrategy.step()` の内部実装は，{py:class}`~saealib.Stage` と {py:class}`~saealib.Pipeline` によってステップ単位で記述されています．各 `Stage` は `OptimizationState` を受け取り，更新後の状態を返す純粋な変換です．

```{mermaid}
flowchart LR
    S0[CountGenerationStage] --> S1[AskStage]
    S1 --> S2[SurrogateScoreStage]
    S2 --> S3[TopKSelectionStage]
    S3 --> S4[TrueEvaluationStage]
    S4 --> S5[ArchiveUpdateStage]
    S5 --> S6[TellStage]
```

`Pipeline` はステージのリストを `functools.reduce` で逐次実行し，`Pipeline` 自体も `Stage` であるため入れ子にできます（`GenerationBasedStrategy` の内部ループなど）．

### 組み込みステージ一覧

| クラス | 名前 (`name`) | 説明 |
|---|---|---|
| {py:class}`~saealib.CountGenerationStage` | `count_generation` | 世代カウンタを 1 増やす |
| {py:class}`~saealib.AskStage` | `ask` | `Algorithm.ask()` で候補解を生成する |
| {py:class}`~saealib.SurrogateFitStage` | `surrogate_fit` | サロゲートをアーカイブにフィットする |
| {py:class}`~saealib.SurrogateScoreStage` | `surrogate_score` | 候補解をサロゲートでスコアリングする |
| {py:class}`~saealib.TopKSelectionStage` | `top_k_selection` | スコア上位 k 個の候補解を選択する |
| {py:class}`~saealib.SortByScoreStage` | `sort_by_score` | スコア降順で全候補解をソートする |
| {py:class}`~saealib.SurrogateOnlyLoopStage` | `surrogate_only_loop` | サロゲートのみによる世代ループを実行する |
| {py:class}`~saealib.TrueEvaluationStage` | `true_evaluation` | 候補解を真の目的関数で評価する |
| {py:class}`~saealib.ArchiveUpdateStage` | `archive_update` | 評価済み解をアーカイブに追加する |
| {py:class}`~saealib.TellStage` | `tell` | `Algorithm.tell()` で個体群を更新する |
| {py:class}`~saealib.InitializationStage` | `initialization` | `Initializer` をステージとしてラップする |

### 組み込み戦略のパイプライン構成

**PreSelectionStrategy**（事前選択）

```
CountGeneration → Ask(n_candidates) → SurrogateScore
    → TopKSelection(k) → TrueEvaluation → ArchiveUpdate → Tell
```

**IndividualBasedStrategy**（個体ベース）

```
CountGeneration → Ask → SurrogateScore → SortByScore
    → TrueEvaluation(top ratio) → ArchiveUpdate → Tell
```

**GenerationBasedStrategy**（世代ベース）

```
SurrogateOnlyLoop(gen_ctrl) → CountGeneration → Ask
    → TrueEvaluation → ArchiveUpdate → Tell
```

`SurrogateOnlyLoop` の内部は `CountGeneration → Ask → SurrogateScore(refit=False) → Tell` を `gen_ctrl` 回繰り返します．

### カスタムステージ

`Stage` を継承して `execute(state)` を実装するだけでカスタムステージを作成できます．

```python
from saealib.pipeline import Pipeline, Stage
from saealib.context import OptimizationState

class LogFEStage(Stage):
    name = "log_fe"
    label = "Log function evaluations"
    notation = r"$\text{log}(fe)$"

    def execute(self, state: OptimizationState) -> OptimizationState:
        print(f"fe={state.fe}")
        return state
```

ステージを名前で検索するには `pipeline["stage_name"]` を使います．

```python
pipeline = Pipeline([CountGenerationStage(), AskStage(algorithm), ...])
ask_stage = pipeline["ask"]  # AskStage インスタンスを返す
```

### 擬似コード生成

`to_pseudocode()` で LaTeX algorithmic 形式の擬似コードを生成できます．

```python
pipeline.to_pseudocode(expand=False)  # 1 行にまとめた表現
pipeline.to_pseudocode(expand=True)   # ステージごとに展開
```

`expand=True` では `SurrogateOnlyLoopStage` が `\For ... \EndFor` ブロックに展開されます．
