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
    GlobalSurrogateManager, RBFsurrogate, ExpectedImprovement,
    IndividualBasedStrategy,
)
from saealib.termination import max_fe

problem = Problem(func, dim=5, lb=-5, ub=5, n_obj=1, weight=-1)

opt = (
    Optimizer(problem)
    .set_algorithm(GA())
    .set_surrogate_manager(
        GlobalSurrogateManager(RBFsurrogate(), ExpectedImprovement())
    )
    .set_strategy(IndividualBasedStrategy(n_eval=1))
    .set_termination(max_fe(100))
)

ctx = opt.run()
```

高レベル API (`minimize` / `maximize`) は，このパイプラインを sensible defaults で自動構成したものです．
