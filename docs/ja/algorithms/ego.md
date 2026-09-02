---
primary_layer: layer2
related_layers: []
page_type: concept
---

# EGO（Efficient Global Optimization）

## 概要

EGOは、評価コストの高い目的関数を対象に、Gaussian Process回帰(GP)によるサロゲートモデルと**期待改善量**(Expected Improvement, EI)という獲得関数を組み合わせた逐次最適化の手法です。

GPの予測分散は学習データから離れた領域ほど大きくなります。 EIはこの予測平均と予測分散の両方から計算されるスカラー値であるため、「予測値が良さそうな領域」と「予測の不確実性が高い領域」を自然に両方カバーし、探索と活用のバランスを取ります。

出典は{cite}`jones1998ego`。具体的な手順は次の擬似コードに示します。

## 擬似コード

```{prf:algorithm} EGO
:label: alg-ego

**Inputs** 目的関数 $f$、探索範囲、初期サンプル数 $n_0$、評価予算 $N$
**Output** 最良解 $x^*$

初期個体群を $n_0$ 点サンプリングし、真の関数 $f$ で評価してアーカイブに追加する
アーカイブ全体にGPを当てはめ、任意の点における予測平均 $\mu(x)$ と予測標準偏差 $\sigma(x)$ を得る
期待改善量 $\mathrm{EI}(x) = (f_{\min} - \mu(x) - \xi)\,\Phi(z) + \sigma(x)\,\phi(z)$（$z = (f_{\min} - \mu(x) - \xi) / \sigma(x)$）を最大化する点 $x^*$ を求める
$x^*$ を真の関数で評価し、アーカイブに追加する
評価予算 $N$ に達するまで2へ戻る
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["初期化器<br/>初期個体群をサンプリング<br/>LHSなどでサンプリング→真の評価<br/>(L1)"] --> ASK
    subgraph GEN["1世代 (IndividualBasedStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>候補を生成"] --> SCORE["SurrogateManager<br/>GPをフィット (L2)<br/>→ EIでスコア化 (L3)"]
        SCORE --> SORT["上位を選択<br/>EIによるevaluation_ratioの割合<br/>(argmax EIを近似)"]
        SORT --> EVAL["真の評価 →<br/>アーカイブに追加<br/>(L4)"]
        EVAL --> TELL["GA.tell()<br/>個体群を更新"]
    end
    GEN --> TERM{"評価予算N<br/>に達したか？"}
    TERM -- "未達 (L5)" --> ASK
    TERM -- "到達" --> RESULT(["最良解x*"])
```

## saealibでの構成

|役割|saealibでの実装|対応ステップ|
|---|---|---|
|探索アルゴリズム本体|`GA`（交叉、突然変異、選択の組み合わせ自体はEGOの定義に含まれない）|L3|
|サロゲートモデル|`SklearnGPRSurrogate`（GP回帰。`sklearn` extraが必要）|L2|
|獲得関数|`ExpectedImprovement`|L3|
|サロゲート管理|`GlobalSurrogateManager`（アーカイブ全体でGPをフィットする）|L2-3|
|評価戦略|`IndividualBasedStrategy`（EI上位の個体だけを真に評価する）|L3-4|

```python
import numpy as np
from saealib import (
    GA,
    Optimizer,
    Problem,
    IndividualBasedStrategy,
    SklearnGPRSurrogate,
    ExpectedImprovement,
)
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.surrogate import GlobalSurrogateManager
from saealib.termination import Termination, max_fe


def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)


problem = Problem(sphere, dim=5, lb=[-5] * 5, ub=[5] * 5, n_obj=1, direction=[-1])

algorithm = GA(
    CrossoverBLXAlpha(prob=0.7, alpha=0.4),
    MutationUniform(prob_var=0.3),
    SequentialSelection(),
    TruncationSelection(),
)
surrogate_manager = GlobalSurrogateManager(SklearnGPRSurrogate())
strategy = IndividualBasedStrategy(evaluation_ratio=0.2)

opt = (
    Optimizer(problem)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_acquisition(ExpectedImprovement())
    .set_strategy(strategy)
    .set_termination(Termination(max_fe(200)))
)
ctx = opt.run()
```

## 原法との差分

EGOの原法は獲得関数の最大化点を一度に1点ずつ求めます。一方、saealibは`GA`が生成した候補プールからEIが最大の候補を選び、`argmax EI`を近似します。`IndividualBasedStrategy.evaluation_ratio`により、選択した複数の候補をまとめて真の評価に回すこともできます。例の交叉、突然変異、親選択、環境選択の組み合わせはsaealibの構成上の選択であり、EGOの定義の一部ではありません。

## パラメータと変種

**ξ（探索と活用のトレードオフ）**：`ExpectedImprovement(xi=...)`で調整します。既定値の`0.01`は、EGOの原提案で規定された値ではなく、実用上のヒューリスティックです。Brochu et al. (2010)は、この値が先行実験で使われたことを紹介しています{cite}`brochu2010tutorial`。$\xi=0$では活用寄りになり、大きくすると探索寄りになります。

## 関連

- [文献リファレンス](../references.md)：出典の完全な書誌情報と、EI以外の獲得関数の出典一覧
- [SurrogateManager](../concepts/surrogate_modeling/surrogate_manager.md)：`GlobalSurrogateManager`の詳しい使い方
- [AcquisitionFunction](../concepts/surrogate_modeling/acquisition_functions.md)：`ExpectedImprovement`を含む獲得関数一覧
- [Surrogate](../concepts/surrogate_modeling/surrogate.md)：`SklearnGPRSurrogate`を含むサロゲートモデルの一覧と、`sklearn` extraの説明
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md)：`IndividualBasedStrategy`の`evaluation_ratio`を含む戦略の一覧
