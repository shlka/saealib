---
primary_layer: layer2
related_layers: []
page_type: concept
---

# MaxUnc（不確実性サンプリング）

MaxUncは、サロゲートモデルの予測不確実性（標準偏差）を基準とし、モデルが最も自信を持てない候補点を次の真の評価対象に選ぶ、探索専用の獲得関数です。

MaxUncは、固有の名称を持つ独立した原法アルゴリズムというより、Bücheらが示したメリット関数で$\alpha \to \infty$とした極限として理解する方が適切です。

## 概要

EGOの期待改善量とGP-UCBの上側信頼限界は、予測平均 $\mu(x)$ と予測標準偏差 $\sigma(x)$ の両方を使って探索と活用のバランスを取ります。MaxUncはこの構成から予測平均の項を完全に取り除き、$\sigma(x)$だけを基準にします。

手続きは構造的にEGO/GP-UCBと同一です。アーカイブ全体にGPを適合させ、予測標準偏差$\sigma(x)$を最大化する点を見つけ、真の関数で評価してアーカイブに追加します。予測平均を使わないため、学習データから遠い領域を優先すると直感的に理解できます。ただし、不確実性の形状はGPのカーネルやノイズなどに依存するため、どのGPでも単純に最遠点を特定するわけではありません。

この構成の背景として、Büche, Schraudolph & Koumoutsakos (2005)はGPを用いたサロゲートモデルの調査論文で、予測平均と予測標準偏差の線形結合であるメリット関数 $f_{\mathrm{M}}(x) = \hat{t}(x) - \alpha \sigma_t(x)$ を提案し、$\alpha$を大きくするほど探索寄りになると述べています{cite}`buche2005gpes`。MaxUncが計算する$\sigma(x)$単独の基準は、このメリット関数で$\alpha \to \infty$とした極限、つまり予測平均の寄与を消し去った場合に相当します。

MaxUncは目的関数の改善を直接狙う基準ではありません。EIやUCBのような活用寄りの基準と対になる探索専用の構成要素として、サロゲートモデル自体の精度を全域にわたって底上げする用途や、他の基準と組み合わせて使う用途に向きます（獲得関数一覧における`MeanPrediction`との対比も参照）。評価予算が小さいうちは、最良解への収束よりもモデルの未知領域を埋めることを優先するため、単独で使うと最終的な目的関数値がEIやLCBほど改善しないことがあります。

## 擬似コード

```{prf:algorithm} MaxUnc
:label: alg-maxunc

**Inputs** 目的関数 $f$、探索範囲、初期サンプル数 $n_0$、評価予算 $N$
**Output** 評価済みアーカイブ（真に評価した点とその関数値の集合）

1. 初期個体群を $n_0$ 点サンプリングし、真の関数 $f$ で評価してアーカイブに追加する
2. アーカイブ全体にGPを当てはめ、任意の点の予測標準偏差 $\sigma(x)$ を得る（予測平均 $\mu(x)$ は基準の計算に用いない）
3. 予測標準偏差を最大化する点 $x^* = \arg\max_x \sigma(x)$ を求める
4. $x^*$ を真の関数で評価し、アーカイブに追加する
5. 評価予算 $N$ に達するまで2へ戻る
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Sample initial population<br/>via LHS etc. → true evaluation<br/>(L1)"] --> ASK
    subgraph GEN["One generation (IndividualBasedStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>Generate candidates"] --> SCORE["SurrogateManager<br/>Fit GP (L2)<br/>→ Score with σ (L3)"]
        SCORE --> SORT["Select top<br/>evaluation_ratio fraction by σ<br/>(approximates argmax σ)"]
        SORT --> EVAL["True evaluation →<br/>add to archive<br/>(L4)"]
        EVAL --> TELL["GA.tell()<br/>Update population"]
    end
    GEN --> TERM{"Evaluation budget N<br/>reached?"}
    TERM -- "Not yet (L5)" --> ASK
    TERM -- "Reached" --> RESULT(["Evaluated archive"])
```

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（交叉、突然変異、選択の組み合わせ自体はMaxUncの定義に含まれない） | L3 |
| サロゲートモデル | `SklearnGPRSurrogate`（GP回帰。`sklearn` extraが必要） | L2 |
| 獲得関数 | `MaxUncertainty`（予測標準偏差のみでスコアリングし、予測平均は参照しない） | L3 |
| サロゲート管理 | `GlobalSurrogateManager`（アーカイブ全体でGPをフィットする） | L2-3 |
| 評価戦略 | `IndividualBasedStrategy`（σ上位の個体だけを真に評価する） | L3-4 |

```python
import numpy as np
from saealib import (
    GA,
    Optimizer,
    Problem,
    IndividualBasedStrategy,
    SklearnGPRSurrogate,
    MaxUncertainty,
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
    .set_acquisition(MaxUncertainty())
    .set_strategy(strategy)
    .set_termination(Termination(max_fe(200)))
)
ctx = opt.run()
```

この例をEGO/GP-UCBの例と同じ200-FE評価予算で実行すると、探索専用であるため最良値がEI/LCBほど改善しない場合があります。これは目的関数の改善ではなくモデルの不確実性削減を目的とする基準の自然な帰結で、想定どおりの挙動です。

## 文献との差分

MaxUncは`GA`が生成した候補プールから候補を選ぶため、saealibでは`argmax σ`を近似します。`IndividualBasedStrategy.evaluation_ratio`により上位候補をまとめて真の評価に回せます。例の交叉、突然変異、親選択、環境選択の組み合わせはsaealibの構成例であり、MaxUncの定義には含まれません。

## パラメータと変種

**weights（複数目的にわたる不確実性の集約）**：`MaxUncertainty(weights=...)`で調整します。多目的問題では各目的の予測標準偏差$\sigma_1(x), \ldots, \sigma_m(x)$を単一のスコアに集約します。既定値`weights=None`では目的間の単純平均（`std.mean(axis=1)`）を使い、`np.ndarray`を渡すとその重みによる加重和を使います。

`MaxUncertainty`には探索と活用のトレードオフを調整するEIの$\xi$やLCBの$\kappa$に相当するパラメータがありません。$\sigma(x)$だけを基準に使う設計のため、活用側の重みを持たないからです。重みを連続的に調整するには[GP-UCB](gp_ucb.md)の`LowerConfidenceBound(kappa=...)`を使い、`kappa`を大きくする方向へ動かします。

## 関連

- [文献リファレンス](../references.md)：出典の完全な書誌情報
- [SurrogateManager](../concepts/surrogate_modeling/surrogate_manager.md)：`GlobalSurrogateManager`の詳しい使い方
- [AcquisitionFunction](../concepts/surrogate_modeling/acquisition_functions.md)：`MaxUncertainty`を含む獲得関数の一覧
- [Surrogate](../concepts/surrogate_modeling/surrogate.md)：`SklearnGPRSurrogate`を含むサロゲートモデルの一覧と`sklearn`追加機能の説明
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md)：`IndividualBasedStrategy`の`evaluation_ratio`を含む戦略の一覧
- [EGO](ego.md)：同じGPサロゲートモデルと`IndividualBasedStrategy`の構成で、獲得関数を活用寄りの期待改善量（EI）に置き換えた手法
- [GP-UCB](gp_ucb.md)：Bücheらのメリット関数と同じ$\mu - \kappa\sigma$構造を持つ`LowerConfidenceBound`獲得関数を使う手法
