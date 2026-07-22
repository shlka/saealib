# EGO（Efficient Global Optimization）

## 概要

EGOは、評価コストの高い目的関数を対象に、Gaussian Process回帰(GP)によるサロゲートモデルと**期待改善量**(Expected Improvement, EI)という獲得関数を組み合わせた逐次最適化の手法です。

GPの予測分散は学習データから離れた領域ほど大きくなります。
EIはこの予測平均と予測分散の両方から計算されるスカラー値であるため、「予測値が良さそうな領域」と「予測の不確実性が高い領域」を自然に両方カバーし、探索(exploration)と活用(exploitation)のバランスを取ります。

出典は{cite}`jones1998ego`。具体的な手順は次の擬似コードに示します。

<!--
参照情報（レビュー用）:
Jones, D. R., Schonlau, M., & Welch, W. J. (1998).
Efficient global optimization of expensive black-box functions.
Journal of Global Optimization, 13(4), 455-492.
DOI: 10.1023/A:1008306431147
OCR: exp_ref/pdfs_ea_saea/user_provided/refs_EA_SAEA/jones1998_ego/jones1998_ego/auto/jones1998_ego.md

Brochu, E., Cora, V. M., & de Freitas, N. (2010).
A tutorial on Bayesian optimization of expensive cost functions...
arXiv:1012.2599
OCR: exp_ref/pdfs/brochu_2010_bo_tutorial/auto/brochu_2010_bo_tutorial.md

- ステップ1: Jones et al. Section 4.2 "THE EGO ALGORITHM"、p.473
  （space-filling designで約10k点をサンプリング）。OCR 303-305行目。
- ステップ2: DACE予測子 ŷ(x) の式(7)、Section 2、p.461。OCR 116行目。
  標準誤差 s(x) の式(9)、Section 2、p.462。OCR 136行目。
  （ここでのμ(x)/σ(x)はŷ(x)/s(x)に対応する言い換え）
- ステップ3の期待改善量の閉形式（ξを含まない）:
  Jones et al. 式(15)、Section 4.1、p.471。OCR 271-273行目。
  ξ（探索と活用のトレードオフ項）: Jones et al.の式(15)自体には存在しない。
  Brochu et al. の Probability of Improvement 式(2)（OCR中の"page 12"、
  508-513行目）が導入する trade-off parameter ξ を、EIにも適用する慣行に
  基づく（Brochu自身のEI式(3)（"page 13"、611-616行目）にもξは現れない）。
  この部分は逐語的な引用ではなく、書き手側の合成。
- ステップ4-5: Jones et al. Section 4.2、p.473（反復手順の記述）。
  OCR 309行目。
-->

## 擬似コード

```{prf:algorithm} EGO
:label: alg-ego

**Inputs** 目的関数 $f$、探索範囲、初期サンプル数 $n_0$、評価予算 $N$
**Output** 最良解 $x^*$

1. 初期集団を $n_0$ 点サンプリングし、真の関数 $f$ で評価してアーカイブに追加する
2. アーカイブ全体にGPを当てはめ、任意の点における予測平均 $\mu(x)$ と予測標準偏差 $\sigma(x)$ を得る
3. 期待改善量 $\mathrm{EI}(x) = (f_{\min} - \mu(x) - \xi)\,\Phi(z) + \sigma(x)\,\phi(z)$（$z = (f_{\min} - \mu(x) - \xi) / \sigma(x)$）を最大化する点 $x^*$ を求める
4. $x^*$ を真の関数で評価し、アーカイブに追加する
5. 評価予算 $N$ に達するまで2へ戻る
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>LHS等で初期集団を<br/>サンプリング→真の評価<br/>(L1)"] --> ASK
    subgraph GEN["1世代分 (IndividualBasedStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>候補解を生成"] --> SCORE["SurrogateManager<br/>GPをフィット (L2)<br/>→ EIでスコアリング (L3)"]
        SCORE --> SORT["EI上位<br/>evaluation_ratio割を選択<br/>（argmax EIの近似）"]
        SORT --> EVAL["真の評価 →<br/>アーカイブに追加<br/>(L4)"]
        EVAL --> TELL["GA.tell()<br/>母集団を更新"]
    end
    GEN --> TERM{"評価予算Nに<br/>到達?"}
    TERM -- "未到達 (L5)" --> ASK
    TERM -- "到達" --> RESULT(["最良解 x*"])
```

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（交叉・突然変異・選択の組み合わせ自体はEGOの定義に含まれない） | 候補解の生成（argmax EIの探索） |
| サロゲートモデル | `SklearnGPRSurrogate`（GP回帰。`sklearn` extraが必要） | L2 |
| 獲得関数 | `ExpectedImprovement` | L3 |
| サロゲート管理 | `GlobalSurrogateManager`（アーカイブ全体でGPをフィットする） | L2-3 |
| 評価戦略 | `IndividualBasedStrategy`（EI上位の個体だけを真に評価する） | L3-4 |

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
surrogate_manager = GlobalSurrogateManager(SklearnGPRSurrogate(), ExpectedImprovement())
strategy = IndividualBasedStrategy(evaluation_ratio=0.2)

opt = (
    Optimizer(problem)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_strategy(strategy)
    .set_termination(Termination(max_fe(200)))
)
ctx = opt.run()
```

交叉・突然変異・選択の具体的な演算子はEGO自体の定義に含まれないため、上記は一例であり任意の`Crossover`/`Mutation`/`ParentSelection`/`SurvivorSelection`に差し替えられます。

## パラメータと変種

**ξ（探索と活用のトレードオフ）**: `ExpectedImprovement(xi=...)`で調整します。
既定値は`0.01`で、Brochu et al. (2010)が推奨する値に基づきます{cite}`brochu2010tutorial`。
$\xi=0$では活用(exploitation)寄りになり、大きくすると探索(exploration)寄りになります。

**evaluation_ratio（逐次評価とバッチ評価の切り替え）**: 文献のEGOは、擬似コードのステップ3-4を1回のループにつき1点だけ実行する逐次アルゴリズムです。
saealibの`IndividualBasedStrategy`は、GAが生成した子個体群のうちEIスコア上位`evaluation_ratio`割をまとめて真評価するバッチ拡張になっています。
`evaluation_ratio`を個体数の逆数程度まで小さくすれば、1点ずつの逐次評価に近づきます。

## 関連

- [文献リファレンス](../references.md) — 出典の完全な書誌情報と、EI以外の獲得関数の出典一覧
- [SurrogateManager](../components/surrogate_manager.md) — `GlobalSurrogateManager`の詳しい使い方
- [AcquisitionFunction](../components/acquisition_functions.md) — `ExpectedImprovement`を含む獲得関数一覧
- [Surrogate](../components/surrogate.md) — `SklearnGPRSurrogate`を含むサロゲートモデル一覧と`sklearn` extraの説明
- [OptimizationStrategy](../components/strategies.md) — `IndividualBasedStrategy`の`evaluation_ratio`を含む戦略一覧
