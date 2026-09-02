---
primary_layer: layer2
related_layers: []
page_type: concept
---

# GP-UCB (Gaussian Process Upper Confidence Bound)

GP-UCBは、評価コストの高い目的関数を対象に、ガウス過程（GP）回帰によるサロゲートモデルと、予測平均と予測標準偏差の線形結合である**上側信頼限界**（UCB）という獲得関数を組み合わせた逐次最適化手法です。

## 概要

GP-UCBは、多腕バンディット問題における**UCB方策**をGP最適化に拡張したものです。
バンディット問題では、各腕の報酬の信頼区間上限が最も高い腕を選び続けることで、探索と活用のバランスを自動的に取ることが知られています。

GP-UCBはこの発想を連続空間上のGP回帰に適用し、候補点 $x$ の**信頼上限** $\mu(x) + \sqrt{\beta_t}\,\sigma(x)$ を最大化する点を次に評価します。
予測平均 $\mu(x)$ が高い点は活用、予測標準偏差 $\sigma(x)$ が大きい点は探索に対応し、$\beta_t$ がこの二項の相対的な重みを制御します。

この手法の理論的核心は、$\beta_t$を固定値ではなく反復回数$t$の関数として選ぶことです。
情報利得の上界から導いた特定の対数スケジュールに従って$\beta_t$を増加させると、累積リグレットに対する劣線形の上界が得られます。具体的な手順を次の擬似コードに示します。{cite}`srinivas2012gpucb`

## 擬似コード

```{prf:algorithm} GP-UCB
:label: alg-gp-ucb

**Inputs** objective function $f$ (maximized as a reward), search domain $D$, GP prior $\mu_0=0,\sigma_0,k$, confidence parameter sequence $\beta_t$
**Output** best solution $x^*$

1. Set $t=1$
2. Choose the point $x_t = \arg\max_{x \in D} \mu_{t-1}(x) + \sqrt{\beta_t}\,\sigma_{t-1}(x)$ that maximizes the upper confidence bound
3. Observe $y_t = f(x_t) + \epsilon_t$
4. Perform a Bayesian update with the observation $y_t$, obtaining the posterior mean $\mu_t(x)$ and posterior standard deviation $\sigma_t(x)$
5. Increment $t$ by 1 and return to step 2 until the evaluation budget is reached
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Sample initial population<br/>via LHS etc. → true evaluation"] --> ASK
    subgraph GEN["One generation (IndividualBasedStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>Generate candidates"] --> SCORE["SurrogateManager<br/>Fit GP (L4)<br/>→ Score with LCB (L2)"]
        SCORE --> SORT["Select top<br/>evaluation_ratio fraction by LCB<br/>(approximates argmax UCB)"]
        SORT --> EVAL["True evaluation →<br/>add to archive<br/>(L3)"]
        EVAL --> TELL["GA.tell()<br/>Update population"]
    end
    GEN --> TERM{"Evaluation budget<br/>reached?"}
    TERM -- "Not yet (L5)" --> ASK
    TERM -- "Reached" --> RESULT(["Best solution x*"])
```

## saealibでの構成

引用文献のGP-UCBは報酬の**最大化**として定式化されているのに対し、`LowerConfidenceBound`は最小化を前提とし、$\mathrm{LCB}(x) = \mu(x) - \kappa\sigma(x)$を計算します。スコアの比較を他の獲得関数と揃えるため、符号を反転して返します（saealib全体の「スコアは高いほど良い」という規約に合わせるためです）。

$\mu(x)$ を最小化空間に変換したうえで符号反転すると $-(\mu(x) - \kappa\sigma(x)) = -\mu(x) + \kappa\sigma(x)$ となり、これは最大化空間での信頼上限 $\mu(x) + \kappa\sigma(x)$ と符号の向きが揃います。
したがって`LowerConfidenceBound`の`kappa`は、論文の $\sqrt{\beta_t}$ に対応します。

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（交叉、突然変異、選択の組み合わせ自体はGP-UCBの定義に含まれない） | 手順 L2 |
| サロゲートモデル | `SklearnGPRSurrogate`（GP回帰。`sklearn` extraが必要） | 手順 L4 |
| 獲得関数 | `LowerConfidenceBound`（`kappa`が論文の $\sqrt{\beta_t}$ に対応、詳細は次節） | L2 |
| サロゲート管理 | `GlobalSurrogateManager`（アーカイブ全体でGPをフィットする） | 「L2, L4」 |
| 評価戦略 | `IndividualBasedStrategy`（UCB上位の個体だけを真に評価する） | 「L2-3」 |

```python
import numpy as np
from saealib import (
    GA,
    Optimizer,
    Problem,
    IndividualBasedStrategy,
    SklearnGPRSurrogate,
    LowerConfidenceBound,
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
    .set_acquisition(LowerConfidenceBound(kappa=2.0))
    .set_strategy(strategy)
    .set_termination(Termination(max_fe(200)))
)
ctx = opt.run()
```

## 文献との差分

GP-UCBの原法は反復ごとにUCB最大点を求めます。saealibは`GA`が生成した候補プールからUCB上位候補を選ぶことで、この`argmax UCB`を近似します。`IndividualBasedStrategy.evaluation_ratio`は、選ばれた候補の複数個をまとめて真の評価に回します。

有限集合$D$の場合、原法は累積リグレットの理論的上限を得るために、反復回数$t$に対して対数的に増加する次の$\beta_t$を選びます。

$$\beta_t = 2 \log\left(\frac{|D|\, t^2 \pi^2}{6\delta}\right)$$

原論文は、理論どおりのスケジュールをそのまま使うと探索的すぎる場合があるとして、係数を交差検証で1/5に縮小した方がよい結果を報告しています。
saealibの既定値（`kappa=2.0`、$t$によらず固定）はこのスケジュールに従わず、この保証を持ちません。

## パラメータと変種

**κ（探索と活用のトレードオフ）**：`LowerConfidenceBound(kappa=...)`で調整します。既定値は`2.0`。

**β_tスケジュール**：固定`kappa`の代わりに`beta_schedule=gp_ucb_beta_schedule(domain_size, delta)`を渡すと、「文献との差分」節の$β_t$の式を再現します。`domain_size`は有限の意思決定集合の要素数$|D|$であり、saealibの連続探索空間に対する離散化点数の代理値ではありません（以下を参照）。
`t = ctx.decision_count + 1`は、ランタイムがこれまでに（同期または非同期で）確定した新規の評価計画の数に、これから下す決定の分の1を加えた値です。
確定した評価計画1件は、それが評価する候補数によらず1決定として数えます。そのため、このページの例にある`GA`＋`evaluation_ratio`構成では、`evaluation_ratio`が世代ごとに複数候補を選ぶ場合、`t`は論文の観測ごとの反復回数とは一致しません。
`t`が論文の反復回数と一致するのは、確定した評価計画がすべて1候補だけを評価する場合（同期実行、1決定につき1回の真の評価）に限られますが、`t`が一致するだけでは引用文献のリグレット上限は再現されません。
定理1は、実際の有限集合$D$上で$\mu_{t-1}(x) + \sqrt{\beta_t}\,\sigma_{t-1}(x)$を網羅的に最適化することを仮定しています。saealibの`GA`が生成する候補プールは、各世代で再サンプリングされた部分集合上でこのargmaxを近似するだけです。また、`gp_ucb_beta_schedule(domain_size=...)`に整数を渡しても連続探索空間がその有限集合$D$になるわけではなく、式に数値を代入するだけです。
定理の保証を再現するには、文字どおりの有限集合$D$上で網羅的に最適化するコンポーネント（saealib組み込みの`GA`/`PSO`アルゴリズムには含まれません）と、論文のノイズおよびカーネルの仮定に一致するGPモデルが必要です。saealibはそのようなコンポーネントを提供していません。
したがって、以下の表は各設定における`t`と`kappa`の挙動だけを示すものであり、引用文献のリグレット上限が成り立つかどうかは示しません。

| 設定 | 使用する`kappa` | 論文の反復回数に対する`t` |
|---|---|---|
| `beta_schedule=None`（既定） | `2.0`固定 | 該当なし（スケジュールなし、リグレット保証もまったくなし） |
| `beta_schedule=gp_ucb_beta_schedule(domain_size, delta)`、バッチ決定 | 上記スケジュールに基づく$\sqrt{\beta_t}$ | `t`が観測数を過小に数える |
| `beta_schedule=gp_ucb_beta_schedule(domain_size, delta)`、1決定1候補 | 上記スケジュールに基づく$\sqrt{\beta_t}$ | 一致 |

## 関連

- [文献リファレンス](../references.md)：出典の完全な書誌情報とLCB以外の獲得関数の出典一覧
- [SurrogateManager](../concepts/surrogate_modeling/surrogate_manager.md)：`GlobalSurrogateManager`の詳しい使い方
- [AcquisitionFunction](../concepts/surrogate_modeling/acquisition_functions.md): `LowerConfidenceBound`を含む獲得関数の一覧
- [Surrogate](../concepts/surrogate_modeling/surrogate.md)：`SklearnGPRSurrogate`を含むサロゲートモデルの一覧と、`sklearn`追加機能の説明
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md): `IndividualBasedStrategy`の`evaluation_ratio`を含む戦略の一覧
- [EGO](ego.md)：同じGPサロゲートモデル＋`IndividualBasedStrategy`の構成を、期待改善量(EI)獲得関数で置き換えた手法
