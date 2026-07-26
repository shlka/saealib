# GP-UCB（Gaussian Process Upper Confidence Bound）

GP-UCBは、評価コストの高い目的関数を対象に、Gaussian Process回帰(GP)によるサロゲートモデルと、予測平均と予測標準偏差の線形結合である**上側信頼限界**(Upper Confidence Bound, UCB)という獲得関数を組み合わせた逐次最適化の手法です。

## 概要

GP-UCBは、多腕バンディット問題における**UCB方策**をGP最適化に拡張したものです。
バンディット問題では、各腕の報酬の信頼区間上限が最も高い腕を選び続けることで、探索と活用のバランスを自動的に取ることが知られています。

GP-UCBはこの発想を連続空間上のGP回帰に適用し、候補点 $x$ の**信頼上限** $\mu(x) + \sqrt{\beta_t}\,\sigma(x)$ を最大化する点を次に評価します。
予測平均 $\mu(x)$ が高い点は活用(exploitation)、予測標準偏差 $\sigma(x)$ が大きい点は探索(exploration)に対応し、$\beta_t$ がこの二項の相対的な重みを制御します。

この手法の理論的な核心は、$\beta_t$ を固定値ではなく反復回数 $t$ に依存する形で選ぶことにあります。
$\beta_t$ を情報利得(information gain)の上界から導かれる特定の対数的なスケジュールに従って増加させると、累積リグレットに劣線形の上界が導出できます{cite}`srinivas2012gpucb`。
具体的な手順は次の擬似コードに示します。

## 擬似コード

```{prf:algorithm} GP-UCB
:label: alg-gp-ucb

**Inputs** 目的関数 $f$（報酬として最大化）、探索範囲 $D$、GP事前分布 $\mu_0=0,\sigma_0,k$、信頼度パラメータ列 $\beta_t$
**Output** 最良解 $x^*$

1. $t=1$ とする
2. 信頼上限 $x_t = \arg\max_{x \in D} \mu_{t-1}(x) + \sqrt{\beta_t}\,\sigma_{t-1}(x)$ を最大化する点を選ぶ
3. $y_t = f(x_t) + \epsilon_t$ を観測する
4. 観測 $y_t$ でベイズ更新を行い、事後平均 $\mu_t(x)$ と事後標準偏差 $\sigma_t(x)$ を得る
5. $t$ を1増やし、評価予算に達するまで2へ戻る
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>LHS等で初期個体群を<br/>サンプリング→真の評価"] --> ASK
    subgraph GEN["1世代分 (IndividualBasedStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>候補解を生成"] --> SCORE["SurrogateManager<br/>GPをフィット (L4)<br/>→ LCBでスコアリング (L2)"]
        SCORE --> SORT["LCB上位<br/>evaluation_ratioの割合を選択<br/>（argmax UCBの近似）"]
        SORT --> EVAL["真の評価 →<br/>アーカイブに追加<br/>(L3)"]
        EVAL --> TELL["GA.tell()<br/>個体群を更新"]
    end
    GEN --> TERM{"評価予算に<br/>到達?"}
    TERM -- "未到達 (L5)" --> ASK
    TERM -- "到達" --> RESULT(["最良解 x*"])
```

## saealibでの構成

論文のAlgorithm 1は報酬の**最大化**として定式化されているのに対し、`LowerConfidenceBound`は最小化を前提に $\mathrm{LCB}(x) = \mu(x) - \kappa\sigma(x)$ を計算し、スコアの大小比較を他の獲得関数と揃えるために符号を反転して返します（`saealib`全体の規約「スコアは高いほど良い」に合わせるため）。

$\mu(x)$ を最小化空間に変換したうえで符号反転すると $-(\mu(x) - \kappa\sigma(x)) = -\mu(x) + \kappa\sigma(x)$ となり、これは最大化空間での信頼上限 $\mu(x) + \kappa\sigma(x)$ と符号の向きが揃います。
したがって`LowerConfidenceBound`の`kappa`は、論文の $\sqrt{\beta_t}$ に対応します。

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（交叉、突然変異、選択の組み合わせ自体はGP-UCBの定義に含まれない） | 候補解の生成（argmax UCBの探索） |
| サロゲートモデル | `SklearnGPRSurrogate`（GP回帰。`sklearn` extraが必要） | L4 |
| 獲得関数 | `LowerConfidenceBound`（`kappa`が論文の $\sqrt{\beta_t}$ に対応、詳細は次節） | L2 |
| サロゲート管理 | `GlobalSurrogateManager`（アーカイブ全体でGPをフィットする） | L2, L4 |
| 評価戦略 | `IndividualBasedStrategy`（UCB上位の個体だけを真に評価する） | L2-3 |

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
surrogate_manager = GlobalSurrogateManager(SklearnGPRSurrogate(), LowerConfidenceBound(kappa=2.0))
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

交叉、突然変異、選択の具体的な演算子はGP-UCB自体の定義に含まれないため、上記は一例であり任意の`Crossover`/`Mutation`/`ParentSelection`/`SurvivorSelection`に差し替えられます。

## パラメータと変種

**κ（探索と活用のトレードオフ）**：`LowerConfidenceBound(kappa=...)`で調整します。既定値は`2.0`。

論文のAlgorithm 1は、この重みを固定値ではなく反復回数 $t$ に依存する $\sqrt{\beta_t}$ として与えます。
たとえば探索空間 $D$ が有限集合の場合、$\beta_t$ は次のように選ぶことで累積リグレットの理論的な上界が導出できます（Theorem 1）。

$$\beta_t = 2 \log\left(\frac{|D|\, t^2 \pi^2}{6\delta}\right)$$

この式は $t$ について対数的に増加するため、探索の重みは反復が進むほど緩やかに大きくなります。
コンパクトな $D$ の場合（Theorem 2）や、GP事前分布を仮定せずRKHSノルムが有界な関数を扱う場合（Theorem 3）にも、それぞれ形は異なるものの、$t$ に関して増加する $\beta_t$ のスケジュールが与えられます。
GP-UCBという名前が指す理論的な貢献は、まさにこの $\beta_t$ のスケジュールと累積リグレットの上界の対応関係にあります。

`LowerConfidenceBound`の`kappa`は反復を通じて固定された定数であり、この $\sqrt{\beta_t}$ のスケジュールを実装していません。
したがって`kappa`を固定したままのGP-UCBは、論文が理論的に導出した意味でのリグレット保証を持たない、素朴な固定重みUCBヒューリスティックです。
論文自身も実験節で、Theorem 1が与える $\beta_t$ をそのまま使うと過剰に探索的になり、交差検証で係数を1/5にスケールした方が性能が良かったと報告しており、実務上は固定または経験的に調整した重みを使うこと自体は論文とも矛盾しません。
ただし、この固定重みの選び方に理論的根拠はなく、`kappa=2.0`という既定値も$\beta_t=4.0$相当の値を固定しているに過ぎません。

$t$ に応じて`kappa`を動的に変更したい場合は、`CallbackManager`で世代ごとに`surrogate_manager.acquisition.kappa`を書き換えることで、Theorem 1相当のスケジュールに近づけられます。

## 関連

- [文献リファレンス](../references.md)：出典の完全な書誌情報とLCB以外の獲得関数の出典一覧
- [SurrogateManager](../components/surrogate_manager.md)：`GlobalSurrogateManager`の詳しい使い方
- [AcquisitionFunction](../components/acquisition_functions.md)：`LowerConfidenceBound`を含む獲得関数一覧
- [Surrogate](../components/surrogate.md)：`SklearnGPRSurrogate`を含むサロゲートモデル一覧と`sklearn` extraの説明
- [OptimizationStrategy](../components/strategies.md)：`IndividualBasedStrategy`の`evaluation_ratio`を含む戦略一覧
- [EGO](ego.md)：同じGPサロゲートモデル＋`IndividualBasedStrategy`の構成を、期待改善量(EI)獲得関数で置き換えた手法
