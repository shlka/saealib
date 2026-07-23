# MaxUnc（不確実性サンプリング）

MaxUncは、サロゲートモデルの予測不確実性（標準偏差）を評価基準とし、モデルがもっとも自信を持てない候補点を次の真の評価対象に選ぶ、探索(exploration)に特化した獲得関数です。

## 概要

EGOの期待改善量やGP-UCBの信頼上限は、予測平均 $\mu(x)$ と予測標準偏差 $\sigma(x)$ の両方を使い、探索と活用のバランスを取る設計でした。
MaxUncはこの構成から予測平均の項を完全に取り除き、$\sigma(x)$ だけを評価基準にします。

手順はEGO/GP-UCBと同型です。
アーカイブ全体にGPを当てはめ、予測標準偏差 $\sigma(x)$ を最大化する点を求め、真の関数で評価してアーカイブに追加します。
予測平均を一切参照しないため、評価対象は常に「モデルがもっとも学習データから離れている」領域に偏り、良さそうな値を積極的に探しにいく挙動にはなりません。

この構成の背景として、Büche, Schraudolph & Koumoutsakos (2005) はGPを用いたサロゲートモデルの調査論文の中で、予測平均と予測標準偏差の線形結合であるメリット関数 $f_{\mathrm{M}}(x) = \hat{t}(x) - \alpha \sigma_t(x)$ を提案し、$\alpha$ を大きくするほど探索寄りになると述べています{cite}`buche2005gpes`。
MaxUncが計算する $\sigma(x)$ 単独の基準は、このメリット関数で $\alpha \to \infty$ とした極限、つまり予測平均の寄与を消し去った場合に相当します。

MaxUncは目的関数の改善を直接狙う基準ではありません。
EIやUCBのような活用寄りの基準と対になる探索専用の構成要素として、サロゲートモデル自体の精度を全域にわたって底上げする用途や、他の基準と組み合わせて使う用途に向きます（獲得関数一覧における`MeanPrediction`との対比も参照）。
評価予算が小さいうちは、最良解への収束よりもモデルの未知領域を埋めることを優先するため、単独で使うと最終的な目的関数値がEIやLCBほど改善しないことがあります。

## 擬似コード

```{prf:algorithm} MaxUnc
:label: alg-maxunc

**Inputs** 目的関数 $f$、探索範囲、初期サンプル数 $n_0$、評価予算 $N$
**Output** 評価済みアーカイブ（真に評価した点とその関数値の集合）

1. 初期個体群を $n_0$ 点サンプリングし、真の関数 $f$ で評価してアーカイブに追加する
2. アーカイブ全体にGPを当てはめ、任意の点における予測標準偏差 $\sigma(x)$ を得る（予測平均 $\mu(x)$ は基準の計算に用いない）
3. 予測標準偏差を最大化する点 $x^* = \arg\max_x \sigma(x)$ を求める
4. $x^*$ を真の関数で評価し、アーカイブに追加する
5. 評価予算 $N$ に達するまで2へ戻る
```

<!--
参照情報（レビュー用）:

Büche, D., Schraudolph, N. N., & Koumoutsakos, P. (2005).
Accelerating evolutionary algorithms with Gaussian process fitness function models.
IEEE Transactions on Systems, Man, and Cybernetics—Part C, 35(2), 183-194.
DOI: 10.1109/TSMCC.2004.841917
OCR: .claude/exp_ref/literature/pdfs/buche2005_gp_es/auto/buche2005_gp_es.md

ページ番号の求め方: 同ディレクトリの `buche2005_gp_es_content_list.json` の `page_idx`
（0始まり）の最大値は11で、これは掲載誌の12ページ分（pp.183-194）に一致する。
したがって「掲載ページ = page_idx + 183」で対応づけた。

- ステップ1: Section IV冒頭の記述（"starts from an initial set of points, obtained,
  e.g., from previous optimization runs, by random sampling..."）、
  および Section IV.C のGPOP要約疑似コード冒頭
  ("while less than N_C/2 points evaluated successfully: ...")。
  Section IV、p.187（OCR 203-209行目）、Section IV.C、p.188（OCR 257-263行目）。
- ステップ2: 予測平均の式(10)・予測標準偏差の式(11)、Section III、p.186。
  OCR 132行目（式10）、136行目（式11）。
  GP自体の学習（ハイパーパラメータ最適化、式15-16）はSection III.A、p.186。
  OCR 160行目（式15）、164行目（式16）。
- ステップ3: **これは論文の逐語的な記述ではなく、書き手側の合成である。**
  論文が実際に提案するのはメリット関数 $f_{\mathrm{M}}(x) = \hat{t}(x) - \alpha\sigma_t(x)$
  （式(19)、Section IV.A、p.187、OCR 218行目）であり、GPOPはこれを
  $\alpha = 0, 1, 2, 4$ の4通り並列に最適化して4点を同時評価する
  （"We optimize 4 merit functions, using α = 0, 1, 2, 4"、p.187、OCR 223行目）。
  論文中に $\alpha$ を明示的に無限大にする、あるいは $\sigma$ 単独を基準とする記述は無い。
  ステップ3の $\arg\max \sigma(x)$ は、この式(19)で予測平均の重みを0にした
  （$\alpha \to \infty$ の）極限として書き手が導出したものである。
- ステップ4: GPOP要約疑似コード中 "evaluate new optima on expensive fitness function"、
  Section IV.C、p.188。OCR 277-281行目。
- ステップ5: GPOP要約疑似コード全体を統べる
  "while termination criterion not reached:" ループ、Section IV.C、p.188。OCR 265行目。

既知の課題（レビュー用・非表示、本文には出さない）:
`.claude/exp_ref/literature/topic_notes/named_algorithms_component_map.md`はMaxUnc samplingを✅Directと分類しているが、
不正確。論文が実際に提案するのは予測平均$\hat{t}(x)$と予測標準偏差$\sigma_t(x)$を組み合わせた
メリット関数$f_{\mathrm{M}}(x) = \hat{t}(x) - \alpha\sigma_t(x)$（式(19)、$\alpha=0,1,2,4$の
4通り並列評価）であり、`saealib`の`MaxUncertainty`が計算する$\sigma(x)$単独の基準は、この式で
$\alpha\to\infty$とした極限に相当する。論文中に$\alpha$を無限大にする、または$\sigma$単独を
基準とする記述はない。ただしSPEA2/NSGA-III/RBF-CORSの各ケースと異なり、これは「実装すべき
手続きが欠けている」というより「単一の獲得関数に単一の論文を紐づける際の極限近似」という性質が強く、
コード側に直接の修正対象があるとは限らない。GitHub Projectsへの起票は行っていない
（要検討としてユーザーへ報告する）。
-->

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>LHS等で初期個体群を<br/>サンプリング→真の評価<br/>(L1)"] --> ASK
    subgraph GEN["1世代分 (IndividualBasedStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>候補解を生成"] --> SCORE["SurrogateManager<br/>GPをフィット (L2)<br/>→ σでスコアリング (L3)"]
        SCORE --> SORT["σ上位<br/>evaluation_ratio割を選択<br/>（argmax σの近似）"]
        SORT --> EVAL["真の評価 →<br/>アーカイブに追加<br/>(L4)"]
        EVAL --> TELL["GA.tell()<br/>個体群を更新"]
    end
    GEN --> TERM{"評価予算Nに<br/>到達?"}
    TERM -- "未到達 (L5)" --> ASK
    TERM -- "到達" --> RESULT(["評価済みアーカイブ"])
```

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（交叉・突然変異・選択の組み合わせ自体はMaxUncの定義に含まれない） | 候補解の生成（argmax σの探索） |
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
surrogate_manager = GlobalSurrogateManager(SklearnGPRSurrogate(), MaxUncertainty())
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

交叉・突然変異・選択の具体的な演算子はMaxUnc自体の定義に含まれないため、上記は一例であり任意の`Crossover`/`Mutation`/`ParentSelection`/`SurvivorSelection`に差し替えられます。

この例のように評価予算をEGO/GP-UCBの例と同じ200 FEに揃えて実行すると、探索専用であるがゆえに最良値がEI/LCBほど改善しない場合があります。
これはMaxUncが目的関数の改善ではなくモデルの不確実性削減を目的とする基準であることの自然な帰結であり、想定どおりの挙動です。

## パラメータと変種

**weights（多目的での不確実性の集約方法）**: `MaxUncertainty(weights=...)`で調整します。
多目的問題では各目的ごとに予測標準偏差 $\sigma_1(x), \ldots, \sigma_m(x)$ が得られるため、それらを1つのスコアに集約する必要があります。
`weights`が`None`の既定値では目的間の単純平均（`std.mean(axis=1)`）を、`np.ndarray`を渡した場合はその重みによる加重和を使います。

EIの$\xi$やLCBの$\kappa$に相当する、探索と活用のトレードオフを調整するパラメータはMaxUncertainty自体には存在しません。
$\sigma(x)$のみを基準とする設計上、活用側の重みを持たないためです。
探索と活用の重みを連続的に調整したい場合は、[GP-UCB](gp_ucb.md)の`LowerConfidenceBound(kappa=...)`を使い、`kappa`を大きくする方向で近づけることになります。

## 関連

- [文献リファレンス](../references.md) — 出典の完全な書誌情報
- [SurrogateManager](../components/surrogate_manager.md) — `GlobalSurrogateManager`の詳しい使い方
- [AcquisitionFunction](../components/acquisition_functions.md) — `MaxUncertainty`を含む獲得関数一覧
- [Surrogate](../components/surrogate.md) — `SklearnGPRSurrogate`を含むサロゲートモデル一覧と`sklearn` extraの説明
- [OptimizationStrategy](../components/strategies.md) — `IndividualBasedStrategy`の`evaluation_ratio`を含む戦略一覧
- [EGO](ego.md) — 同じGPサロゲートモデル＋`IndividualBasedStrategy`の構成を、活用寄りの期待改善量(EI)獲得関数で置き換えた手法
- [GP-UCB](gp_ucb.md) — Büche et al.のメリット関数と同じ$\mu - \kappa\sigma$の構造を持つ`LowerConfidenceBound`獲得関数を使う手法
