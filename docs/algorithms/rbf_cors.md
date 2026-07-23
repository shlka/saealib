# CORS-RBF（Constrained Optimization using Response Surfaces）

CORS-RBFは、評価コストの高い目的関数を対象に、RBF(Radial Basis Function)補間によるサロゲートモデルを使って次の評価点を1点ずつ選ぶ逐次最適化の手法です。
Regis & Shoemaker (2005)が提案した枠組みCORS(Constrained Optimization using Response Surfaces)を、RBFサロゲートモデルで具体化した実装がCORS-RBFです。

## 概要

RBF補間は、学習点を通る滑らかな曲面を再構成するだけで、GP回帰のような予測分散を持ちません。
そのため、サロゲートモデルの予測値をそのまま最小化して次の評価点を選ぶ素朴な方法は、既に良い値が観測された点の周辺だけを繰り返し探索してしまい、真の関数の局所最小値ですらない点に収束しかねません。

CORSはこの問題を、候補点選択そのものに**距離制約**を組み込むことで回避します。
各反復で解く補助問題は、サロゲートモデル $\hat f_i(x)$ の最小化に加えて、次の候補点が既存の評価済み点すべてから距離 $\beta_i \Delta_i$ 以上離れていることを要求する制約付き最適化になります（$\Delta_i$は既存点集合からの最大最小距離）。
$\beta_i$は反復ごとに、1に近い値（大域探索寄り）から0（局所探索寄り、つまりサロゲートモデルの単純な最小化）まで周期的に変化する数列(**search pattern**)として与えられ、GPの予測分散が担っていた探索(exploration)の役割をこの距離制約が肩代わりします。

この距離制約は副産物ではなく、CORSの核心です。
search patternに0でない値が1つでも含まれていれば、サロゲートモデルの種類や初期評価点の選び方によらず、任意の連続関数の大域最小値に収束することが証明されています。

出典は{cite}`regis2005cors`。具体的な手順は次の擬似コードに示します。

<!--
参照情報（レビュー用）:
Regis, R. G., & Shoemaker, C. A. (2005).
Constrained global optimization of expensive black box functions using
radial basis functions.
Journal of Global Optimization, 31(1), 153-171.
DOI: 10.1007/s10898-004-0570-0
OCR: .claude/exp_ref/literature/pdfs/regis2005_cors/auto/regis2005_cors.md
（OCRファイル内のpage_idx（0始まり）に153を足すと実際の掲載ページ番号になることを、
_content_list.jsonのpage_number要素と突き合わせて確認済み: page_idx1→p.154, 3→p.156,
4→p.157, 5→p.158, 9→p.162, 10→p.163, 11→p.164）。

- 概要の「素朴な最小化は局所最小値ですらない点に収束しかねない」という記述:
  Section 1、p.155（"a naive implementation of these methods, where the global
  minimizer of the current approximating surface is always selected for
  function evaluation may converge to some point which may not even be a
  local minimizer of the actual function (Gutmann, 2001b; Jones, 2001a)"）。
  OCR 27行目。Gutmann/Jonesは孫引きで、本ページでは{cite}引用しない。
- 擬似コードステップ1-2: Step 1, Step 2, Step 3.1（Section 2.1）、p.157。
  OCR 45-51行目。
- 擬似コードステップ3: Step 3.2の制約付き最小化問題、式(1)。p.157。OCR 53-65行目。
  $\Delta_i$の定義（式(2)）はp.158。OCR 70行目。
- 擬似コードステップ4: Step 3.3, Step 3.4。p.158。OCR 75-77行目。
- 擬似コードステップ5: search patternの周期構造（$\beta_i=\beta_{i+N+1}$、
  $1\geqslant\beta_1\geqslant\cdots\geqslant\beta_{N+1}=0$）。p.158。OCR 83行目。
- 収束性（「search patternに0でない値が1つでも含まれていれば...」の要約）:
  Theorem 2、Corollary 3、Section 3、p.159-160。OCR 109-139行目。
- RBF補間モデル自体（$s(x)=\sum\lambda_i\phi(\|x-x_i\|)+p(x)$、式(6)）:
  Section 4.2、p.163。OCR 158-162行目。
- 実験で使われたカーネル・多項式項（thin plate spline + 1次多項式 $p(x)$）:
  Section 4.3、p.164（"we used a particular radial basis function model of
  the form (6) where φ is a thin plate spline and p(x) is a linear
  polynomial"）。OCR 216行目。

逐語訳ではなく、Step 1-3.4の枠組み記述（Section 2.1）を1つの擬似コードに要約したもの。
変数名（$S_i$, $\hat f_i$, $\Delta_i$, $\beta_i$）は論文の記法をそのまま保持している。

既知の課題（レビュー用・非表示、本文には出さない）:
擬似コードステップ3のCORS補助問題（距離制約付き最小化）が未実装で、saealibの`MeanPrediction`は
予測平均$\hat f_i(x)$をそのままスコアとして返すだけで、距離制約$\|x-x_j\|\geqslant\beta_i\Delta_i$も
$\beta_i$の周期的切り替えも持たない。これは論文が「純粋な貪欲探索(search patternが$\langle 0\rangle$の
特殊ケース)」と呼び、"is prone to prematurely converging to a point that may not even be a local
minimizer"（Section 2.1、p.155/157）と明言して推奨しない構成そのものに相当する。距離制約こそが
論文の大域収束性の証明（Theorem 2、Corollary 3、Section 3、p.159-160）を支えているため、
NSGA-III/SPEA2の既知の課題と同様に実装上の不足として扱う。修正候補: `MeanPrediction`とは別に
距離制約付き獲得関数（またはIndividualBasedStrategy側でのペナルティ）を追加する。
`.claude/exp_ref/literature/topic_notes/named_algorithms_component_map.md`の「RBF-EGO / CORS」行（✅Direct表記）も
この点を反映しておらず要更新。GitHub Projectsへの起票はまだ行っていない。
-->

## 擬似コード

```{prf:algorithm} CORS-RBF
:label: alg-rbf-cors

**Inputs** 目的関数 $f$、探索領域 $\mathcal{D}$、初期評価点集合 $S_1 = \{x_1, \ldots, x_k\}$、距離パラメータの周期列(search pattern) $\langle \beta_1, \ldots, \beta_{N+1}=0 \rangle$
**Output** 最良解 $x^*$

1. $S_1$ を真の関数 $f$ で評価し、$i := 1$ とする
2. これまでの評価済みデータ $D_i = \{(x, f(x)) \mid x \in S_i\}$ にRBFサロゲートモデル $\hat f_i$ をフィットする
3. 制約付き最小化問題 $\min_{x \in \mathcal{D}} \hat f_i(x) \ \mathrm{s.t.} \ \|x - x_j\| \geqslant \beta_i \Delta_i \ (j=1,\ldots,|S_i|)$ を解いて候補点 $x_{k+i}$ を求める（$\Delta_i$は既存評価点集合からの最大最小距離）
4. $x_{k+i}$ を真の関数で評価し、$S_{i+1} := S_i \cup \{x_{k+i}\}$ に追加する
5. 終了条件に達するまで、周期列に従い $\beta_i$ を更新して $i := i+1$ とし2へ戻る
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>初期個体群をサンプリング<br/>→真の評価<br/>(L1)"] --> ASK
    subgraph GEN["1世代分 (IndividualBasedStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>候補点を生成"] --> SCORE["SurrogateManager<br/>RBFをフィット (L2)<br/>→ 予測平均でスコアリング<br/>(L3)"]
        SCORE --> SORT["予測平均上位<br/>evaluation_ratio割を選択"]
        SORT --> EVAL["真の評価 →<br/>アーカイブに追加<br/>(L4)"]
        EVAL --> TELL["GA.tell()<br/>個体群を更新"]
    end
    GEN --> TERM{"評価予算Nに<br/>到達?"}
    TERM -- "未到達 (L5)" --> ASK
    TERM -- "到達" --> RESULT(["最良解 x*"])
```

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（交叉・突然変異・選択の組み合わせ自体はCORSの定義に含まれない） | 候補点の生成 |
| サロゲートモデル | `RBFSurrogate`（RBF補間。既定は`gaussian_kernel`だが、任意のカーネル関数を注入できる） | L2 |
| 獲得関数 | `MeanPrediction`（予測平均をそのままスコア化する） | L3 |
| サロゲート管理 | `GlobalSurrogateManager`（アーカイブ全体でRBFをフィットする） | L2-3 |
| 評価戦略 | `IndividualBasedStrategy`（予測平均上位の個体だけを真に評価する） | L3-4 |

```python
import numpy as np
from saealib import (
    GA,
    Optimizer,
    Problem,
    IndividualBasedStrategy,
    RBFSurrogate,
    gaussian_kernel,
    MeanPrediction,
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
surrogate_manager = GlobalSurrogateManager(
    RBFSurrogate(gaussian_kernel, dim=5), MeanPrediction()
)
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

交叉・突然変異・選択の具体的な演算子はCORS自体の定義に含まれないため、上記は一例であり任意の`Crossover`/`Mutation`/`ParentSelection`/`SurvivorSelection`に差し替えられます。

## パラメータと変種

**kernel（RBFカーネルの選択）**: `RBFSurrogate(kernel=...)`で任意のカーネル関数を注入できます。
既定値は`gaussian_kernel`だが、文献の数値実験ではthin plate spline（$\phi(r) = r^2 \log r$）と1次多項式の付加項$p(x)$を組み合わせたモデルが使われています。
saealibの`RBFSurrogate`は多項式項$p(x)$を持たない純粋なRBF補間（学習データの平均を差し引いた残差にフィットする）であるため、文献の設定を厳密に再現するにはカーネルの差し替えだけでは不十分です。

**evaluation_ratio（逐次評価とバッチ評価の切り替え）**: 文献のCORSは、擬似コードのステップ3-4を1回のループにつき1点だけ実行する逐次アルゴリズムです。
saealibの`IndividualBasedStrategy`は、GAが生成した子個体群のうち予測平均上位`evaluation_ratio`割をまとめて真評価するバッチ拡張になっています。
`evaluation_ratio`を個体数の逆数程度まで小さくすれば、1点ずつの逐次評価に近づきます。

## 関連

- [文献リファレンス](../references.md) — 出典の完全な書誌情報
- [Surrogate](../components/surrogate.md) — `RBFSurrogate`/`gaussian_kernel`を含むサロゲートモデル一覧
- [AcquisitionFunction](../components/acquisition_functions.md) — `MeanPrediction`を含む獲得関数一覧
- [SurrogateManager](../components/surrogate_manager.md) — `GlobalSurrogateManager`の詳しい使い方
- [OptimizationStrategy](../components/strategies.md) — `IndividualBasedStrategy`の`evaluation_ratio`を含む戦略一覧
