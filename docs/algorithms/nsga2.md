# NSGA-II（Nondominated Sorting Genetic Algorithm II）

NSGA-IIは、多目的最適化における選択機構として最も広く使われる遺伝アルゴリズムです。
非優越ソートと混雑度距離を組み合わせ、パレートフロントへの収束と解集合の多様性維持を同時に達成します。

## 概要

多目的最適化には、パレートフロントへの収束と、解集合内の多様性維持という、独立した二つの目標があります。

従来のNSGA（非優越ソート遺伝アルゴリズム）は、この二つを非優越ソートと共有関数(sharing function)で達成していましたが、共有関数は分散パラメータ $\sigma_{\mathrm{share}}$ の手動調整を要し、計算量も $O(N^2)$ でした。

NSGA-IIは、この共有関数を**混雑度距離**(crowding distance)に基づく**混雑比較演算子**($\prec_n$)に置き換え、パラメータ不要な多様性維持を実現します。

各個体は非優越ランク $i_{\mathrm{rank}}$ と混雑度距離 $i_{\mathrm{distance}}$ の二つの属性を持ちます。
$\prec_n$ はランクが低い（良い）個体を優先し、同ランク内では混雑度距離が大きい（周囲が疎な）個体を優先します。

さらにNSGA-IIは、親個体群 $P_t$ と子個体群 $Q_t$ を結合した $2N$ 個体からエリート選択を行い、優れた解が世代を跨いで失われないようにします(elitism)。

出典は{cite}`deb2002nsga2`。具体的な手順は次の擬似コードに示します。

## 擬似コード

<!--
参照情報（レビュー用）:
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
A fast and elitist multiobjective genetic algorithm: NSGA-II.
IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
DOI: 10.1109/4235.996017

以下の擬似コードは、論文中の独立した3つの断片（fast-non-dominated-sort,
crowding-distance-assignment, Main Loop）を1つの手順に統合したもの。
逐語訳ではなく、変数名・記法は論文のものを保持しつつ日本語の指示文に言い換えている。

- ステップ1: 論文 Section III-C 冒頭の地の文（枠囲み外）。
  「usual binary tournament selection, recombination, and mutation
  operators are used to create Q_0」に基づく。p.185。
- ステップ2-4, 6-8: 論文 Section III-C の Main Loop 枠囲み擬似コード。p.185。
  OCR: exp_ref/pdfs_ea_saea/user_provided/refs_EA_SAEA/Deb_NSGAII/
  Deb_NSGAII/auto/Deb_NSGAII.md 129-140行目。
- ステップ3の非優越ソート自体の詳細手続き:
  論文 Section III-A `fast-non-dominated-sort(P)`。p.183。
  OCR: 同上ファイル 55-77行目。
- ステップ5の混雑度距離の詳細手続き:
  論文 Section III-B `crowding-distance-assignment(I)`。p.184。
  OCR: 同上ファイル 85-92行目（距離計算式は式(1)ではなく本文中の無番号の式）。
- ≺n（crowded-comparison operator）の定義: 論文 Section III-B 式(なし、
  無番号の定義式)。p.184。OCR: 同上ファイル 109-123行目。
-->

```{prf:algorithm} NSGA-II
:label: alg-nsga2

**Inputs** 目的関数群、個体数 $N$、初期集団 $P_0$
**Output** 最終世代のパレートフロント

1. $t=0$ とし、$P_0$ をランダム生成した上で、二項トーナメント選択・交叉・突然変異により子個体群 $Q_0$ を生成する
2. 親子を結合した集団 $R_t = P_t \cup Q_t$（サイズ $2N$）を作る
3. $R_t$ を非優越ソートし、フロント列 $\mathcal{F} = (\mathcal{F}_1, \mathcal{F}_2, \ldots)$ を得る
4. $P_{t+1} = \emptyset$ とし、$\mathcal{F}_i$ が丸ごと収まる限り、フロントを順に $P_{t+1}$ へ追加する
5. 途中で収まらなくなった最後のフロント $\mathcal{F}_l$ について、各個体の混雑度距離を計算する
6. $\mathcal{F}_l$ を $\prec_n$ で降順ソートし、$P_{t+1}$ が $N$ 個体になるまで先頭から採用する
7. $P_{t+1}$ に対し、$\prec_n$ を選択基準とする二項トーナメント選択・交叉・突然変異を適用し、$Q_{t+1}$ を生成する
8. $t = t+1$ として2へ戻り、終了条件に達するまで繰り返す
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>初期集団P0を生成<br/>(L1)"] --> GEN
    subgraph GEN["1世代分 (DirectStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>二項トーナメント選択→<br/>SBX交叉→<br/>多項式突然変異でQtを生成<br/>(L1, 7)"] --> EVAL["真の評価<br/>（サロゲートを介さない）"]
        EVAL --> COMB["GA.tell()<br/>Rt = Pt ∪ Qt を結合<br/>(L2)"]
        COMB --> SORT["NSGA2Comparator.sort_population()<br/>非優越ソート→<br/>混雑度距離<br/>(L3-6)"]
        SORT --> TRUNC["TruncationSelection<br/>上位N個体をPt+1として採用<br/>(L4-6)"]
    end
    GEN --> TERM{"終了条件に<br/>到達?"}
    TERM -- "未到達 (L8)" --> GEN
    TERM -- "到達" --> RESULT(["パレートフロント"])
```

## 計算量

非優越ソートは $O(MN^2)$（$M$は目的数、$N$は個体数）、混雑度距離の計算は $O(MN\log N)$、$\prec_n$によるソートは $O(N\log N)$ です。

1世代あたりの支配的なコストは非優越ソートであり、全体の計算量は $O(MN^2)$ になります{cite}`deb2002nsga2`。

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（`ask()`で交叉・突然変異、`tell()`で $R_t=P_t\cup Q_t$ の結合と生存選択を実行） | L1-2, 7 |
| 親選択 | `TournamentSelection(tournament_size=2)`（`ctx.comparator.compare_population`で勝者を決定） | L1, 7 |
| 交叉 | `CrossoverSBX(prob=0.9, eta=20.0)` | L1, 7 |
| 突然変異 | `MutationPolynomial(eta=20.0)` | L1, 7 |
| 非優越ソート＋混雑度距離 | `NSGA2Comparator`（`sort_population`が内部で`non_dominated_sort`と`crowding_distance_all_fronts`を呼ぶ） | L3-6 |
| 生存選択 | `TruncationSelection()`（`comparator.sort_population`の順に上位 $N$ 個体を残す） | L4-6 |
| 評価戦略 | `DirectStrategy`（サロゲートを介さず、`GA.ask()`が生成した候補を全て真の目的関数で評価する） | L2 |

```python
from saealib import GA, NSGA2Comparator, Optimizer
from saealib.benchmarks import zdt1
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.strategies import DirectStrategy
from saealib.termination import Termination, max_fe


problem = zdt1(n_var=10)
problem.comparator = NSGA2Comparator()

algorithm = GA(
    CrossoverSBX(prob=0.9, eta=20.0),
    MutationPolynomial(eta=20.0),
    TournamentSelection(tournament_size=2),
    TruncationSelection(),
)

opt = (
    Optimizer(problem)
    .set_algorithm(algorithm)
    .set_strategy(DirectStrategy())
    .set_termination(Termination(max_fe(2000)))
)
ctx = opt.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

`problem.comparator = NSGA2Comparator()`の行は、`n_obj > 1`のときの既定値と同じであるため省略できます。
サロゲートを一切使わないため、`Optimizer`に`set_surrogate_manager()`を呼ぶ必要がありません。

## パラメータと変種

**$\eta_c$・$\eta_m$（分布指数）**: `CrossoverSBX(eta=...)`と`MutationPolynomial(eta=...)`で調整します。
値が大きいほど親に近い子個体を生成します（探索が保守的になる）。
論文の実数値実験では両方とも$20$が使われており、これがsaealib側のコード例の既定値でもあります{cite}`deb2002nsga2`。

**$p_m$（変数単位の突然変異確率）**: 論文は $p_m = 1/n$（$n$は決定変数の数）を使います。
これは個体レベルの`prob`ではなく、変数ごとの適用確率`prob_var`に対応します。
`MutationPolynomial(prob_var=None)`（既定値）では$\min(0.5,\, 1/\mathrm{dim})$が自動設定され、次元数が大きいほど論文の設定に近づきます。

**タイブレークの扱い**: 論文の擬似コードは、$\mathcal{F}_l$を$\prec_n$で降順ソートするとしか述べておらず、混雑度距離が同値な個体同士の順序は規定していません。
`TruncationSelection(randomize_ties=False)`（既定値）は`sort_population`が返す決定的な順序をそのまま使い、この記述に対応します。
`randomize_ties=True`にすると、打ち切り境界で同値な個体をシャッフルしてから切り詰めます。

## 関連

- [文献リファレンス](../references.md) — 出典の完全な書誌情報
- [Comparator](../components/comparators.md) — `NSGA2Comparator`の詳しい仕様
- [Crossover](../components/crossover.md) — `CrossoverSBX`を含む交叉演算子一覧
- [Mutation](../components/mutation.md) — `MutationPolynomial`を含む突然変異演算子一覧
- [ParentSelection](../components/parent_selection.md) — `TournamentSelection`の詳しい使い方
- [SurvivorSelection](../components/survivor_selection.md) — `TruncationSelection`の詳しい使い方
- [OptimizationStrategy](../components/strategies.md) — `DirectStrategy`を含む戦略一覧
- [NonDominatedSorting](../components/nondominated_sorting.md) — 非優越ソートの実装詳細
