# NSGA-III

NSGA-IIIは、NSGA-IIの選択機構を四つ以上の目的を持つ多目的最適化(many-objective optimization)向けに拡張した遺伝アルゴリズムです。
混雑度距離による多様性維持を、あらかじめ配置した参照点(reference point)へのニッチ保存操作に置き換え、目的数が増えても解集合の分布を維持します。

## 概要

目的数が四つ以上に増えると、NSGA-IIの混雑度距離は多様性維持の役割を十分に果たせなくなります。
ランダムに生成した個体群のうち非優越な個体の割合は目的数の増加とともに指数的に増えるため、優越関係による絞り込みだけでは次世代個体群を埋められなくなり、混雑度距離が担う多様性維持の比重が相対的に大きくなるためです。

NSGA-IIIは、この混雑度距離を、目的空間上にあらかじめ配置した**参照点**(reference point)へのニッチ保存操作に置き換えます。
各世代で、個体群全体の**理想点**(ideal point)と**極端点**(extreme point)から超平面の切片を求めて目的関数を正規化し、各個体を最も近い参照点への垂直距離で対応付けます。
まだ完全には受理できていない最後のフロントでは、割り当て済みの個体数（ニッチカウント）が少ない参照点を優先し、そこに対応付けられた個体のうち垂直距離が最小の個体から順に選ぶことで、参照点ごとにほぼ均等な数の解を残します。

NSGA-IIIは、このニッチ保存操作によってすでに多様性を確保しているため、NSGA-IIのような優越関係に基づく親選択を用いません。
次世代個体群から親をランダムに選び、交叉・突然変異を適用して子個体群を生成します。

参照点の一様配置には、Das and Dennisが提案した単体格子法(simplex-lattice design)を用います(Das & Dennis, 1998)。

出典は{cite}`deb2014nsga3`。具体的な手順は次の擬似コードに示します。

<!--
参照情報（レビュー用）:

Deb, K., & Jain, H. (2014).
An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints.
IEEE Transactions on Evolutionary Computation, 18(4), 577-601.
DOI: 10.1109/TEVC.2013.2281535
OCR: .claude/exp_ref/pdfs_ea_saea/user_provided/refs_EA_SAEA/k2012009/k2012009/auto/k2012009.md

Das, I., & Dennis, J. E. (1998).
Normal-boundary intersection: A new method for generating the Pareto surface
in nonlinear multicriteria optimization problems.
SIAM Journal on Optimization, 8(3), 631-657.
DOI: 10.1137/S1052623496307510
（Das & Dennis自体は未読。deb2014nsga3内の言及・要約経由でのみ引用しており、
  実装 uniform_weight_vectors のdocstringも同じ扱い）

ページオフセットの確認方法: このOCRファイルは、全ページに"This article has been
accepted for publication in a future issue of this journal, but has not been
fully edited. Content may change prior to final publication."という透かし文言が
現れる、IEEEの査読受理後・組版前の早期公開版（プレプリント）である。
k2012009_middle.jsonのdiscarded_blocks（フッタ）を確認すると、page_idx=0の
フッタが"1"、page_idx=1が"2"、…という単純な+1オフセットの独自ページ番号
（own page = page_idx + 1）になっている。この独自ページ番号（1-23）は、
最終組版版の巻号ページ（577-601、25ページ）とは版が異なり単純な線形対応には
ならないため、以下の"own p.N"はこのOCRファイル内の独自ページ番号であり、
最終出版ページ番号ではないことに注意（Deb_NSGAII/jones1998_egoのOCRは最終
組版版のためown page=出版ページが一致するが、本ファイルは異なる）。

以下の擬似コードは、論文 Section IV「PROPOSED ALGORITHM: NSGA-III」の
Algorithm 1（Generation t of NSGA-III procedure、own p.5）を主軸に、そこから
呼び出されるAlgorithm 2（Normalize、own p.5-6）・Algorithm 3（Associate、
own p.6）・Algorithm 4（Niching、own p.6-7）の3つの副手続きを1つの手順に
統合したもの。逐語訳ではなく、副手続きの中身をステップ本文に要約して
言い換えている。

- Inputs/Output: Algorithm 1冒頭。own p.5。OCR 110-113行目。
- ステップ1: Section IV冒頭の地の文（NSGA-IIのフレームワークを踏襲する旨）と
  Algorithm 1 Step 2。own p.4（地の文）・p.5（Algorithm 1）。OCR 89-91, 115行目。
- ステップ2: Algorithm 1 Step 3。own p.5。OCR 116行目。
- ステップ3: Algorithm 1 Step 4。own p.5。OCR 117行目。
- ステップ4: Section IV-A「Classification of Population into Non-dominated
  Levels」の地の文（St/Fl/Pt+1の定義）とAlgorithm 1 Step 5-13。own p.5。
  OCR 93-96, 118-126行目。
- ステップ5: Section IV-C「Adaptive Normalization of Population Members」の
  地の文（理想点z^min、ASF式(4)、超平面切片式(5)）とAlgorithm 2。
  own p.5-6。OCR 138-178行目。
- ステップ6: Section IV-D「Association Operation」の地の文とAlgorithm 3
  （垂直距離d^⊥）。own p.6。OCR 180-205行目。
- ステップ7: Section IV-E「Niche-Preservation Operation」の地の文と
  Algorithm 4。own p.6-7。OCR 189-234行目。ニッチカウントρ_jの初期値が
  Pt+1=St\Flから計算される旨はOCR 191, 207行目。
- ステップ8: Section IV-F「Genetic Operations to Create Offspring Population」。
  own p.7。OCR 211-213行目。「we do not employ any explicit selection
  operation with NSGA-III ... constructed by applying the usual crossover
  and mutation operators by randomly picking parents from Pt+1」という
  記述に基づく。NSGA-II/SPEA2のような優越関係ベースの二項トーナメント選択は
  NSGA-III本来の定義には含まれない。

saealibの実装（NSGA3Comparator、src/saealib/comparators/comparators.py 1208行目〜、
補助関数_normalize_objectives/_associate_to_reference_points/_niche_count_selectは
1061-1205行目）の検証結果:

- 正規化（_normalize_objectives）は、理想点の並進・ASFによる極端点探索
  （eps=1e-6）・超平面切片の連立方程式求解という、Algorithm 2の手続き
  （式(4)(5)、own p.5-6）を忠実に実装している。
  ただし、Section IV-C本文（own p.5、OCR 140, 159行目）は理想点・極端点を
  「∪_{τ=0}^t S_τ」（生成0からtまでの全世代の和集合）から求めると述べる一方、
  Algorithm 2 Step 2/4（own p.6、OCR 165, 167-169行目）の疑似コード自体は
  「s∈S_t」（現世代の集合のみ）と書いている。論文の地の文と疑似コードが
  この点で食い違っており、_normalize_objectivesは疑似コード（現世代のみ）
  の記述に一致する実装になっている。これはsaealib側の省略ではなく、論文
  内部の地の文と疑似コードの不一致に対して疑似コードの定義を採用した結果。
  なお、_normalize_objectivesが対象とするのは、sort_populationに渡された
  Rtの実行可能個体全体であり、論文のS_t（F_1..F_lのみ、F_l以降を含まない）
  とは厳密には母集団が異なる。目的ごとの最小値・ASF最小化点は通常F_1に
  属するため実用上はほぼ一致するはずだが、理論上完全に同一の値になる
  保証はない。
- 超平面が縮退する場合（連立方程式が特異、または切片が非正になる場合）の
  フォールバック（f_trans.max(axis=0)を切片として使う、コメント参照）は、
  Part I本文には明記がない。特異ケースへの対処自体は後続の実装（pymoo等）
  で広く行われる一般的な手当てだが、この対応の具体的な典拠は未確認であり、
  saealib側の実装上の補完として扱う。
- 対応付け（_associate_to_reference_points）は、Algorithm 3（垂直距離d^⊥、
  own p.6）をそのまま実装している。
- ニッチ保存（_niche_count_select）は、Algorithm 4（own p.6-7）のロジックを
  フロントごとに逐次処理する形に書き換えている。各反復でpool_refsを
  「現在の候補プール(front_local)に実際に存在する参照点」に絞り込むことで、
  論文Algorithm 4の14-15行目（I_j̄が空ならZ^rからj̄を除いて再試行する）と
  等価な効果を、集合からの明示的な除去なしに実現している。また、
  sort_populationは全フロントに対してこの手続きを適用し（論文は最後の
  非全採択フロントF_lのみに適用）、ニッチカウントを世代内で前のフロントから
  順に積み上げることで、TruncationSelectionによる上位N個体の切り詰めが
  結果的に境界フロントの正しい部分集合を選ぶようにしている。これは論文の
  逐語的な手続きではなくsaealib側の書き換えだが、境界フロント到達時点での
  ニッチカウントの値は論文の定義と一致する。

`.claude/exp_ref/named_algorithms_component_map.md`のNSGA-III行（✅Direct表記、
「Reference-point association; niche count drives preference within fronts」）は、
正規化・対応付け・ニッチ保存の核となる計算については概ね妥当な評価だが、上記の
「論文内部の地の文と疑似コードの不一致」への言及がなく、またSection IV-Fの
「no explicit selection operation」を見落としたまま二項トーナメント選択を
前提にすると誤った再現になる点も注記されていない。
-->

## 擬似コード

```{prf:algorithm} NSGA-III
:label: alg-nsga3

**Inputs** 目的関数群、参照点集合 $Z^r$（構造化点 $Z^s$ またはユーザー指定点 $Z^a$）、個体数 $N$、初期集団 $P_0$
**Output** 最終世代の個体群 $P_{t+1}$

1. $t=0$ とし、ランダム生成した $P_0$ から交叉・突然変異により子個体群 $Q_0$ を生成する
2. 親子を結合した集団 $R_t = P_t \cup Q_t$（サイズ $2N$）を作る
3. $R_t$ を非優越ソートし、フロント列 $\mathcal{F} = (\mathcal{F}_1, \mathcal{F}_2, \ldots)$ を得る
4. $S_t = \emptyset$ とし、$|S_t| \geq N$ となるまでフロントを $\mathcal{F}_1$ から順に $S_t$ へ加える。最後に加えたフロントを $\mathcal{F}_l$、$P_{t+1} = \bigcup_{i=1}^{l-1}\mathcal{F}_i$、$K = N - |P_{t+1}|$ とする（$|S_t|=N$ ならそのまま $P_{t+1}=S_t$ とし8へ進む）
5. $S_t$ の理想点・極端点から超平面の切片を求めて目的関数を正規化し、$Z^r$ を正規化後の目的空間に配置する
6. $S_t$ の各個体を、原点から $Z^r$ の各点を通る参照線への垂直距離が最小の点に対応付ける
7. $\mathcal{F}_1, \ldots, \mathcal{F}_{l-1}$ 上の対応付けから各参照点のニッチカウント $\rho_j$ を求め、ニッチカウントが最小の参照点を優先しながら $\mathcal{F}_l$ から $K$ 個体を選んで $P_{t+1}$ に加える
8. $P_{t+1}$ からランダムに親を選び、交叉・突然変異を適用して $Q_{t+1}$ を生成する。$t=t+1$ として2へ戻り、終了条件に達するまで繰り返す
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>初期集団P0を生成<br/>(L1)"] --> GEN
    subgraph GEN["1世代分 (DirectStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>ランダムに親を選択→<br/>SBX交叉→<br/>多項式突然変異でQtを生成<br/>(L1, 8)"] --> EVAL["真の評価<br/>（サロゲートを介さない）"]
        EVAL --> COMB["GA.tell()<br/>Rt = Pt ∪ Qt を結合<br/>(L2)"]
        COMB --> SORT["NSGA3Comparator.sort_population()<br/>非優越ソート→適応的正規化→<br/>参照点への対応付け→<br/>ニッチ保存選択<br/>(L3-7)"]
        SORT --> TRUNC["TruncationSelection<br/>上位N個体をPt+1として採用<br/>(L4-7)"]
    end
    GEN --> TERM{"終了条件に<br/>到達?"}
    TERM -- "未到達 (L8)" --> GEN
    TERM -- "到達" --> RESULT(["最終世代の個体群"])
```

## 計算量

非優越ソートは $O(N\log^{M-2}N)$（$M$は目的数、$N$は個体数）であり、これはNSGA-IIの $O(MN^2)$ とは異なる漸近的な計算量になります。
正規化・対応付け・ニッチ保存を合わせた1世代あたりの最悪計算量は、$O(N^2\log^{M-2}N)$ または $O(N^2 M)$ のうち大きい方になります{cite}`deb2014nsga3`。

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（`ask()`で交叉・突然変異、`tell()`で $R_t=P_t\cup Q_t$ の結合と生存選択を実行） | L1-2, 8 |
| 親選択 | `TournamentSelection(tournament_size=1)`（トーナメントサイズ1では比較処理自体が実行されないため、論文がSection IV-Fで述べる「$P_{t+1}$からランダムに親を選ぶ」動作に対応する） | L1, 8 |
| 交叉 | `CrossoverSBX(prob=1.0, eta=30.0)` | L1, 8 |
| 突然変異 | `MutationPolynomial(eta=20.0)` | L1, 8 |
| 参照点生成 | `uniform_weight_vectors(n_obj, n_divisions)`（Das-Dennis法の単体格子で $Z^r$ の初期値 $Z^s$ を生成） | L5 |
| 非優越ソート＋正規化＋対応付け＋ニッチ保存 | `NSGA3Comparator`（`sort_population`が内部で`_normalize_objectives`・`_associate_to_reference_points`・`_niche_count_select`を順に呼ぶ） | L3-7 |
| 生存選択 | `TruncationSelection()`（`comparator.sort_population`の順に上位 $N$ 個体を残す） | L4-7 |
| 評価戦略 | `DirectStrategy`（サロゲートを介さず、`GA.ask()`が生成した候補を全て真の目的関数で評価する） | L2 |

```python
from saealib import GA, NSGA3Comparator, Optimizer, uniform_weight_vectors
from saealib.benchmarks import dtlz2
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.strategies import DirectStrategy
from saealib.termination import Termination, max_fe


problem = dtlz2(n_obj=3)
reference_points = uniform_weight_vectors(n_obj=3, n_divisions=8)
problem.comparator = NSGA3Comparator(reference_points)

algorithm = GA(
    CrossoverSBX(prob=1.0, eta=30.0),
    MutationPolynomial(eta=20.0),
    TournamentSelection(tournament_size=1),
    TruncationSelection(),
)

opt = (
    Optimizer(problem)
    .set_algorithm(algorithm)
    .set_strategy(DirectStrategy())
    .set_termination(Termination(max_fe(3000)))
)
ctx = opt.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

`problem.comparator = NSGA3Comparator(reference_points)`の行は省略できません。
NSGA-IIでは`NSGA2Comparator`が`n_obj > 1`のときの既定値なので同じ行を省略できましたが、NSGA-IIIではSPEA2と同様に明示的な代入が必要になります。

2目的のZDTベンチマークではなく3目的のDTLZ2を使ったのは、NSGA-IIIが四つ以上の目的を持つ多目的最適化を主眼にした手法であり、3目的以上の問題で初めて参照点ベースのニッチ保存の効果が現れるためです。

## パラメータと変種

**$\eta_c$（SBX分布指数）と交叉確率$p_c$**: 論文のTable IIは、NSGA-IIIで$p_c=1$（`CrossoverSBX(prob=1.0)`）、$\eta_c=30$（`CrossoverSBX(eta=30.0)`）を使ったと報告しています。
NSGA-IIの既定値（$p_c=0.9$、$\eta_c=20$）より交叉確率・分布指数のいずれも大きく、親に近い子個体をより高い確率で生成する設定になっています{cite}`deb2014nsga3`。

**個体数$N$と参照点数$H$の対応**: 論文は、参照点数$H$に対して個体数$N$を$H$以上で最小の4の倍数に選ぶことを推奨しています。
`Optimizer`は`set_initializer()`を呼ばない場合`LHSInitializer(n_init_population=4*dim)`を既定値として使うため、この個体数は決定変数の次元`dim`にのみ依存し、`H`とは連動しません。
`H`に対して個体数を意図的に揃えたい場合は、`uniform_weight_vectors`が返す配列の行数を確認したうえで、`set_initializer()`で`n_init_population`を明示的に指定します。

**親選択がトーナメントでない理由**: 論文Section IV-Fは、NSGA-IIIがニッチ保存操作によってすでに多様性を確保しているため、明示的な選択演算子を用いず親をランダムに選ぶと述べています{cite}`deb2014nsga3`。
`TournamentSelection(tournament_size=1)`は、トーナメントサイズが1のとき比較処理自体が実行されないため、この「ランダムな親選択」を表現します。
`tournament_size`を2以上に変更すると、NSGA-II/SPEA2と同じ優越関係ベースの選択圧が加わり、論文が意図的に排除した選択機構を導入することになります。

**dominator（支配述語）の差し替え**: `NSGA3Comparator(reference_points, dominator=...)`で、既定の`ParetoDominator`以外の[Dominator](../components/dominance.md)を注入できます。
非優越ソート自体の結果が変わるため、フロント分割・ニッチ保存の対象母集団もこの支配述語に依存します。

## 関連

- [文献リファレンス](../references.md) — 出典の完全な書誌情報
- [Comparator](../components/comparators.md) — `NSGA3Comparator`の詳しい仕様、`reference_points`引数、`rng`の遅延生成
- [Crossover](../components/crossover.md) — `CrossoverSBX`を含む交叉演算子一覧
- [Mutation](../components/mutation.md) — `MutationPolynomial`を含む突然変異演算子一覧
- [ParentSelection](../components/parent_selection.md) — `TournamentSelection`の詳しい使い方
- [SurvivorSelection](../components/survivor_selection.md) — `TruncationSelection`の詳しい使い方
- [OptimizationStrategy](../components/strategies.md) — `DirectStrategy`を含む戦略一覧
- [NonDominatedSorting](../components/nondominated_sorting.md) — 非優越ソートの実装詳細
- [Dominator](../components/dominance.md) — `dominator`引数として差し替え可能な支配述語一覧
