# NSGA-III

NSGA-IIIは、NSGA-IIの選択機構を4つ以上の目的を持つ多目的最適化(many-objective optimization)向けに拡張した遺伝アルゴリズムです。
混雑度距離による多様性維持を、あらかじめ配置した参照点(reference point)へのニッチ保存操作に置き換え、目的数が増えても解集合の分布を維持します。

## 概要

目的数が4つ以上に増えると、NSGA-IIの混雑度距離は多様性維持の役割を十分に果たせなくなります。
ランダムに生成した個体群のうち非優越な個体の割合は目的数の増加とともに指数的に増えるため、優越関係による絞り込みだけでは次世代個体群を埋められなくなり、混雑度距離が担う多様性維持の比重が相対的に大きくなるためです。

NSGA-IIIは、この混雑度距離を、目的空間上にあらかじめ配置した**参照点**(reference point)へのニッチ保存操作に置き換えます。
各世代で、個体群全体の**理想点**(ideal point)と**極端点**(extreme point)から超平面の切片を求めて目的関数を正規化し、各個体を最も近い参照点への垂直距離で対応付けます。
まだ完全には受理できていない最後のフロントでは、割り当て済みの個体数（ニッチカウント）が少ない参照点を優先し、そこに対応付けられた個体のうち垂直距離が最小の個体から順に選ぶことで、参照点ごとにほぼ均等な数の解を残します。

NSGA-IIIは、このニッチ保存操作によってすでに多様性を確保しているため、NSGA-IIのような優越関係に基づく親選択を用いません。
次世代個体群から親をランダムに選び、交叉と突然変異を適用して子個体群を生成します。

参照点の一様配置には、Das and Dennisが提案した単体格子法(simplex-lattice design)を用います(Das & Dennis, 1998)。

出典は{cite}`deb2014nsga3`。具体的な手順は次の擬似コードに示します。

## 擬似コード

```{prf:algorithm} NSGA-III
:label: alg-nsga3

**Inputs** 目的関数群、参照点集合 $Z^r$（構造化点 $Z^s$ またはユーザー指定点 $Z^a$）、個体数 $N$、初期個体群 $P_0$
**Output** 最終世代の個体群 $P_{t+1}$

1. $t=0$ とし、ランダム生成した $P_0$ から交叉と突然変異により子個体群 $Q_0$ を生成する
2. 親子を結合した個体群 $R_t = P_t \cup Q_t$（サイズ $2N$）を作る
3. $R_t$ を非優越ソートし、フロント列 $\mathcal{F} = (\mathcal{F}_1, \mathcal{F}_2, \ldots)$ を得る
4. $S_t = \emptyset$ とし、$|S_t| \geq N$ となるまでフロントを $\mathcal{F}_1$ から順に $S_t$ へ加える。最後に加えたフロントを $\mathcal{F}_l$、$P_{t+1} = \bigcup_{i=1}^{l-1}\mathcal{F}_i$、$K = N - |P_{t+1}|$ とする（$|S_t|=N$ ならそのまま $P_{t+1}=S_t$ とし8へ進む）
5. $S_t$ の理想点と極端点から超平面の切片を求めて目的関数を正規化し、$Z^r$ を正規化後の目的空間に配置する
6. $S_t$ の各個体を、原点から $Z^r$ の各点を通る参照線への垂直距離が最小の点に対応付ける
7. $\mathcal{F}_1, \ldots, \mathcal{F}_{l-1}$ 上の対応付けから各参照点のニッチカウント $\rho_j$ を求め、ニッチカウントが最小の参照点を優先しながら $\mathcal{F}_l$ から $K$ 個体を選んで $P_{t+1}$ に加える
8. $P_{t+1}$ からランダムに親を選び、交叉と突然変異を適用して $Q_{t+1}$ を生成する。$t=t+1$ として2へ戻り、終了条件に達するまで繰り返す
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>初期個体群P0を生成<br/>(L1)"] --> GEN
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
正規化、対応付け、ニッチ保存を合わせた1世代あたりの最悪計算量は、$O(N^2\log^{M-2}N)$ または $O(N^2 M)$ のうち大きい方になります{cite}`deb2014nsga3`。

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（`ask()`で交叉と突然変異、`tell()`で $R_t=P_t\cup Q_t$ の結合と生存選択を実行） | L1-2, 8 |
| 親選択 | `TournamentSelection(tournament_size=1)`（トーナメントサイズ1では比較処理自体が実行されないため、論文がSection IV-Fで述べる「$P_{t+1}$からランダムに親を選ぶ」動作に対応する） | L1, 8 |
| 交叉 | `CrossoverSBX(prob=1.0, eta=30.0)` | L1, 8 |
| 突然変異 | `MutationPolynomial(eta=20.0)` | L1, 8 |
| 参照点生成 | `uniform_weight_vectors(n_obj, n_divisions)`（Das-Dennis法の単体格子で $Z^r$ の初期値 $Z^s$ を生成） | L5 |
| 非優越ソート＋正規化＋対応付け＋ニッチ保存 | `NSGA3Comparator`（`sort_population`が内部で`_normalize_objectives`/`_associate_to_reference_points`/`_niche_count_select`を順に呼ぶ） | L3-7 |
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

2目的のZDTベンチマークではなく3目的のDTLZ2を使ったのは、NSGA-IIIが4つ以上の目的を持つ多目的最適化を主眼にした手法であり、3目的以上の問題で初めて参照点ベースのニッチ保存の効果が現れるためです。

## パラメータと変種

**$\eta_c$（SBX分布指数）と交叉確率$p_c$**：論文のTable IIは、NSGA-IIIで$p_c=1$（`CrossoverSBX(prob=1.0)`）、$\eta_c=30$（`CrossoverSBX(eta=30.0)`）を使ったと報告しています。
NSGA-IIの既定値（$p_c=0.9$、$\eta_c=20$）より交叉確率と分布指数のいずれも大きく、親に近い子個体をより高い確率で生成する設定になっています{cite}`deb2014nsga3`。

**個体数$N$と参照点数$H$の対応**：論文は、参照点数$H$に対して個体数$N$を$H$以上で最小の4の倍数に選ぶことを推奨しています。
`Optimizer`は`set_initializer()`を呼ばない場合`LHSInitializer(n_init_population=4*dim)`を既定値として使うため、この個体数は決定変数の次元`dim`にのみ依存し、`H`とは連動しません。
`H`に対して個体数を意図的に揃えたい場合は、`uniform_weight_vectors`が返す配列の行数を確認したうえで、`set_initializer()`で`n_init_population`を明示的に指定します。

**親選択がトーナメントでない理由**：論文Section IV-Fは、NSGA-IIIがニッチ保存操作によってすでに多様性を確保しているため、明示的な選択演算子を用いず親をランダムに選ぶと述べています{cite}`deb2014nsga3`。
`TournamentSelection(tournament_size=1)`は、トーナメントサイズが1のとき比較処理自体が実行されないため、この「ランダムな親選択」を表現します。
`tournament_size`を2以上に変更すると、NSGA-II/SPEA2と同じ優越関係ベースの選択圧が加わり、論文が意図的に排除した選択機構を導入することになります。

**dominator（支配述語）の差し替え**：`NSGA3Comparator(reference_points, dominator=...)`で、既定の`ParetoDominator`以外の[Dominator](../components/dominance.md)を注入できます。
非優越ソート自体の結果が変わるため、フロント分割とニッチ保存の対象個体群もこの支配述語に依存します。

## 関連

- [文献リファレンス](../references.md)：出典の完全な書誌情報
- [Comparator](../components/comparators.md)：`NSGA3Comparator`の詳しい仕様、`reference_points`引数、`rng`の遅延生成
- [Crossover](../components/crossover.md)：`CrossoverSBX`を含む交叉演算子一覧
- [Mutation](../components/mutation.md)：`MutationPolynomial`を含む突然変異演算子一覧
- [ParentSelection](../components/parent_selection.md)：`TournamentSelection`の詳しい使い方
- [SurvivorSelection](../components/survivor_selection.md)：`TruncationSelection`の詳しい使い方
- [OptimizationStrategy](../components/strategies.md)：`DirectStrategy`を含む戦略一覧
- [NonDominatedSorting](../components/nondominated_sorting.md)：非優越ソートの実装詳細
- [Dominator](../components/dominance.md)：`dominator`引数として差し替え可能な支配述語一覧
