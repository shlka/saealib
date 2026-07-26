# SPEA2（Strength Pareto Evolutionary Algorithm 2）

SPEA2は、初代SPEA(Strength Pareto Evolutionary Algorithm)の適応度割り当てとアーカイブ管理を改良した多目的進化アルゴリズムです。
支配関係に基づく適応度と$k$最近傍密度を組み合わせて個体を順位付けし、固定サイズの外部アーカイブを境界解が失われない打ち切り手続きで維持します。

## 概要

SPEAは初期の多目的進化アルゴリズムとして高い性能を示しましたが、2つの弱点を抱えていました。

1つは、同じアーカイブ個体に支配される個体同士が同一の適応度を持ってしまい、優劣を区別できない点です。
もう1つは、アーカイブが上限を超えたときに使うクラスタリング手法が、非優越解集合の外周にある境界解を失うことがある点です。

SPEA2は、この2つの弱点をそれぞれ独立した仕組みで解消します。
各個体$i$には、自分が支配する個体数を表す**強度**(strength) $S(i)$が割り当てられ、$i$を支配する個体群の強度の総和が**生の適応度**(raw fitness) $R(i)$になります。
$R(i)=0$は$i$が非優越であることを意味し、値が大きいほど多くの（かつ強い）個体に支配されていることを示します。
同一の$R(i)$を持つ個体同士を区別するため、目的空間上で$k$番目に近い個体との距離$\sigma_i^k$の逆数を**密度**(density) $D(i)=1/(\sigma_i^k+2)$として加えます。
最終的な**適応度**(fitness) $F(i)=R(i)+D(i)$は値が小さいほど優れており、非優越個体は常に$F(i)<1$になります。

SPEA2は個体群とは別に、サイズを固定した外部アーカイブを維持します。
各世代で、個体群とアーカイブを合わせた集合から非優越個体($F(i)<1$)を新しいアーカイブへコピーします。
コピー後のアーカイブがちょうど規定サイズに収まればそのまま採用し、不足する場合は$F(i)$が小さい劣解から順に補充します。
規定サイズを超える場合は、**打ち切り演算子**(truncation operator)を適用し、最近傍距離が最小の個体を1体ずつ、距離を再計算しながら取り除きます。
この手続きは、同じ距離を持つ個体が並ぶタイを2番目と3番目に近い個体との距離で順に解消するため、境界解が誤って除去されにくいです。

出典は{cite}`zitzler2001spea2`。具体的な手順は次の擬似コードに示します。

## 擬似コード

```{prf:algorithm} SPEA2
:label: alg-spea2

**Inputs** 個体群サイズ $N$、アーカイブサイズ $\bar N$、最大世代数 $T$
**Output** 非優越解集合 $A$

1. 初期化：初期個体群 $P_0$ を生成し、空のアーカイブ $\bar P_0 = \emptyset$ を用意して $t=0$ とする
2. 適応度割り当て：$P_t$ と $\bar P_t$ に含まれる各個体の適応度 $F(i) = R(i) + D(i)$ を計算する
3. 環境選択：$P_t \cup \bar P_t$ の非優越個体を $\bar P_{t+1}$ にコピーする。$|\bar P_{t+1}| > \bar N$ なら打ち切り演算子で削減し、$|\bar P_{t+1}| < \bar N$ なら $F(i)$ が小さい劣解から順に補充する
4. 終了判定：$t \geq T$ または他の終了条件を満たせば、$\bar P_{t+1}$ 中の非優越個体を $A$ として出力し停止する
5. 交配選択：$\bar P_{t+1}$ に対して二項トーナメント選択（復元抽出）を行い、交配プールを満たす
6. 変異：交配プールに交叉と突然変異を適用して $P_{t+1}$ を生成し、$t=t+1$ として2へ戻る
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>初期個体群P0を生成<br/>(L1)"] --> GEN
    subgraph GEN["1世代分 (DirectStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>二項トーナメント選択→<br/>SBX交叉→<br/>多項式突然変異で子個体群を生成<br/>(L5, 6)"] --> EVAL["真の評価<br/>（サロゲートを介さない）"]
        EVAL --> COMB["GA.tell()<br/>個体群と子個体群を<br/>単一プールに結合<br/>(L3)"]
        COMB --> FIT["SPEA2Comparator._fitness()<br/>spea2_fitnessが<br/>S(i)→R(i)→D(i)→F(i)を計算<br/>(L2)"]
        FIT --> SORT["SPEA2Comparator.sort_population()<br/>F(i)昇順ソート（環境選択）<br/>(L3)"]
        SORT --> TRUNC["TruncationSelection<br/>上位N個体をPt+1として採用"]
    end
    GEN --> TERM{"終了条件に<br/>到達?"}
    TERM -- "未到達 (L4)" --> GEN
    TERM -- "到達" --> RESULT(["非優越解集合 A"])
```

## 計算量

適応度計算($S(i)$/$R(i)$/$D(i)$)は$O(M^2)$（$M=N+\bar N$）で、密度推定のための距離ソートを含めると$O(M^2\log M)$になります。
打ち切り演算子は最悪$O(M^3)$、平均では$O(M^2\log M)$です{cite}`zitzler2001spea2`。

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（`ask()`で交叉と突然変異、`tell()`で個体群と子個体群を単一プールに結合し生存選択を実行） | L1, 6 |
| 親選択 | `TournamentSelection(tournament_size=2)`（二項トーナメント。`compare_population`経由で勝者を決定） | L5 |
| 交叉 | `CrossoverSBX(prob=0.9, eta=20.0)` | L6 |
| 突然変異 | `MutationPolynomial(eta=20.0)` | L6 |
| 適応度計算 | `SPEA2Comparator`（`sort_population`内部で`spea2_fitness`が$S(i)$/$R(i)$/$D(i)$/$F(i)$を計算） | L2 |
| 環境選択 | `SPEA2Comparator.sort_population()` + `TruncationSelection()`（$F(i)$昇順ソート後に上位$N$個体を採用） | L3 |
| 評価戦略 | `DirectStrategy`（サロゲートを介さず、`GA.ask()`が生成した候補を全て真の目的関数で評価する） | L2（世代内の評価） |

```python
from saealib import GA, SPEA2Comparator, Optimizer
from saealib.benchmarks import zdt1
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.strategies import DirectStrategy
from saealib.termination import Termination, max_fe


problem = zdt1(n_var=10)
problem.comparator = SPEA2Comparator()

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

`problem.comparator = SPEA2Comparator()`の行は省略できません。
NSGA-IIでは`NSGA2Comparator`が`n_obj > 1`のときの既定値なので同じ行を省略できましたが、SPEA2ではそうではありません。

## パラメータと変種

**アーカイブサイズ$\bar N$と個体群サイズ$N$**：論文はこの2つを独立に設定できる一般形として定義しています{cite}`zitzler2001spea2`。
saealibの`GA.tell()`は個体群と子個体群を単一プールとして結合するため、$\bar N$は$N$と同じ値を使います。

**dominator（支配述語）の差し替え**：`SPEA2Comparator(dominator=...)`で、既定の`ParetoDominator`以外の[Dominator](../components/dominance.md)を注入できます。
$S(i)$/$R(i)$の計算はこの支配述語に依存するため、差し替えるとSPEA2の適応度そのものが変わります。

## 関連

- [文献リファレンス](../references.md)：出典の完全な書誌情報
- [Comparator](../components/comparators.md)：`SPEA2Comparator`の詳しい仕様
- [Crossover](../components/crossover.md)：`CrossoverSBX`を含む交叉演算子一覧
- [Mutation](../components/mutation.md)：`MutationPolynomial`を含む突然変異演算子一覧
- [ParentSelection](../components/parent_selection.md)：`TournamentSelection`の詳しい使い方
- [SurvivorSelection](../components/survivor_selection.md)：`TruncationSelection`の詳しい使い方
- [OptimizationStrategy](../components/strategies.md)：`DirectStrategy`を含む戦略一覧
- [Dominator](../components/dominance.md)：`dominator`引数として差し替え可能な支配述語一覧
