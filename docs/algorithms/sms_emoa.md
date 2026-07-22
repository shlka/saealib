# SMS-EMOA（S metric selection EMOA）

SMS-EMOAは、被支配超体積（dominated hypervolume、$\mathcal{S}$メトリック）を選択基準に直接組み込んだ、定常状態（steady-state）の多目的進化アルゴリズムです。
非優越ソートで個体群をフロントに分割したうえで、最下位フロント内の超体積寄与度が最小の個体を1体ずつ淘汰し、世代を追うごとに個体群全体の被支配超体積を単調に増大させます。

## 概要

超体積指標は、パレートフロント近似の質を測る指標として広く使われています。
基準点$\mathbf{y}_{\mathrm{ref}}$を1つ固定すると、解集合$B$が支配する領域のルベーグ測度$\mathcal{S}(B, \mathbf{y}_{\mathrm{ref}})$が定義でき、有限のパレートフロント近似において$\mathcal{S}$を最大化することが真のパレート集合を求めることと等価になることが知られています。

SMS-EMOAは、この超体積指標を評価用途にとどめず、選択演算子そのものに採用します。
NSGA-IIと同じ**非優越ソート**で個体群をフロント$\mathcal{R}_1, \ldots, \mathcal{R}_v$に分割し、最下位フロント$\mathcal{R}_v$の中で、除いたときに$\mathcal{S}$メトリックの減少量が最小となる個体を1体だけ淘汰します。
この減少量を**排他的超体積寄与度**(exclusive hypervolume contribution) $\Delta_{\mathcal{S}}(s, \mathcal{R}_v) := \mathcal{S}(\mathcal{R}_v) - \mathcal{S}(\mathcal{R}_v \setminus \{s\})$と呼びます。

超体積の計算コストが高いため、SMS-EMOAは**定常状態**(steady-state)の世代交代を採ります。
1世代につき交叉・突然変異で新個体を1体だけ生成し、個体群サイズ$\mu$を保つために既存個体を1体だけ淘汰します。
$(\mu+\lambda)$世代交代のように$\binom{\mu+\lambda}{\mu}$通りの組み合わせを比較する必要がなく、最下位フロント内で高々$\mu+1$回の$\mathcal{S}$メトリック評価に抑えられます。

出典は{cite}`beume2007smsemoa`。具体的な手順は次の擬似コードに示します。

<!--
参照情報（レビュー用）:

Beume, N., Naujoks, B., & Emmerich, M. (2007).
SMS-EMOA: Multiobjective selection based on dominated hypervolume.
European Journal of Operational Research, 181(3), 1653-1669.
DOI: 10.1016/j.ejor.2006.08.008
OCR: .claude/exp_ref/pdfs_ea_saea/user_provided/refs_EA_SAEA/beume2007_sms_emoa/beume2007_sms_emoa/auto/beume2007_sms_emoa.md

ページオフセットの確認方法: OCR中のdiscarded_blocks（フッタ）を確認すると、
beume2007_sms_emoa_middle.jsonのpdf_info[1].discarded_blocksに"1654"が現れ、
これはpage_idx=1に対応する。誌面の書誌情報「European Journal of Operational
Research 181 (2007) 1653-1669」と合わせると、real_page = page_idx + 1653
（page_idx=0が誌面の初出ページ1653）という単純な+1653オフセットになる
（content_list.json中のpage_numberエントリでpage_idx=2→"1655"、
page_idx=3→"1656"であることも確認済み）。

以下の擬似コードは、論文 Section 2.1「Details of the SMS-EMOA」のAlgorithm 1
（SMS-EMOA、p.1655）とAlgorithm 2（Reduce(Q)、p.1655）を1つの手順に統合したもの。
逐語訳ではなく、繰り返し構造(repeat/until)を通し番号の手順に言い換えている。

- Inputs/Output、ステップ1: Algorithm 1 Step 1（P0 <- init()）。p.1655。
  OCR 59-61行目。
- ステップ2: Algorithm 1 Step 3-4（repeat; q_{t+1} <- generate(P_t)）。p.1655。
  OCR 62-63行目。変異演算子の内訳（SBX交叉+多項式突然変異、Deb et al.のe-MOEAと同じ）
  は脚注1、p.1655、OCR 57行目。
- ステップ3: Algorithm 1 Step 5前半（Q = P_t ∪ {q_{t+1}}の結合）。p.1655。
  OCR 64行目、およびAlgorithm 2 Step 1（fast-nondominated-sort(Q)）。p.1655。OCR 71行目。
- ステップ4: Algorithm 2 Step 1の非優越ソート本体の説明（地の文、fast-nondominated-sort
  がNSGA-IIのものと同じ旨）。p.1655。OCR 68行目。
- ステップ5: Algorithm 2 Step 2（r <- argmin_{s in R_v}[Delta_S(s, R_v)]）と、
  地の文での場合分け（|R_v| > 1のときのみ計算する旨）。式(3)の定義。p.1655。
  OCR 72, 75-79行目。
- ステップ6: Algorithm 2 Step 3（P_{t+1} = Q \ {r}）。p.1655。OCR 73行目。
  単調性の不変条件 S(P_t) <= S(P_{t+1})（式(4)）は p.1656、OCR 83-85行目。
- ステップ7: Algorithm 1 Step 6-7（t <- t+1; until termination condition）。p.1655。
  OCR 65-66行目。

擬似コードに現れない実装上の補足（本文中で言及）:
- 定常状態選択の採用理由（μ+1回のS評価で済む旨）: Section 2.1.1
  "Steady-state selection"、p.1656、OCR 87-89行目。
- 参照点y_refの動的な再計算（各世代で「現在の最悪目的関数値+1.0」を採用する旨）と、
  2目的の場合はy_refを使わず両端の極値解を無条件に残す旨: Section 2.1.3
  "Handling of boundary solutions"、p.1656、OCR 95-99行目。
- 個体数を一定に保つ設計（アーカイブ方式を採らない理由）: Section 2.1.2
  "Population size"、p.1656、OCR 91-93行目。
- 支配点数d(s, P(t))に基づく代替のReduce手続き（Algorithm 3、"SMS-EMOA dp"）:
  Section 2.2 "Selection variants of SMS-EMOA"、p.1656-1657、OCR 101-119行目。
  この変種はsaealibに実装されていない（後述）。

saealibの実装（HypervolumeComparator、src/saealib/comparators/comparators.py 719行目〜）
の検証結果:

- HypervolumeComparator.sort_population()は、非優越ソート後、各フロント内で
  hypervolume_contributions（src/saealib/utils/indicators.py）による排他的HV寄与度の
  降順ソートを二次キーに使う。この計算自体（式(3)のΔ_Sの定義）はAlgorithm 2 Step 2と
  一致する。
- ただし、この計算をHypervolumeComparatorは**全てのフロントに対して**適用し、
  母集団全体の完全なランキングを作る。論文のAlgorithm 2は、最下位フロントR_vただ1つに
  対してのみΔ_Sを計算し、その中の最小個体1体を淘汰するだけである（他のフロントは
  ΔS計算なしでそのままP_{t+1}に残る）。この一般化はクラスdocstring内の
  `.. note:: **Generalization from SMS-EMOA.**`で公にドキュメント化されている
  意図的な設計変更であり、隠れた未実装ではない。TruncationSelection/TournamentSelectionと
  組み合わせて汎用的に使うための一般化である旨が明記されている。
  なお、後述のように1世代1個体（steady-state）の構成で使う限り、実際に切り詰められるのは
  常に1個体のみであり、その1個体は必ず最下位フロントに属するため、このモジュール docstring
  でのべた一般化の効果は現れない
  （TruncationSelectionが切り詰めるのはsort_populationが返す配列の末尾1件だけであり、
  それは最悪フロント内で最も寄与度が低い個体と一致する）。
- 参照点の扱いは、論文（Section 2.1.3）が「各世代で現在の最悪目的関数値+1.0」という
  絶対量のオフセットを使うのに対し、saealibのhypervolume_contributions（reference_point
  未指定時）は「最悪目的関数値 + margin * （最悪値-最良値）」という相対的なマージン
  （既定margin=0.1）を使う点で異なる。また、2目的の場合に両端の極値解を無条件に残す
  という論文の特別扱い（Section 2.1.3前半）もsaealib側には実装されておらず、
  常に基準点越しの排他的寄与度で一律に評価される。
- Section 2.2の代替Reduce手続き（支配点数d(s, P(t))による選択、通称"SMS-EMOA dp"）は
  saealibに実装されていない。HypervolumeComparatorはΔ_Sベースの基本版（"SMS-EMOA"）のみ
  提供する。

`.claude/exp_ref/named_algorithms_component_map.md`の「SMS-EMOA (selection)」行
（✅表記、実装キー`HV`、注記「Worst individual = smallest exclusive HV contribution」）は、
Reduceの核となる選択則については妥当だが、以下の点への言及がない:
1. HypervolumeComparator自体が「全フロントへの一般化」であること（docstringには明記済み）。
2. 参照点の計算式（絶対+1.0 vs 相対margin）と2目的特別扱いの省略という、論文の定義との
   細部の相違。
3. 表の「(GA only)」という注記はsaealib側の生存選択がGAのtell()を通ることを指しているが、
   1世代1個体という定常状態のオフスプリング数についての言及がなく、この点はテーブル外の
   別ノート「SMSEGOAcquisition + HypervolumeComparator」の行にも現れない。
   この定常状態パターンをsaealibでどう構成するかは、本ページで新たに検証した。
-->

## 擬似コード

```{prf:algorithm} SMS-EMOA
:label: alg-sms-emoa

**Inputs** 目的関数群、個体数 $\mu$、初期集団 $P_0$
**Output** 最終世代の個体群 $P_{t+1}$

1. $t=0$ とし、$\mu$ 個体からなる初期集団 $P_0$ を生成する
2. 交叉・突然変異により、$P_t$ から新個体 $q_{t+1}$ を1体だけ生成する
3. $Q = P_t \cup \{q_{t+1}\}$（サイズ $\mu+1$）を非優越ソートし、フロント列 $\mathcal{R}_1, \ldots, \mathcal{R}_v$ を得る
4. 最下位フロント $\mathcal{R}_v$ を特定する（$|\mathcal{R}_v|=1$ ならその1個体がそのまま淘汰対象になる）
5. $|\mathcal{R}_v| > 1$ のとき、$\mathcal{R}_v$ 内の各個体 $s$ について排他的超体積寄与度 $\Delta_{\mathcal{S}}(s, \mathcal{R}_v) = \mathcal{S}(\mathcal{R}_v) - \mathcal{S}(\mathcal{R}_v \setminus \{s\})$ を計算し、最小の個体 $r$ を選ぶ
6. $P_{t+1} = Q \setminus \{r\}$ とする（$\mathcal{S}(P_t) \leq \mathcal{S}(P_{t+1})$ が常に成り立つ）
7. $t=t+1$ として2へ戻り、終了条件に達するまで繰り返す
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>μ個体の初期集団P0を生成<br/>(L1)"] --> GEN
    subgraph GEN["1世代分 (SteadyStateStrategy.step)"]
        direction TB
        ASK["GA.ask(n_offspring=1)<br/>親をランダムに選択→<br/>SBX交叉→<br/>多項式突然変異で新個体q_t+1を1体生成<br/>(L2)"] --> EVAL["真の評価<br/>（サロゲートを介さない）"]
        EVAL --> COMB["GA.tell()<br/>Q = Pt ∪ {q_t+1} を結合<br/>(L3)"]
        COMB --> SORT["HypervolumeComparator.sort_population()<br/>非優越ソート→<br/>フロント内HV寄与度<br/>(L3-5)"]
        SORT --> TRUNC["TruncationSelection<br/>末尾1個体を淘汰<br/>（最下位フロントで寄与度最小）<br/>(L4-6)"]
    end
    GEN --> TERM{"終了条件に<br/>到達?"}
    TERM -- "未到達 (L7)" --> GEN
    TERM -- "到達" --> RESULT(["最終世代の個体群"])
```

## 計算量

超体積計算自体は、点数について多項式だが目的数について指数的な計算量を持ちます。
saealibの`hypervolume`（再帰的スライシング）は$O(n^{m-1} n \log n)$（$n$は点数、$m$は目的数）です。

排他的寄与度は1点抜き（leave-one-out）でサイズ$k$のフロントに対し$k$回のHV計算を要するため、フロント1つ分の計算は$O(k^{m} \log k)$になります。
論文のAlgorithm 2は最下位フロント（サイズ高々$\mu+1$）だけにこれを適用するため、1世代あたり$O(\mu^{m} \log \mu)$に収まります{cite}`beume2007smsemoa`。

`HypervolumeComparator`は全フロントに対して寄与度を計算する一般化を行いますが、フロントサイズの総和は$\mu+1$を超えないため、漸近的な上界は$O(\mu^{m} \log \mu)$のまま変わりません。

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（`ask(n_offspring=1)`で新個体を1体だけ生成、`tell()`で $Q=P_t\cup\{q_{t+1}\}$ の結合と生存選択を実行） | L2-3, 6 |
| 親選択 | `TournamentSelection(tournament_size=1)`（比較処理を伴わない一様ランダム選択。論文は親選択方式を明記していない） | L2 |
| 交叉 | `CrossoverSBX(prob=0.9, eta=20.0)` | L2 |
| 突然変異 | `MutationPolynomial(eta=20.0)` | L2 |
| 非優越ソート＋フロント内HV寄与度 | `HypervolumeComparator`（`sort_population`が内部で非優越ソートと`hypervolume_contributions`を呼ぶ） | L3-5 |
| 生存選択 | `TruncationSelection()`（`comparator.sort_population`の末尾1個体、すなわち最下位フロントで寄与度最小の個体を淘汰） | L4-6 |
| 評価戦略 | 独自の`SteadyStateStrategy`（`DirectStrategy`の1世代1個体版。後述） | 2, 6-7（世代内の評価） |

`DirectStrategy`は`AskStage`に`n_offspring`を渡さず、既定で個体群サイズと同数の子個体を生成する$(\mu+\lambda)$世代交代を組みます。
SMS-EMOAは1世代につき新個体を1体だけ生成する定常状態のアルゴリズムであるため、`DirectStrategy`をそのまま使うと擬似コードのステップ2と食い違います。
[OptimizationStrategy](../components/strategies.md)の「独自Strategyの実装方法」で示されている手順に従い、`AskStage(n_offspring=1)`を指定した`DirectStrategy`相当のパイプラインを組んで対応します。

```python
from saealib import GA, HypervolumeComparator, Optimizer, OptimizationStrategy, Pipeline
from saealib.benchmarks import zdt1
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.stages import (
    ArchiveUpdateStage,
    AskStage,
    CountGenerationStage,
    TellStage,
    TrueEvaluationStage,
)
from saealib.termination import Termination, max_fe


class SteadyStateStrategy(OptimizationStrategy):
    """DirectStrategyの1世代1個体版（SMS-EMOAの定常状態選択）。"""

    requires_surrogate = False

    def step(self, ctx, provider):
        cbmanager = getattr(provider, "cbmanager", None)
        pipeline = Pipeline(
            [
                CountGenerationStage(),
                AskStage(provider.algorithm, n_offspring=1, cbmanager=cbmanager),
                TrueEvaluationStage(provider.evaluator, cbmanager=cbmanager),
                ArchiveUpdateStage(),
                TellStage(provider.algorithm),
            ]
        )
        return pipeline.execute(ctx)


problem = zdt1(n_var=10)
problem.comparator = HypervolumeComparator()

algorithm = GA(
    CrossoverSBX(prob=0.9, eta=20.0),
    MutationPolynomial(eta=20.0),
    TournamentSelection(tournament_size=1),
    TruncationSelection(),
)

opt = (
    Optimizer(problem)
    .set_algorithm(algorithm)
    .set_strategy(SteadyStateStrategy())
    .set_termination(Termination(max_fe(2000)))
)
ctx = opt.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

`problem.comparator = HypervolumeComparator()`の行は省略できません。
NSGA-IIでは`NSGA2Comparator`が`n_obj > 1`のときの既定値なので同じ行を省略できましたが、SPEA2・NSGA-IIIと同様、SMS-EMOAでも明示的な代入が必要になります。

## パラメータと変種

**定常状態か$(\mu+\lambda)$世代交代か**: 論文のAlgorithm 1は1世代1個体の定常状態(steady-state)を前提としており、これは計算コストの高い超体積評価を最下位フロントの高々$\mu+1$回に抑えるための設計選択です{cite}`beume2007smsemoa`。
上記のコード例はこれに忠実な構成です。
一方、`DirectStrategy`をそのまま使い`AskStage`の`n_offspring`を指定しない構成（NSGA-II/SPEA2/NSGA-IIIと同じ$(\mu+\lambda)$パターン）も動作はしますが、その場合`HypervolumeComparator`の「全フロントへの一般化」が実際に効いてくる点に注意してください。
1世代で$\mu$個体を新規生成すると、非優越ソート後に複数のフロントにまたがって多数の個体が淘汰されうるため、最下位フロントだけを見る論文の定義から外れ、全フロントにHV寄与度ランキングを及ぼす一般化された生存選択に切り替わります。

**参照点の扱い**: `HypervolumeComparator(reference_point=...)`で固定値を指定できるほか、既定の`None`では世代ごと・フロントごとに自動計算されます。
自動計算は「最悪目的関数値 + `margin` * （最悪値-最良値）」という相対マージン（既定`margin=0.1`）であり、論文が使う「最悪目的関数値 + 1.0」という絶対オフセットとは式が異なります（Section 2.1.3）。
また、論文は2目的の場合に両端の極値解を基準点計算なしで無条件に残しますが、saealibにこの特別扱いはなく、常に基準点越しの寄与度で一律に評価します。

**親選択の方式**: 論文のAlgorithm 1は「変異演算子によって新個体を生成する」とのみ述べ、親をどう選ぶかを明記していません（NSGA-IIやSPEA2のような優越関係ベースのトーナメント選択の記述はない）。
`TournamentSelection(tournament_size=1)`は、トーナメントサイズが1のとき比較処理自体が実行されないため、個体群からの一様ランダム選択を表現する構成として採用しました。

**代替のReduce手続き（"SMS-EMOA dp"）**: 論文Section 2.2は、超体積寄与度の代わりに支配点数$d(s, P(t))$を使う高速な変種を提案しています。
`HypervolumeComparator`はこの変種を実装しておらず、$\Delta_{\mathcal{S}}$による基本版のみを提供します。

**dominator（支配述語）の差し替え**: `HypervolumeComparator(reference_point=..., dominator=...)`で、既定の`ParetoDominator`以外の[Dominator](../components/dominance.md)を注入できます。
非優越ソートの結果が変わるため、フロント分割・寄与度計算の対象母集団もこの支配述語に依存します。

## 関連

- [文献リファレンス](../references.md) — 出典の完全な書誌情報
- [Comparator](../components/comparators.md) — `HypervolumeComparator`の詳しい仕様、母集団相対的なComparatorの扱い
- [Crossover](../components/crossover.md) — `CrossoverSBX`を含む交叉演算子一覧
- [Mutation](../components/mutation.md) — `MutationPolynomial`を含む突然変異演算子一覧
- [ParentSelection](../components/parent_selection.md) — `TournamentSelection`の詳しい使い方
- [SurvivorSelection](../components/survivor_selection.md) — `TruncationSelection`の詳しい使い方
- [OptimizationStrategy](../components/strategies.md) — 独自Strategyの実装方法、`AskStage`の`n_offspring`
- [NonDominatedSorting](../components/nondominated_sorting.md) — 非優越ソートの実装詳細
- [Dominator](../components/dominance.md) — `dominator`引数として差し替え可能な支配述語一覧
