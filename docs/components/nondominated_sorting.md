# NonDominatedSorter

[ParetoComparator](comparators.md)系のComparator（`NSGA2Comparator`/`HypervolumeComparator`/`NSGA3Comparator`/`RNSGA2Comparator`/`EpsilonDominanceComparator`）は、個体群をフロントへ振り分ける処理を`sorter`という差し替え可能な引数に委ねています。
[Dominator](dominance.md)が「2点間の支配関係の定義」を担うのに対し、`sorter`は「その支配関係を使ってどうフロントに割り振るか」というもう1つの独立した差し替え軸です。

## NonDominatedSorterの役割

`NonDominatedSorter`は、抽象基底クラスではなく`Protocol`（構造的部分型）として定義されています。
`__call__(f, direction=None, *, dominator=None) -> tuple[ranks, fronts]`というシグネチャを満たす呼び出し可能オブジェクトであれば、クラスを継承しなくても`sorter`として渡せます。

`ranks`は各個体のフロント番号（`0`が最良）を表す配列、`fronts`は`fronts[i]`がフロント`i`に属する個体のインデックスリストであるという契約を満たします。

## 組み込みNonDominatedSorter

`non_dominated_sort`/`dda_non_dominated_sort`はいずれも、上記の`(ranks, fronts)`契約を満たす関数です（クラスとしての階層は持たない）。

| 関数 | アルゴリズム |
|---|---|
| `non_dominated_sort` | {cite}`deb2002nsga2`の非優越ソート。フロントを1つずつ剥がしていく方式（front-peeling） |
| `dda_non_dominated_sort` | Dominance-Degree Approach（Zhou et al., 2017がdominance degree matrixを、Mishra & Senwar, 2020がDDA-ENSのフロント割り当てを提案） |

`non_dominated_sort`の計算量はO(MN²)だが、支配行列をNumPyでベクトル化して構築するため実測では高速に動作します。
`dda_non_dominated_sort`は、`non_dominated_sort`と完全に同一の`(ranks, fronts)`を返すことが保証されているそのまま置き換え可能な代替関数であり、個体数`N`や目的数`M`が大きい場合（`M > 100`）のスケーラビリティ向けに用意されています。

両者とも、NaNを含む行の扱いは共通しています。
NaNを含む行は、通常のフロント分割から除外され、最終フロントの後にセンチネルフロントとして1個体ずつ追加されます。

`sorter`と`dominator`は独立した差し替え軸です。
`non_dominated_sort`/`dda_non_dominated_sort`はどちらも、内部で`dominator.dominance_matrix()`を呼ぶだけで、支配関係の定義自体には関与しません。
`dominator`を省略すると`ParetoDominator`が既定として使われます。

## 補助関数

Pareto系Comparatorの内部実装で使われる補助関数も、公開APIとして単体で利用できます。

**`crowding_distance(f_front)`**：単一フロント内の混雑度距離を計算します。`NSGA2Comparator`が使います。
境界解（各目的の最小値と最大値を取る解）には`inf`が割り当てられます。

**`crowding_distance_all_fronts(f, fronts)`**：`non_dominated_sort`が返す全フロントに対して`crowding_distance`を適用します。

**`spea2_fitness(f, direction=None, dominator=None)`**：`SPEA2Comparator`が使うfitness計算{cite}`zitzler2001spea2`。
strength（支配する個体数）、raw fitness（自分を支配する個体群のstrengthの総和）、density（k近傍距離の逆数）の3要素から算出します。

```{warning}
`spea2_fitness`の戻り値は「低いほど良い」という慣習であり、`saealib`全体の「高いほど良い」というスコア規約とは逆になっています。
他のComparatorへそのまま渡さないよう注意してください。
```

## NonDominatedSorterの拡張方法

`NonDominatedSorter`はProtocolであるため、`(ranks, fronts)`契約を満たす関数を1つ書くだけで独自実装として使えます。
基底クラスを継承する必要はありません。

```python
import numpy as np
from saealib import non_dominated_sort


def logged_non_dominated_sort(f, direction=None, *, dominator=None):
    """既存の実装に前処理を挟むだけの、Protocolを満たす最小の関数。"""
    print(f"sorting {len(f)} individuals")
    return non_dominated_sort(f, direction, dominator=dominator)
```

`ParetoComparator(sorter=logged_non_dominated_sort, ...)`のように渡せば、既存の`Comparator`実装を変更せずにソート方式を差し替えられます。

## 関連コンポーネント

- [Dominator](dominance.md) — `sorter`と対になる、2点間の支配関係の定義
- [Comparator](comparators.md) — `sorter`引数を持つ`ParetoComparator`系のComparator

## 参照

- {py:class}`saealib.NonDominatedSorter`
- {py:func}`saealib.non_dominated_sort`
- {py:func}`saealib.dda_non_dominated_sort`
- {py:func}`saealib.crowding_distance`
- {py:func}`saealib.crowding_distance_all_fronts`
- {py:func}`saealib.spea2_fitness`
