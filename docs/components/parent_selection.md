# ParentSelection

`GA`(`saealib.GA`)は、交叉に使う親個体を個体群から選ぶ処理を、`ParentSelection`という差し替え可能な演算子に委ねている。
選択圧（優れた個体をどれだけ優遇するか）を変えたいときは、`GA`本体ではなくこの`ParentSelection`だけを差し替えればよい。

## ParentSelectionの役割

`ParentSelection`が実装を要求するメソッドは`select(ctx, population, n_pair, n_parents, rng=...)`の1つだけである。
`n_pair`組、それぞれ`n_parents`個体からなる親のインデックスを、`shape (n_pair, n_parents)`の配列で返す。

## 組み込みParentSelection

| クラス | パラメータ | 特徴 |
|---|---|---|
| `TournamentSelection` | `tournament_size` | `tournament_size`個体をランダムに抽出し、最良個体を選ぶ操作を`n_pair * n_parents`回繰り返す{cite}`miller1995tournament` |
| `SequentialSelection` | なし | 比較を行わず、個体群のインデックスを順番に割り当てるだけ |
| `RouletteWheelSelection` | なし | 順位に基づく線形ランクベースの確率でルーレット選択する |

`TournamentSelection`は、`ctx.comparator.compare_population`による1対1比較を繰り返してトーナメント内の最良個体を選ぶ{cite}`blickle1996selection`。
`compare_population`は[SPEA2Comparator](comparators.md)や[HypervolumeComparator](comparators.md)のように2点だけの直接比較（`compare()`）が使えないComparatorでも定義されているため、どの`Comparator`と組み合わせても問題なく動作する。

`SequentialSelection`は比較を一切行わないため、選択圧という概念自体を持たない最も単純な選択方式である。
どの`Comparator`とも組み合わせられる。

`RouletteWheelSelection`は、生の適応度ではなく`ctx.comparator.sort_population`が返す順位を確率に変換する。
これにより、目的関数値が負の値を取ったりスケールが大きく異なったりする問題でも、数値的な問題を起こさずに選択確率を計算できる。

```{note}
`SequentialSelection`のみ`@register()`済みで、`TournamentSelection`/`RouletteWheelSelection`は現状Registry未登録である。
Registry経由でクラスを文字列から解決する使い方をする場合はこの違いに注意する。
```

## 独自ParentSelectionの実装方法

独自の選択方式が必要な場合は、`ParentSelection`を継承して`select()`だけを実装すればよい。
次の例は、比較を行わず親を完全にランダムに選ぶ選択方式である。

```python
import numpy as np
from saealib import ParentSelection


class RandomPairSelection(ParentSelection):
    """親個体を完全ランダムに選ぶ選択方式。"""

    def select(self, ctx, population, n_pair, n_parents, rng=np.random.default_rng()):
        n_pop = len(population)
        return rng.integers(0, n_pop, size=(n_pair, n_parents))
```

`ctx.comparator`を参照する実装にすれば、`TournamentSelection`のように選択圧を持つ独自方式にもできる。

## 関連コンポーネント

- [Algorithm](algorithm.md) — `GA`が`ParentSelection`をどう組み合わせるか
- [Crossover](crossover.md) — `ParentSelection`が選んだ親個体を受け取る演算子
- [Comparator](comparators.md) — `TournamentSelection`/`RouletteWheelSelection`が個体を比較する際に使う

## 参照

- {py:class}`saealib.ParentSelection`
- {py:class}`saealib.TournamentSelection`
- {py:class}`saealib.SequentialSelection`
- {py:class}`saealib.RouletteWheelSelection`
