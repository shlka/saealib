# Dominator

[ParetoComparator](comparators.md)系のComparator（`NSGA2Comparator`/`HypervolumeComparator`/`NSGA3Comparator`/`RNSGA2Comparator`/`EpsilonDominanceComparator`）は、支配関係の定義そのものを`Dominator`という差し替え可能なコンポーネントに委ねています。
Pareto支配以外の選好関係を使いたいときは、これらのComparator本体ではなく`dominator`引数だけを差し替えればよいです。

## Dominatorの役割

`Dominator`が実装を要求するメソッドは`dominance_matrix(f, direction=None)`の1つだけです。
`f`は`shape (n, n_obj)`の目的関数値の行列で、`D[i, j]`が「行`i`が行`j`を支配するか」を示す`shape (n, n)`のブール行列を返します。
`f`はNaNを含まない前提であり、呼び出し側がその保証を担います。

2点だけを比較する`dominates(fa, fb, direction=None) -> bool`も持ちますが、これは`dominance_matrix`から導出される既定実装であり、独自に`Dominator`を実装する際は`dominance_matrix`さえ実装すれば`dominates`との整合性が自動的に保証されます。
内部では`fa`/`fb`を2行のスタックにして`dominance_matrix`を呼びます。
`fa`にNaNが含まれる場合は常に`False`（支配しない）を返す一方、`fb`側のNaNは`+inf`に置き換えられ「常に支配される」個体として扱われる、という非対称な扱いになっています。

## 組み込みDominator

| クラス | パラメータ | 特徴 |
|---|---|---|
| `ParetoDominator` | なし | 標準Pareto支配。全目的で以下、かつ少なくとも1目的で真に小さい場合に支配とみなす（既定） |
| `EpsilonDominator` | `eps, mode="additive"` | ε-boxに量子化してから内部で`ParetoDominator`に委譲する |

`EpsilonDominator`は、{cite}`laumanns2002epsilon`のε-支配を実装したものです。
量子化モードは2種類あります。

- **additive**（既定）：各目的の箱番号を`floor(f_i / eps_i)`で計算する
- **multiplicative**：`floor(log(f_i) / log(1 + eps_i))`で計算する。目的関数値が全て正であることが前提で、そうでない場合は`ValueError`になる

`eps`はスカラーまたは`shape (n_obj,)`の配列で指定し、全要素が正でなければなりません（そうでない場合は構築時に`ValueError`）。

[EpsilonDominanceComparator](comparators.md)は、`ParetoComparator`の`dominator`引数に`EpsilonDominator(eps, mode)`を渡すだけの薄いラッパーです。

```{note}
モジュール内に残る`_pareto_dominates`/`_dominance_matrix`という関数はdeprecatedな後方互換ラッパーです。
新規コードでは`ParetoDominator().dominates(...)`/`ParetoDominator().dominance_matrix(...)`を直接使ってください。
```

## 独自Dominatorの実装方法

独自の支配関係が必要な場合は、`Dominator`を継承して`dominance_matrix()`だけを実装すればよいです。
次の例は、各目的にスケーリング係数をかけてからPareto支配を適用します。

```python
import numpy as np
from saealib import Dominator, ParetoDominator


class WeightedDominator(Dominator):
    """各目的にスケーリング係数をかけてから通常のPareto支配を適用する。"""

    def __init__(self, scale):
        self.scale = np.asarray(scale, dtype=float)
        self._pareto = ParetoDominator()

    def dominance_matrix(self, f, direction=None):
        return self._pareto.dominance_matrix(f * self.scale, direction)
```

`EpsilonDominator`と同様に、既存の`ParetoDominator`へ変換後の値を委譲する実装にすると、直接目的関数値を比較するロジックを自分で書かずに済みます。

## 関連コンポーネント

- [Comparator](comparators.md) — `ParetoComparator`系の`dominator`引数から`Dominator`を注入する
- [NonDominatedSorter](nondominated_sorting.md) — `Dominator`と対になる、フロントへの振り分け方の差し替え点
- [Population](population.md) — `ParetoArchive`が非優越解判定に`Dominator`を使う

## 参照

- {py:class}`saealib.Dominator`
- {py:class}`saealib.ParetoDominator`
- {py:class}`saealib.EpsilonDominator`
