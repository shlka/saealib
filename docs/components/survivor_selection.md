# SurvivorSelection

`GA`(`saealib.GA`)は、次世代に残す個体を選択プールから選ぶ処理を、`SurvivorSelection`という差し替え可能な演算子に委ねています。
世代交代の方式を変えたいときは、`GA`本体ではなくこの`SurvivorSelection`だけを差し替えればよいです。

## SurvivorSelectionの役割

`SurvivorSelection`が実装を要求するメソッドは`select(ctx, pool, n_survivors) -> np.ndarray`の1つだけです。
`pool`は、親個体群と子個体群など`Algorithm`側が構築した統合済みの`Population`で、その中から生き残る`n_survivors`件のインデックスを返します。

`(μ+λ)`方式（親と子を合わせたプールから選ぶ）か`(μ,λ)`方式（子だけのプールから選ぶ）かは、`pool`に何を含めるかという`Algorithm`側の構築方法で決まります。
`SurvivorSelection`のインターフェース自体はどちらの方式かを区別しません。

## 組み込みSurvivorSelection

| クラス | パラメータ | 特徴 |
|---|---|---|
| `TruncationSelection` | `randomize_ties=False` | `ctx.comparator.sort_population(pool)`でソートし、上位`n_survivors`件を採用する打ち切り選択 |

`randomize_ties=True`にすると、打ち切り境界で同点（`compare_population`が`0`を返す）の個体群をシャッフルしてから打ち切ります。
既定の`False`では、`sort_population`が返す順序をそのまま使う決定的な打ち切りになります。
このタイブレーク処理は`ctx.rng`を消費するため、`randomize_ties=True`を使う場合はチェックポイント再開時の乱数状態にも影響する点に注意してください。

`TruncationSelection`は`@register()`済みです。

## 独自SurvivorSelectionの実装方法

独自の世代交代方式が必要な場合は、`SurvivorSelection`を継承して`select()`だけを実装すればよいです。
次の例は、最良個体1件を必ず残し、残りはランダムに選ぶ生存選択です。

```python
import numpy as np
from saealib import SurvivorSelection


class ElitistSurvivorSelection(SurvivorSelection):
    """最良個体1件を必ず残し、残りはランダムに選ぶ。"""

    def select(self, ctx, pool, n_survivors):
        sorted_idx = ctx.comparator.sort_population(pool)
        best = sorted_idx[:1]
        rest_pool = sorted_idx[1:]
        rest = ctx.rng.choice(rest_pool, size=n_survivors - 1, replace=False)
        return np.concatenate([best, rest])
```

トーナメント式の生存選択や年齢ベースの入れ替えのように、`sort_population`によるランキングを前提としない方式も、同じ`select()`のシグネチャの範囲で実装できます。

## 関連コンポーネント

- [Algorithm](algorithm.md) — `GA.tell()`が`pool`をどう構築し`SurvivorSelection`をどう呼ぶか
- [Comparator](comparators.md) — `sort_population`/`compare_population`による個体の順位付け

## 参照

- {py:class}`saealib.SurvivorSelection`
- {py:class}`saealib.TruncationSelection`
