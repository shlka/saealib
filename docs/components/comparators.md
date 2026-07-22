# Comparator

`Problem`は、解の優劣を判断する処理を`Comparator`という差し替え可能なトップレベルコンポーネントに委ねています。
`Problem`の`comparator`引数で渡します。

## Comparatorの役割

`Comparator`は、`__init__`自体を含めて4つのメソッド全てが抽象メソッドです。

- **`__init__(weights, eps_cv, eps_obj, direction=None)`**：`weights`/`eps_cv`/`eps_obj`/`direction`を保持します
- **`sort_population(population) -> np.ndarray`**：個体群全体を優劣順に並べたインデックス配列を返します
- **`compare_population(population, idx_a, idx_b) -> int`**：個体群中の2個体を比較します（`-1`：aが優れる、`1`：bが優れる、`0`：同等）
- **`compare(fa, cv_a, fb, cv_b) -> int`**：`Population`を介さず、目的関数値と制約違反から直接2点を比較する軽量版

`__init__`が抽象メソッドである点は他のコンポーネントと異なります。
独自の`Comparator`を実装するサブクラスは、必ず`super().__init__(weights, eps_cv, eps_obj, direction=...)`を呼ばなければなりません。

## 組み込みComparator

| クラス | 使いどころ |
|---|---|
| `SingleObjectiveComparator` | 単目的問題 |
| `WeightedSumComparator` | 重み付き線形結合によるスカラー化。単目的でも多目的でも使える |
| `ParetoComparator` | 支配関係のみでランク付け（混雑度などの副次的な指標を持たない） |
| `NSGA2Comparator` | Paretoランク＋混雑度距離{cite}`deb2002nsga2` |
| `SPEA2Comparator` | 個体群全体に依存するSPEA2 fitness{cite}`zitzler2001spea2` |
| `HypervolumeComparator` | フロントランク＋排他的HV寄与度（SMS-EMOA風）{cite}`beume2007smsemoa` |
| `EpsilonDominanceComparator` | ε-支配によるランク付け{cite}`laumanns2002epsilon` |
| `NSGA3Comparator` | 参照点によるニッチ保存{cite}`deb2014nsga3` |
| `RNSGA2Comparator` | ユーザー指定の参照点による選好誘導{cite}`deb2006rnsga2` |

`SingleObjectiveComparator(direction=None, *, eps_cv=1e-6, eps_obj=1e-6)`は`direction`を省略でき、その場合は最小化として扱います。

`WeightedSumComparator(direction, *, eps_cv=1e-6, eps_obj=1e-6)`は`direction`が必須で、省略すると`TypeError`になります。
このクラスに限り、渡した`direction`がそのままスカラー化の重みとして使われます（`score = f @ direction`）。
符号だけでなく大きさも重みとして機能するため、[Problem](problem.md)で述べた「`direction`は符号のみ、重みの大きさは別の概念」という一般的な役割分担とは異なる、このクラス固有の扱いです。

`ParetoComparator(direction=None, *, eps_cv=1e-6, eps_obj=1e-6, sorter=non_dominated_sort, dominator=None)`は、支配関係のみで個体群をランク付けします。
`NSGA2Comparator`/`HypervolumeComparator`/`NSGA3Comparator`/`RNSGA2Comparator`/`EpsilonDominanceComparator`の共通基底であり、具象クラスとして単独でも使えます。
`dominator`引数は[Dominator](dominance.md)、`sorter`引数は[NonDominatedSorter](nondominated_sorting.md)を注入する差し替え点で、互いに独立した軸です。

`NSGA2Comparator`は、`ParetoComparator`に混雑度距離による副次的な順位付けを加えます。
ソート結果は`Population`のキャッシュ（`get_cache`/`set_cache`）に保存され、個体群が変更されるまで世代内で使い回されます。

### 個体群相対的なComparator

`SPEA2Comparator`と`HypervolumeComparator`は、いずれも`compare()`を呼ぶと`NotImplementedError`を送出します。
SPEA2のfitnessも排他的HV寄与度も、個体群全体に依存する指標であり、2点だけからは計算できないためです。
これはバグではなく意図した設計であり、`is_population_relative=True`というクラス属性がその旨を示すマーカーになっています。

PSOのpbest更新や`PairwiseComparisonSet`のように、2点だけの比較（`compare()`）が必要な場面でこれらのComparatorは使えません。
そのような場面では代わりに`ParetoComparator`を使います。
`compare_population()`（個体群のインデックスを介した比較）はどちらのクラスでも定義されているため、トーナメント選択などはそのまま使えます。

`HypervolumeComparator`のHV計算は、フロットごとにO(N)回のleave-one-out評価を行います。
目的数が多い問題では、この計算コストが大きくなります。

```{note}
`HypervolumeComparator`の内部実装とは別に、`saealib.hypervolume(f, reference_point)`という公開関数があります。
これは最適化後の結果を評価する性能指標として単体で使えるもので、`HypervolumeComparator`とは無関係です。
詳細は[Utils](../api/utils.md)を参照してください。
```

### 参照点を使うComparator

`NSGA3Comparator(reference_points, direction=None, *, ...)`は、`reference_points`（`shape (n_ref, n_obj)`、単体シンプレックス上の点）が必須引数です。
通常は`saealib.utils.weight_vectors.uniform_weight_vectors(n_obj, n_divisions)`で一様に生成したものを渡します。
`rng`プロパティは遅延生成され、`Optimizer`実行時は`Runner`が`ctx.rng`からspawnした乱数生成器を注入します。
この内部rngはチェックポイントの保存対象に含まれず、再開時は新しくspawnし直されます。

`RNSGA2Comparator(reference_points, epsilon=0.001, direction=None, *, ...)`は、`NSGA3Comparator`とは異なり参照点が単体シンプレックス上にある必要はなく、ユーザーが望む目的関数値（aspiration point）をそのまま指定できます。
`epsilon`は、同じ参照点に近い解同士を間引くε-clearingの半径です。

`EpsilonDominanceComparator(eps, mode="additive", direction=None, *, ...)`は、`ParetoComparator`の`dominator`を[EpsilonDominator](dominance.md)に差し替えるだけの薄いラッパーです。

[DecompositionComparator](decomposition.md)は、MOEA/D風のスカラー化によるランク付けを行うComparatorです。
詳細はそちらのページで扱います。

## 独自Comparatorの実装方法

独自の順位付け方式が必要な場合は、`Comparator`を継承して4つのメソッド全てを実装します。
`__init__`は必ず`super().__init__(weights, eps_cv, eps_obj, direction=...)`を呼びます。

```python
import numpy as np
from saealib import Comparator


class RandomComparator(Comparator):
    """常に実行可能性のみを考慮する単純な例。"""

    def __init__(self, direction=None, *, eps_cv=1e-6, eps_obj=1e-6):
        super().__init__(np.empty(0), eps_cv, eps_obj, direction=direction)

    def sort_population(self, population):
        cv = population.get_array("cv")
        return np.argsort(cv)

    def compare_population(self, population, idx_a, idx_b):
        cv = population.get_array("cv")
        return self.compare(None, cv[idx_a], None, cv[idx_b])

    def compare(self, fa, cv_a, fb, cv_b):
        if cv_a > self.eps_cv and cv_b <= self.eps_cv:
            return 1
        if cv_b > self.eps_cv and cv_a <= self.eps_cv:
            return -1
        return 0
```

`SPEA2Comparator`/`HypervolumeComparator`のように、個体群全体に依存する指標を実装する場合は、`is_population_relative = True`というクラス属性を立て、`compare()`で理由を説明する`NotImplementedError`を送出する設計パターンが使えます。

## 関連コンポーネント

- [Dominator](dominance.md) — `ParetoComparator`系の`dominator`引数
- [NonDominatedSorter](nondominated_sorting.md) — `ParetoComparator`系の`sorter`引数
- [Decomposition](decomposition.md) — `DecompositionComparator`が使うスカラー化関数
- [Problem](problem.md) — `comparator`引数の渡し方、既定選択のルール
- [ParentSelection](parent_selection.md) / [SurvivorSelection](survivor_selection.md) — `Comparator`を使って個体を選ぶ演算子

## 参照

- {py:class}`saealib.Comparator`
- {py:class}`saealib.SingleObjectiveComparator`
- {py:class}`saealib.WeightedSumComparator`
- {py:class}`saealib.ParetoComparator`
- {py:class}`saealib.NSGA2Comparator`
- {py:class}`saealib.SPEA2Comparator`
- {py:class}`saealib.HypervolumeComparator`
- {py:class}`saealib.EpsilonDominanceComparator`
- {py:class}`saealib.NSGA3Comparator`
- {py:class}`saealib.RNSGA2Comparator`
- {py:func}`saealib.hypervolume`
