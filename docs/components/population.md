# Population

`saealib`のアルゴリズムは、個体群を`Population`という構造化配列コンテナとして扱います。
`Initializer`が実行開始時に構築し、以後は`OptimizationState`の`population`/`archive`/`pareto_archive`フィールドとして各コンポーネントに共有されます。

## Populationが表すもの

`Population`は、設計変数`x`、目的関数値`f`、制約値`g`、制約違反`cv`、アルゴリズム固有の付加属性を、列ごとの配列として保持するコンテナです。
`Generic[T_Individual]`だが、通常は継承せずインスタンスとして使います。

保持する属性のスキーマは`PopulationAttribute(name, dtype, shape, default)`のリストで定義します。
`x`/`f`/`g`/`cv`という標準属性に加えて、`Algorithm.get_required_attrs(problem)`が返すアルゴリズム固有の属性（PSOの速度やpbestなど）が[Initializer](initialization.md)によって動的に組み立てられ、このスキーマに反映されます。

## 主要な属性とメソッド

| メソッド | 役割 |
|---|---|
| `get_array(key)` / `get_readonly_array(key)` | 属性の生配列を取得する。後者は書き込み禁止のviewを返す |
| `update_array(key, value)` | 属性配列を一括更新する |
| `get(key, default=None)` | 存在しない属性なら`default`を返す、安全な取得 |
| `append(element=None, **kwargs)` | 個体を1件追加する |
| `extend(other)` | 別の`Population`またはdictから個体群をまとめて追加する |
| `extract(indices)` | インデックス配列/スライスで部分集合を新しい`Population`として取り出す |
| `truncate(new_size)` / `delete(index)` / `clear()` | サイズ変更と削除 |
| `reorder(order)` / `argsort(name, reverse=False)` | 並べ替え |
| `empty_like(capacity=None)` | 同じスキーマの空`Population`を作る |
| `set_cache(key, value)` / `get_cache(key)` | 計算結果をPopulation変更まで有効なキャッシュとして保持する |
| `pop[i]` / `pop[a:b]` | 単一intなら`Individual`を、sliceなら部分集合の`Population`を返す |
| `len(pop)` | 個体数 |

`set_cache`/`get_cache`によるキャッシュは、`append`/`delete`/`update_array`など個体群を変更する操作を呼ぶたびに自動的に無効化されます。
[NSGA2Comparator](comparators.md)がフロント分割と混雑度距離の計算結果を世代内で使い回す際に、この仕組みを利用しています。

```python
import numpy as np
from saealib import Population, PopulationAttribute

attrs = [
    PopulationAttribute("x", np.float64, (2,)),
    PopulationAttribute("f", np.float64, (1,)),
    PopulationAttribute("cv", np.float64, ()),
]
pop = Population(attrs, init_capacity=4)
pop.append(x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0)
pop.append(x=np.array([0.3, 0.4]), f=np.array([2.0]), cv=0.0)

pop.x  # shape (2, 2) の設計変数配列
pop[0]  # 先頭個体の Individual ビュー
pop[0:1]  # 先頭1件だけの Population
```

### Individual

`Individual`は`pop[i]`で得られる、単一個体への軽量なビューです。
実データを複製せず、参照元の`Population`と自分のインデックスだけを保持します。

`get_readonly_value(key)`/`update_value(key, value)`で値の読み書きができるほか、`ind.x`/`ind.f = ...`のような属性アクセスでも同じ読み書きができます。
参照元の`Population`の構造（個体数や並び順）が変わった後に古い`Individual`を使うと、無効な参照として例外になります。

## Archive

`Archive`は、`ArchiveMixin`を`Population`にミックスインした具象クラスで、評価済み解を重複なく蓄積する目的で使います。

`add(element, **kwargs)`は`append`とほぼ同じ引数を取りますが、重複解を無視する点が異なります。
重複判定に使う属性は`key_attr`引数（既定`"x"`）で指定し、`atol`/`rtol`で許容誤差を調整します。
`get_knn(x, k)`はkd-tree（初回呼び出し時に遅延構築される）による近傍検索を提供し、[LocalSurrogateManager](surrogate_manager.md)の既定`training_set`が候補ごとの局所学習データを集める際に使います。

```python
from saealib import Archive

arc = Archive(attrs, init_capacity=4, key_attr="x")
arc.add(x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0)
arc.add(x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0)  # 重複解は無視される
idx, dist = arc.get_knn(np.array([0.1, 0.2]), k=1)
```

## ParetoArchive

`ParetoArchive`は、`ParetoMixin`を`Population`にミックスインした具象クラスで、非優越解集合を常時維持します。

新規解を追加するたびに、その解に支配される既存解を削除し、新規解自体が既存解に支配されている場合はその新規解を破棄します。
支配関係の判定はfeasibility-first方式で行われます。
実行可能解（`cv <= eps_cv`）は常に実行不可能解を支配し、両方が実行可能な場合にのみ[Dominator](dominance.md)の`dominates`が使われます。

`dominator`引数で支配関係の定義を差し替えられます。
`eps_cv`の既定値は`0.0`（厳密に実行可能な解のみを許容可能とみなす）だが、`Optimizer`実行中はこの値が毎世代`problem.handler.feasibility_threshold`で上書きされます。
`0.0`という既定値は、`ParetoArchive`を`Optimizer`から切り離して単体で使う場合にのみ意味を持ちます。

## 限定的な拡張点

`ArchiveMixin`/`ParetoMixin`は、`Population`のサブクラスに多重継承でミックスインするという前提で設計されています。
独自の集団管理ロジックが必要な場合、これらのMixinを組み合わせた新しいクラス（`class MyArchive(ArchiveMixin, Population): ...`）を定義できます。
また[Algorithm](algorithm.md)の`population_class`/`archive_class`をオーバーライドすれば、`Initializer`が生成するPopulation/Archiveを独自サブクラスに差し替えられます。

## 関連コンポーネント

- [Initializer](initialization.md) — `Population`/`Archive`/`ParetoArchive`を実行開始時に構築する
- [OptimizationState](optimization_state.md) — 構築後の`Population`/`Archive`/`ParetoArchive`を保持する状態オブジェクト
- [Algorithm](algorithm.md) — `population_class`/`archive_class`で具象クラスを差し替える
- [Comparator](comparators.md) — `set_cache`/`get_cache`によるソート結果の使い回し
- [Dominance](dominance.md) — `ParetoArchive`が非優越解判定に使う`Dominator`
- [SurrogateManager](surrogate_manager.md) — `Archive.get_knn`を使う局所学習データの収集

## 参照

- {py:class}`saealib.Population`
- {py:class}`saealib.Individual`
- {py:class}`saealib.PopulationAttribute`
- {py:class}`saealib.Archive`
- {py:class}`saealib.ArchiveMixin`
- {py:class}`saealib.ParetoArchive`
- {py:class}`saealib.ParetoMixin`
