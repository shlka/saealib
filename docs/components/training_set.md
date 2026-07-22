# TrainingSet

`GlobalSurrogateManager`/`LocalSurrogateManager`/`PairwiseSurrogateManager`は、学習データの構築を`TrainingSet`という差し替え可能なコンポーネントに委ねています。
`training_set`引数に渡します。

## TrainingSetの役割

`TrainingSet`が実装を要求するメソッドは`build(archive, population, ctx, candidate_x=None) -> TrainingData`の1つだけです。
`candidate_x`は`LocalSurrogateManager`が候補ごとのk-NNクエリ中心として渡す引数で、`GlobalSurrogateManager`からは`None`で呼ばれます。

`TrainingData`は`train_x`（通常`shape (n_train, dim)`。`PairwiseComparisonSet`のみ`(n_train, 2*dim)`）と`train_y`（回帰なら`(n_train, n_obj)`、分類やランキングなら`(n_train,)`）を持つデータクラスです。

組み込みの8クラスは、次の2つの直交する軸で整理されています。

- **データソース軸**：archive / population / k近傍 / ペア / 参照点のどこから学習データを取るか
- **ラベリング軸**：生の目的関数値（回帰）/ 二値分類 / 多段階ランキング / ペア比較のどう値を割り当てるか

## 文献パターンとの対応

| パターン | 文献 | 対応クラス |
|---|---|---|
| P1 CA-LLSO | {cite}`wei2021callso` | `LevelBasedSet` |
| P2 CPS-MOEA | {cite}`zhang2018cpsmoea` | `TopKBipartitionSet` |
| P3 Pairwise SAEA | {cite}`hao2024pairwise` | `PairwiseComparisonSet` |
| P4 SAPSO pbest | {cite}`tian2019sapso` | `ReferencePointComparisonSet` |
| P5 CSEA / pre-selection | （一般） | `KNNObjectiveSet`, `ArchiveObjectiveSet` |
| P6 Constraint BO | {cite}`regis2005cors,letham2019constraintbo` | `ConstraintObjectiveSet`, `KNNConstraintObjectiveSet` |

## 組み込みTrainingSet

| クラス | パラメータ | 内容 |
|---|---|---|
| `ArchiveObjectiveSet` | なし | アーカイブ全体を生の目的関数値で使う。`GlobalSurrogateManager`の既定 |
| `KNNObjectiveSet` | `n_neighbors=50` | `candidate_x`のk近傍アーカイブ点。`LocalSurrogateManager`の既定 |
| `ConstraintObjectiveSet` | なし | アーカイブ全体を生の制約値`g`で使う |
| `KNNConstraintObjectiveSet` | `n_neighbors=50` | `ConstraintObjectiveSet`のk-NN版 |
| `FeasibilityClassificationSet` | `source="archive"` | `cv <= eps_cv`による二値分類ラベル |
| `TopKBipartitionSet` | `source="archive", top_ratio=0.5` | ソート後、上位`floor(n * top_ratio)`件をlabel=1、残りをlabel=0とする二値分類ラベル |
| `LevelBasedSet` | `source="population", n_levels=5` | ソート後、`n_levels`個の等分割グループへ多段階ラベル付けする |
| `PairwiseComparisonSet` | `source="archive", n_pairs=None, rng=None` | 2点をペアにして比較した勝敗をラベルとする |
| `ReferencePointComparisonSet` | `ref_source="population_best"` | アーカイブ点が参照点を支配するかどうかの二値ラベル |

`ConstraintObjectiveSet`/`KNNConstraintObjectiveSet`は、問題が制約を持たない場合（`archive.g`が0列）は`ValueError`になります。

`FeasibilityClassificationSet`の実行可能性判定に使う`eps_cv`は`ctx.problem.eps_cv`から取得され、`ctx=None`のときは`1e-6`が使われます。

`source`引数を持つクラス（`FeasibilityClassificationSet`/`TopKBipartitionSet`/`LevelBasedSet`/`PairwiseComparisonSet`）は共通して、`source="population"`を指定したときに`population=None`だと`ValueError`になります。

`PairwiseComparisonSet`は、2点`(a, b)`ごとに`train_x = [x_a, x_b]`を連結した`shape (n_pairs, 2*dim)`の配列を作り、`comparator.compare(f_a, cv_a, f_b, cv_b) <= 0`（aがbに勝つか同等）なら`1`、そうでなければ`0`をラベルとします。
`n_pairs=None`の場合は全ペア`n*(n-1)/2`を使います。

```{warning}
`PairwiseComparisonSet`の`train_x`は`(n_pairs, 2*dim)`という特殊な形状であり、`RBFSurrogate`のような標準的な回帰サロゲートとは形状が非互換です。
[ComparisonSurrogate](surrogate.md)系のペア比較専用サロゲートと組み合わせる必要があります。
```

`ReferencePointComparisonSet`は、`PairwiseComparisonSet`と異なり`train_x`が`(n_archive, dim)`のみなので、`GlobalSurrogateManager`/`LocalSurrogateManager`と互換性があります。

## 独自TrainingSetの実装方法

独自の学習データ抽出方式が必要な場合は、`TrainingSet`を継承して`build()`だけを実装すればよいです。
次の例は、直近に追加された`k`件だけを学習データとして使う実装です。

```python
from saealib import TrainingSet, TrainingData


class RecentKSet(TrainingSet):
    """直近に追加されたk件だけを学習データとして使う。"""

    def __init__(self, k: int = 20):
        self.k = k

    def build(self, archive, population, ctx, candidate_x=None):
        x = archive.get_array("x")[-self.k:]
        y = archive.get_array("f")[-self.k:]
        return TrainingData(train_x=x, train_y=y)
```

## 関連コンポーネント

- [SurrogateManager](surrogate_manager.md) — `training_set`引数を持つマネージャー
- [Surrogate](surrogate.md) — `TrainingData`を渡す先。`PairwiseComparisonSet`は`ComparisonSurrogate`系との組み合わせが必要
- [Comparator](comparators.md) — `TopKBipartitionSet`/`LevelBasedSet`/`PairwiseComparisonSet`/`ReferencePointComparisonSet`が使うソートと比較

## 参照

- {py:class}`saealib.TrainingSet`
- {py:class}`saealib.TrainingData`
- {py:class}`saealib.ArchiveObjectiveSet`
- {py:class}`saealib.KNNObjectiveSet`
- {py:class}`saealib.ConstraintObjectiveSet`
- {py:class}`saealib.KNNConstraintObjectiveSet`
- {py:class}`saealib.FeasibilityClassificationSet`
- {py:class}`saealib.TopKBipartitionSet`
- {py:class}`saealib.LevelBasedSet`
- {py:class}`saealib.PairwiseComparisonSet`
- {py:class}`saealib.ReferencePointComparisonSet`
