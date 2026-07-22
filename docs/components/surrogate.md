# Surrogate

`saealib`は、目的関数を近似する予測モデルの責務を`Surrogate`という差し替え可能なコンポーネントに限定している。
`Surrogate`はfit/predictだけを知り、予測値をどうスコアに変換するか（[AcquisitionFunction](acquisition_functions.md)）や、学習データをどこから集めるか（[TrainingSet](training_set.md)、[SurrogateManager](surrogate_manager.md)経由）は一切知らない。

## Surrogateの役割

`Surrogate`が実装を要求するメソッドは2つある。

**`fit(train_x, train_y) -> None`**：`shape (n_samples, n_features)`の入力と`shape (n_samples, n_obj)`（単目的なら`(n_samples,)`も可）の出力でモデルを学習する。

**`predict(test_x) -> SurrogatePrediction`**：`shape (n_samples, n_features)`の入力に対する予測を返す。

クラス属性`provides_uncertainty: bool = False`は、予測が不確実性（標準偏差）を伴うかどうかを示す。
既定は`False`で、ガウス過程の実装だけが`True`にオーバーライドする。

`Surrogate`と`predict()`の間には2つのマーカー基底クラスがある。

**`RegressionSurrogate`**：`train_y`が実数値の目的関数出力である回帰サロゲート向けのマーカー。

**`ComparisonSurrogate`**：`train_y`が`{0, 1}`の二値比較ラベルである比較サロゲート向け。
`predict_proba(test_x) -> SurrogatePrediction`（値は`[0, 1]`の勝率）が主インターフェースで、`predict()`は既定で`predict_proba()`に委譲する。

## SurrogatePrediction

`predict()`の戻り値は`SurrogatePrediction`という統一データクラスである。

| フィールド | 内容 |
|---|---|
| `value` | 予測値。`shape (n_samples, n_obj)` |
| `std` | 不確実性（標準偏差）。提供しないサロゲートでは`None` |
| `label` | 分類モデルのみが持つクラスラベル |
| `tell_f` | `algorithm.tell()`を呼ぶ前に子個体の`f`へ代入する値（プロパティ） |
| `metadata` | SHAP値など実装固有の付加情報を格納する`dict` |

`tell_f`は、コンストラクタ引数`_tell_f`が未設定なら`value`にフォールバックするプロパティである。
`has_uncertainty`/`has_label`/`has_tell_f`という3つの真偽値プロパティで、それぞれの値が設定されているかを確認できる。

`_tell_f`にNaN配列を明示的に渡すと、`value`をそのまま`tell_f`として使わせないようにできる。
サロゲートの予測値が目的関数値ではない量（novelty scoreなど）を表す場合、この仕組みでpbestなどの汚染を防ぐ。
[SurrogateManager](surrogate_manager.md)の`ArchiveBasedManager`系がこの手法を使う。

## 組み込みSurrogate

**`RBFSurrogate(kernel, dim)`**：RBF補間によるサロゲート{cite}`gutmann2001rbf,regis2005cors`（RBF補間自体の起源はHardy, 1971）。
`gaussian_kernel(x1, x2, sigma=2.0)`が既定で使われるカーネルだが、`kernel`引数はカーネル関数を公開APIとして受け取る設計であり、ユーザーが任意のカーネルを注入できる。
`predict()`は`std=None`を明示的に返す（RBF補間は不確実性を提供しない）。

**`PerObjectiveSurrogate(surrogates)`**：`RegressionSurrogate`のサブクラスで、目的ごとに異なるサロゲートを割り当てる合成クラス。
`fit`時に`train_y`の列数と`len(surrogates)`が一致しないと`ValueError`になる。
`provides_uncertainty`は、構成する全サロゲートが`True`の場合のみ`True`を返す複合判定になっている。

### 外部ライブラリアダプタ

scikit-learn互換API経由の回帰サロゲートには`Sklearn`という接頭辞が付く。

| クラス | モデル |
|---|---|
| `SklearnGPRSurrogate` | Gaussian Process{cite}`sacks1989dace,rasmussen2006gpml`。`provides_uncertainty=True`の唯一の実装 |
| `SklearnRFRSurrogate` | Random Forest回帰 |
| `SklearnSVMSurrogate` | SVM |
| `SklearnNNSurrogate` | MLP |
| `SklearnXGBSurrogate` | XGBoost（`xgboost` extra） |
| `SklearnLGBMSurrogate` | LightGBM（`lightgbm` extra） |
| `TorchSurrogate` | PyTorchベースのモデル（`torch` extra） |

`SklearnGPRSurrogate`は`return_std=True`でGPカーネルから標準偏差を計算し、`provides_uncertainty=True`を返す。

実行可能性予測向けの分類サロゲートには次のクラスがある。

| クラス | モデル |
|---|---|
| `SklearnClassificationSurrogate` | scikit-learn互換の分類モデル全般 |
| `SklearnRFCClassificationSurrogate` | Random Forest分類 |
| `SklearnSVCClassificationSurrogate` | SVM分類 |

これらの分類サロゲートの学習データ抽出方法は[TrainingSet](training_set.md)の`FeasibilityClassificationSet`を参照する。
ペア比較には、これらの分類サロゲートではなく`ComparisonSurrogate`系の専用実装を使い、[SurrogateManager](surrogate_manager.md)の`PairwiseSurrogateManager`および`PairwiseComparisonSet`と組み合わせる。

各extraのインストール方法は[インストール](../getting_started/installation.md)を参照する。

```{note}
scikit-learn/XGBoost/LightGBM/PyTorch以外のBoTorch/SMTアダプタ、およびpymooのアルゴリズム/演算子アダプタは、現状`saealib`に実装されていない。
`pyproject.toml`にも対応するextraは存在しない。
```

## 拡張フック

フィット後の後処理だけを足したい場合、サブクラスを新設せずに`with_post_fit(fn)`で既存の`Surrogate`インスタンスへ処理を追加できる。
`with_post_fit`は元のインスタンスを変更せず、`fn`を追加したコピーを返す。

```python
from saealib import RBFSurrogate, gaussian_kernel


def log_fit(train_x, train_y, ctx=None):
    print(f"fit on {len(train_x)} samples")


base = RBFSurrogate(gaussian_kernel, dim=2)
logged = base.with_post_fit(log_fit)
```

`fn`のシグネチャは`fn(train_x, train_y, ctx) -> None`である。

## 独自Surrogateの実装方法

独自の予測モデルが必要な場合は、`Surrogate`を継承して`fit()`/`predict()`を実装する。
回帰なら`RegressionSurrogate`、比較なら`ComparisonSurrogate`（`predict_proba()`を実装）を継承先に選ぶ。

次の例は、最近傍点の目的関数値をそのまま予測値として返す単純な回帰サロゲートである。

```python
import numpy as np
from saealib import RegressionSurrogate, SurrogatePrediction


class NearestNeighborSurrogate(RegressionSurrogate):
    """最近傍点の目的関数値をそのまま予測値として返す単純なサロゲート。"""

    def fit(self, train_x, train_y):
        self.train_x = np.asarray(train_x, dtype=float)
        self.train_y = np.asarray(train_y, dtype=float)

    def predict(self, test_x):
        test_x = np.atleast_2d(test_x)
        dists = np.linalg.norm(self.train_x[None, :, :] - test_x[:, None, :], axis=2)
        nearest = dists.argmin(axis=1)
        value = self.train_y[nearest]
        return SurrogatePrediction(value=value)
```

## 不確実性対応表

不確実性ベースの[AcquisitionFunction](acquisition_functions.md)を使うには、`Surrogate`が`std`を返す必要がある。

| クラス | `provides_uncertainty` |
|---|---|
| `SklearnGPRSurrogate` | `True` |
| `RBFSurrogate` / `SklearnRFRSurrogate` / `SklearnSVMSurrogate` / `SklearnNNSurrogate` / `SklearnXGBSurrogate` / `SklearnLGBMSurrogate` / `TorchSurrogate` | `False` |
| `PerObjectiveSurrogate` | 構成する全サロゲートが`True`の場合のみ`True` |

`Optimizer.validate()`は、`AcquisitionFunction`の`requires_uncertainty`と`Surrogate`の`provides_uncertainty`の不整合を検出して警告する。

## 関連コンポーネント

- [SurrogateManager](surrogate_manager.md) — `Surrogate`のfit/predictを協調させ、スコアリングと組み合わせる
- [TrainingSet](training_set.md) — `Surrogate`に渡す学習データの抽出方法
- [AcquisitionFunction](acquisition_functions.md) — `predict()`の結果をスコアへ変換する
- [サロゲート精度評価と動的切り替え](surrogate_switching.md) — サロゲートの汎化性能の評価
- [インストール](../getting_started/installation.md) — 各extraのインストール方法

## 参照

- {py:class}`saealib.Surrogate`
- {py:class}`saealib.RegressionSurrogate`
- {py:class}`saealib.ComparisonSurrogate`
- {py:class}`saealib.SurrogatePrediction`
- {py:class}`saealib.RBFSurrogate`
- {py:func}`saealib.gaussian_kernel`
- {py:class}`saealib.PerObjectiveSurrogate`
- {py:class}`saealib.SklearnGPRSurrogate`
- {py:class}`saealib.SklearnRFRSurrogate`
- {py:class}`saealib.SklearnSVMSurrogate`
- {py:class}`saealib.SklearnNNSurrogate`
- {py:class}`saealib.SklearnXGBSurrogate`
- {py:class}`saealib.SklearnLGBMSurrogate`
- {py:class}`saealib.TorchSurrogate`
- {py:class}`saealib.SklearnClassificationSurrogate`
- {py:class}`saealib.SklearnRFCClassificationSurrogate`
- {py:class}`saealib.SklearnSVCClassificationSurrogate`
