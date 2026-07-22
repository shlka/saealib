# 外部ライブラリとの連携

`saealib`は、外部の機械学習ライブラリを、独自の抽象基底クラスの背後に薄くラップしたアダプタを提供します。

アダプタが翻訳するのは`Problem`/`Population`/`ctx`といった`saealib`側のデータ表現だけで、学習アルゴリズムそのものは外部ライブラリの実装をそのまま使います。

現時点では、サロゲートモデルのアダプタ(scikit-learn、XGBoost、LightGBM、PyTorch)が実装されています。

## インストール

各アダプタは、対応する`extra`を指定してインストールしたときだけ使えます。

```bash
pip install "saealib[sklearn]"
```

インストール方法とextra一覧の詳細は[インストール](../getting_started/installation.md)を参照してください。

対応する`extra`をインストールしていない状態でアダプタをインポートすると、`ImportError`になります。

## サロゲートアダプタ

各アダプタは`saealib`の`Surrogate`基底クラスを実装しており、組み込みの`RBFSurrogate`と同じように`surrogate`引数へ渡せます。

| クラス | 対応する`extra` | ラップするモデル |
|--------|--------|--------|
| `SklearnGPRSurrogate` | `sklearn` | Gaussian Process Regressor |
| `SklearnRFRSurrogate` | `sklearn` | Random Forest Regressor |
| `SklearnSVMSurrogate` | `sklearn` | Support Vector Regression |
| `SklearnNNSurrogate` | `sklearn` | Multi-layer Perceptron |
| `SklearnXGBSurrogate` | `xgboost` | XGBoost回帰 |
| `SklearnLGBMSurrogate` | `lightgbm` | LightGBM回帰 |
| `TorchSurrogate` | `torch` | 任意のPyTorch `nn.Module` |

コンストラクタへのキーワード引数は、そのまま対応するライブラリのモデルへ渡されます。

```python
import numpy as np
from saealib import minimize, SklearnGPRSurrogate


def expensive_func(x):
    return np.sum(x**2)


DIM = 10

result = minimize(
    expensive_func,
    dim=DIM,
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
    surrogate=SklearnGPRSurrogate(),
    max_fe=300,
    seed=0,
)
```

`Surrogate`インスタンスを`surrogate`引数に渡す挙動は、[単目的最適化](single_objective.md)の「コンポーネントの切り替え」で示した`RBFSurrogate`の例と同じで、内部で`LocalSurrogateManager`にラップされます。

分類問題向けのアダプタ(実行可能性分類など)や、各アダプタの詳細な引数は[Surrogate](../components/surrogate.md)を参照してください。

<!--
## 今後の拡張

現時点でアダプタが存在するのはサロゲートモデルだけで、`Algorithm`や`Crossover`/`Mutation`など他のコンポーネントに対する外部ライブラリアダプタ(pymooのアルゴリズムを利用するものなど)は未実装です。

今後、サロゲート以外のアダプタが追加されたときは、本ページに節を追加していきます。
-->

## 参照

- {py:class}`saealib.Surrogate`
- {py:class}`saealib.SklearnGPRSurrogate` / {py:class}`saealib.SklearnRFRSurrogate` / {py:class}`saealib.SklearnSVMSurrogate` / {py:class}`saealib.SklearnNNSurrogate`
- {py:class}`saealib.SklearnXGBSurrogate` / {py:class}`saealib.SklearnLGBMSurrogate`
- {py:class}`saealib.TorchSurrogate`
- {py:func}`saealib.minimize`
