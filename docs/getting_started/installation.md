# インストール

## 動作要件
- Python >= 3.10

依存パッケージの正確なバージョンは[pyproject.toml](https://github.com/shlka/saealib/blob/main/pyproject.toml)を参照してください。

## インストール方法

pipによるインストールは次のように行います。
このインストールでは最小の依存関係のみインストールします。
```bash
pip install saealib
```
すべての依存関係をフルインストールする場合は次のように行います。
```bash
pip install "saealib[all]"
```
オプションの指定については[こちら](#install-options)を参照してください。

## バージョンの指定
バージョンの指定は次のように行います。
```bash
pip install "saealib==X.Y.Z"
# example:
pip install "saealib=0.1.0"
```
最新のプレリリースバージョンは次のように指定します。
```bash
pip install --pre saealib
```
:::{warning}
プレリリースバージョンのAPIは予告なく変更・削除される可能性があります。
:::

(install-options)=
## オプションの指定
一部のパッケージは依存関係を追加することで有効になります。
依存関係のインストールは次のようにオプションを指定することで同時にインストールできます。
```bash
pip install "saealib[opt1,opt2,...]"
# example:
pip install "saealib[sklearn,parallel]"
```
すべての依存関係をインストールする場合は次のように指定します。
```bash
pip install "saealib[all]"
```
すべてのオプションは次の表に示す通りです。
| Extra | Adds |
|---|---|
| `sklearn` | scikit-learn-based surrogates |
| `xgboost` | XGBoost surrogate |
| `lightgbm` | LightGBM surrogate |
| `torch` | PyTorch-based components |
| `parallel` | joblib-based parallel evaluation |
| `all` | everything above |

## インストールの確認
インストールが完了したら、次のコマンドでバージョンを確認できます。
```bash
python -c "import saealib; print(saealib.__version__)"
```

:::{seealso}
ソースからの開発用インストールについては[CONTRIBUTING.md](https://github.com/shlka/saealib/blob/main/CONTRIBUTING.md)を参照してください。
:::
