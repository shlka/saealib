# Installation

## 動作要件

- Python 3.10 以降
- [NumPy](https://numpy.org/) >= 1.22.1
- [SciPy](https://scipy.org/) >= 1.0.0

## PyPI からのインストール

```bash
pip install saealib
```

## バージョンを指定してインストール

```bash
pip install "saealib=={{version}}"
```

## 仮想環境へのインストール（推奨）

依存関係の競合を避けるため，仮想環境へのインストールを推奨します．

::::{tab-set}
:::{tab-item} venv
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install saealib
```
:::
:::{tab-item} uv
```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install saealib
```
:::
::::

## インストールの確認

```python
import saealib
print(saealib.__version__)
```
