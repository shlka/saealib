# Quickstart

## 最初の最適化を実行する

`saealib` では，目的関数・変数の次元・探索範囲を指定するだけで最適化を実行できます．

::::{tab-set}
:::{tab-item} minimize
```python
import numpy as np
from saealib import minimize

def sphere(x):
    return np.sum(x ** 2)

result = minimize(sphere, dim=5, lb=[-5] * 5, ub=[5] * 5, seed=0, verbose=False)

print(result.X)   # 最適解の設計変数
print(result.F)   # 最適解の目的関数値
print(result.fe)  # 真の関数評価回数
```
:::
:::{tab-item} maximize
```python
import numpy as np
from saealib import maximize

def neg_sphere(x):
    return -np.sum(x ** 2) + 10

result = maximize(neg_sphere, dim=5, lb=[-5] * 5, ub=[5] * 5, seed=0, verbose=False)

print(result.X)   # 最適解の設計変数
print(result.F)   # 最適解の目的関数値
print(result.fe)  # 真の関数評価回数
```
:::
::::

## 結果の読み方

| 属性 | 説明 |
|------|------|
| `result.X` | 最適解の設計変数．形状 `(dim,)` |
| `result.F` | 最適解の目的関数値．形状 `(n_obj,)` |
| `result.fe` | 最適化に使用した真の関数評価回数 |
| `result.gen` | 完了した世代数 |

## 次のステップ

- **チュートリアル** — アルゴリズムやサロゲートモデルのカスタマイズ方法を学ぶ
- **API リファレンス** — `minimize` / `maximize` の全パラメータを確認する
