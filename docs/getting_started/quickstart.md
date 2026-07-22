# クイックスタート
インストールは完了していますか？
インストール方法は[こちら](./installation.md)を参照してください。

## 初めての実行
まずは最小の記述で最適化問題を解いてみましょう。
```python
from saealib import minimize
from saealib.benchmarks import rastrigin

problem = rastrigin(n_var=10)
result = minimize(func=problem)
print(f"objective: {result.f}")
print(f"solution: {result.x}")
print(f"evaluated: {result.fe}")
print(f"generation: {result.gen}")
```
ここでは`saealib`が提供するベンチマークパッケージから、設計変数空間が10次元のRastrigin関数(`saealib.benchmarks.rastrigin`)の最小化問題を解いています。
`minimize()` / `maximize()`は、パラメータを指定するだけで最適化を実行できる高レベルAPIです。

## 任意の関数を最適化
前節ではベンチマーク問題を使用しましたが、ここでは任意の関数を最適化する例を見てみましょう。
`saealib`のベンチマークパッケージを利用する場合は必要なパラメータが自動的にAPIへ渡されますが、任意の関数(`callable`)を渡す場合はいくつかのパラメータを自分で指定する必要があります。
```python
import numpy as np
from saealib import minimize


def rastrigin(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


result = minimize(func=rastrigin, dim=10, lb=[-5.12] * 10, ub=[5.12] * 10)
print(f"objective: {result.f}")
print(f"solution: {result.x}")
```
ここでは10次元のRastrigin関数を定義して最小化問題を解いています。
`func`パラメータはnumpy配列を受け取り評価値を返すような任意の`callable`オブジェクトを指定できます。
シミュレーション(CAE)や機械学習モデルの学習など、評価コストの高い目的関数をここに指定することで、効率的なパラメータ探索に応用できます。

## 次のステップ
ここで紹介した機能は`saealib`の一部分です。
詳細なガイドは次のページを参照してください。

- [チュートリアル](../tutorials/index.md): 具体的なシチュエーション別の使い方ガイド
- [コンポーネント](../components/index.md): コンポーネントごとの詳しい使い方
- [API Reference](../api/index.md): 全パラメータのリファレンス
