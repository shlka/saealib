# 混合変数最適化

連続変数だけでなく、整数変数やカテゴリカル変数を含む問題を、`saealib`で解きます。

アルゴリズム、サロゲート、評価戦略の切り替え方は[単目的最適化](single_objective.md)の「コンポーネントの切り替え」と共通です。

## 問題設定

設計変数ごとに、連続値、整数値、カテゴリの3種類が混在する目的関数を想定します。

```python
def func(x):
    # x[0]: continuous, x[1]: integer, x[2]: categorical index
    return x[0] ** 2 + (x[1] - 3) ** 2 + (0.0 if x[2] == 1 else 5.0)
```

`x[2]`は3つのカテゴリ(`0`, `1`, `2`)のいずれかを表す値で、`1`を選んだときだけペナルティが付きません。

## 変数の定義

各次元の型は`Variable`のサブクラスで指定し、`Problem`の`variables`引数に渡します。

| クラス | 意味 |
|--------|------|
| `ContinuousVariable(lb, ub)` | 連続値。`lb`/`ub`を省略した通常の`Problem`はこの型として扱われる |
| `IntegerVariable(lb, ub)` | 整数値 |
| `CategoricalVariable(categories)` | カテゴリのリストから1つを選ぶ |

```python
import numpy as np
from saealib import Problem, ContinuousVariable, IntegerVariable, CategoricalVariable

variables = [
    ContinuousVariable(-5.0, 5.0),
    IntegerVariable(0, 10),
    CategoricalVariable([0, 1, 2]),
]

problem = Problem(
    func=func,
    dim=3,
    n_obj=1,
    direction=np.array([-1.0]),
    variables=variables,
)
```

`variables`を渡すと、各要素の`lb`/`ub`から`Problem.lb`/`Problem.ub`が自動的に導出されるため、`lb`/`ub`引数は指定しません。

## 高レベルAPI: minimize

```python
from saealib import minimize

result = minimize(problem, max_fe=500, seed=0)
print(result.x, result.f)
```

`GA`は、変数の型ごとに専用の交叉と突然変異の演算子（整数用の`CrossoverIntegerSBX`/`MutationIntegerUniform`、カテゴリカル用の`CrossoverCategorical`/`MutationCategorical`）をデフォルトで備えており、`variables`を渡すだけで型に応じた探索が行われます。

`PSO`は速度に基づく更新方式のため、整数変数とカテゴリカル変数を正しく扱えません。

混合変数問題では`algorithm='GA'`（デフォルト）のまま使ってください。

## オペレータのカスタマイズ

整数用とカテゴリカル用の演算子も、`GA`のキーワード引数で個別に指定できます。

```python
from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
    CrossoverIntegerSBX,
    CrossoverCategorical,
    MutationIntegerUniform,
    MutationCategorical,
)

ga = GA(
    crossover=CrossoverBLXAlpha(0.7, 0.4),
    mutation=MutationUniform(0.3),
    parent_selection=SequentialSelection(),
    survivor_selection=TruncationSelection(),
    integer_crossover=CrossoverIntegerSBX(0.7, eta=15.0),
    integer_mutation=MutationIntegerUniform(0.3),
    categorical_crossover=CrossoverCategorical(0.7),
    categorical_mutation=MutationCategorical(0.3),
)

result = minimize(problem, algorithm=ga, max_fe=500, seed=0)
```

各演算子の詳細は[Algorithm](../components/algorithm.md)を参照してください。

## 参照

- {py:class}`saealib.Problem`
- {py:class}`saealib.ContinuousVariable` / {py:class}`saealib.IntegerVariable` / {py:class}`saealib.CategoricalVariable`
- {py:class}`saealib.GA`
- {py:func}`saealib.minimize`
