# 制約付き最適化

設計変数に制約がある問題を、`saealib`で解きます。

アルゴリズム、サロゲート、評価戦略の切り替え方は[単目的最適化](single_objective.md)の「コンポーネントの切り替え」と共通です。

## 問題設定

目的関数に加えて、解が満たすべき不等式制約`g(x) <= 0`を持つ問題を想定します。

```python
import numpy as np


def expensive_func(x):
    return np.sum(x**2)


def g1(x):
    # require the sum of the design variables to be at least 1
    return 1.0 - np.sum(x)


DIM = 5
LB = [-5.0] * DIM
UB = [5.0] * DIM
```

`g1(x) <= 0`を満たす解だけが実行可能解です。

## 制約の定義

制約は`InequalityConstraint(func, threshold=0.0)`で定義し、`Problem`の`constraints`引数に渡します。

```python
from saealib import InequalityConstraint, Problem, minimize

constraint = InequalityConstraint(g1, threshold=0.0)

problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=LB,
    ub=UB,
    constraints=[constraint],
)

result = minimize(problem, max_fe=1000, seed=0)
print(result.x, result.f)
print(constraint.violation(result.x))  # 0.0 means the constraint is satisfied
```

`g(x) >= threshold`の形で制約を課したい場合は、`func`の符号を反転させて渡します。

等式制約`h(x) = 0`は、`InequalityConstraint`を符号反転で組み合わせる代わりに`EqualityConstraint(func, tolerance=1e-6)`を使います。

```python
from saealib import EqualityConstraint


def h1(x):
    # require the sum of the design variables to be exactly 1
    return np.sum(x) - 1.0


equality = EqualityConstraint(h1, tolerance=1e-6)
```

`EqualityConstraint`は、`|h(x)| <= tolerance`を満たす解を実行可能解として扱います。

## 実行可能性の確認

`Problem`の`eps_cv`（既定値`1e-6`）以下の制約違反量`cv`を持つ解が実行可能解です。

```python
archive_cv = result.ctx.archive.get_array("cv")
feasible = archive_cv <= problem.eps_cv
print(f"feasible: {feasible.sum()} / {len(archive_cv)}")
```

## ConstraintHandlerによるカスタマイズ

複数の制約から`cv`を集約する方法と、その値をもとに実行可能性を判定する方法は、`ConstraintHandler`が決めます。

`Problem`の`handler`引数を省略すると、`StaticToleranceHandler(eps_cv=problem.eps_cv)`が使われ、`cv <= eps_cv`かどうかで実行可能性を判定します。

| クラス | 動作 |
|--------|------|
| `StaticToleranceHandler` | 固定の許容誤差`eps_cv`で実行可能性を判定する（既定） |
| `EpsilonConstraintHandler` | 許容誤差を世代とともに0へ近づけ、徐々に実行可能領域へ解を追い込む |
| `GradientRepairHandler` | 制約の勾配を使って実行不可能解を修復する |

`EpsilonConstraintHandler`には、世代番号を受け取り許容誤差を返す関数を渡します。

```python
from saealib import EpsilonConstraintHandler


def schedule(gen):
    return max(0.0, 1.0 - gen * 0.05)


problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=LB,
    ub=UB,
    constraints=[constraint],
    handler=EpsilonConstraintHandler(schedule),
)
result = minimize(problem, max_fe=1000, seed=0)
```

## 参照

- {py:class}`saealib.InequalityConstraint` / {py:class}`saealib.EqualityConstraint`
- {py:class}`saealib.Problem`
- {py:class}`saealib.ConstraintHandler` / {py:class}`saealib.StaticToleranceHandler` / {py:class}`saealib.EpsilonConstraintHandler` / {py:class}`saealib.GradientRepairHandler`
- {py:func}`saealib.minimize`
