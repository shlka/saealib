# ConstraintHandler

制約の基本的な定義方法（`InequalityConstraint`/`EqualityConstraint`）、`Problem`の`handler`引数の指定方法、組み込み3種のハンドラの選び方は[制約付き最適化](../tutorials/constraints.md)で扱っています。
このページでは、それらを踏まえた上で独自の`ConstraintHandler`を書きたい場合に絞って説明します。

## ConstraintHandlerの役割

`ConstraintHandler`は、制約の処理方式を差し替え可能なライフサイクルフックの集合として公開しています。

```
Ask            -> [repair(x, constraints, lb, ub)]
               -> evaluate f, g
               -> [compute_cv(constraints, x, g)]        -> cv
               -> [augment_objective(f, constraints, x, g)] -> f'
Tell           -> Comparator(f', cv) with eps_cv = feasibility_threshold
Generation end -> [on_generation_end(gen, population)]
```

抽象メソッドは`compute_cv`だけであり、残りのフックはいずれも既定実装を持ちます。
独自の`ConstraintHandler`は、必要なフックだけをオーバーライドすればよいです。

**`repair(x, constraints, lb, ub, **kwargs)`**：交叉と突然変異の後、評価の前に呼ばれます。
既定は`np.clip(x, lb, ub)`で、境界にクリップするだけです。

**`compute_cv(constraints, x, g)`**（抽象）：制約群を単一の`cv`値へ集約します。

**`augment_objective(f, constraints, x, g)`**：目的関数値を制約情報で変換します。
既定は恒等関数（変換しない）で、ペナルティ関数法やaugmented Lagrangian法を実装する際にオーバーライドする場所です。
組み込み3種のハンドラはいずれもこのフックをオーバーライドしていないため、現状は将来の拡張点として空いています。

**`feasibility_threshold`**（プロパティ）：既定`1e-6`。
この値が`Comparator`の`eps_cv`として使われ、`Optimizer`実行中は毎世代同期されます。

**`on_generation_end(gen, population)`**：世代の終わりに呼ばれます。
既定はno-opで、`EpsilonConstraintHandler`のようにε値を世代ごとに更新する内部状態を持つハンドラが使います。

## 組み込みハンドラがオーバーライドするフック

| クラス | オーバーライドするフック |
|---|---|
| `StaticToleranceHandler` | `compute_cv` / `feasibility_threshold` |
| `EpsilonConstraintHandler` | `compute_cv` / `feasibility_threshold` / `on_generation_end` |
| `GradientRepairHandler` | `repair` / `compute_cv` |

`EpsilonConstraintHandler`{cite}`mezuramontes2011epsilon`は、`schedule(gen) -> float`という関数でεを世代ごとに更新します。
`linear_epsilon_schedule(eps0, n_gen)`/`exponential_epsilon_schedule(eps0, decay)`という組み込みのスケジュール生成関数も公開されています。

`GradientRepairHandler`{cite}`chootinan2006gradientrepair`は、`EqualityConstraint.gradient()`を使ったNewton風の1ステップで`repair()`をオーバーライドします。
`gradient()`が`None`を返す制約（勾配が未提供の制約）は、repair時にスキップされます。

## InequalityConstraint/EqualityConstraint側の拡張点

`InequalityConstraint`自体にも2つの拡張点があります。

**`gradient(x)`**：既定は`None`を返します。
オーバーライドして解析的な勾配ベクトルを返すようにすると、`GradientRepairHandler`が使えるようになります。

**`violation_from_value(g)`**：生の制約値`g(x)`から違反量への変換を定義します。
既定は`max(0, g - threshold)`。
`EqualityConstraint`はこのメソッドだけをオーバーライドし、`max(0, |h(x)| - tolerance)`という独自の変換を定義しています。

## 独自ConstraintHandlerの実装方法

独自の制約処理戦略が必要な場合は、`ConstraintHandler`を継承して`compute_cv()`だけを実装すればよいです。
次の例は、制約違反量を目的関数値へ加算するペナルティ関数法です。

```python
from saealib import ConstraintHandler


class PenaltyHandler(ConstraintHandler):
    """違反量をペナルティとして目的関数値に加算する。"""

    def __init__(self, penalty_coeff: float = 1e3):
        self.penalty_coeff = penalty_coeff

    def compute_cv(self, constraints, x, g):
        return sum(max(0.0, gi) for gi in g)

    def augment_objective(self, f, constraints, x, g):
        cv = self.compute_cv(constraints, x, g)
        return f + self.penalty_coeff * cv
```

`augment_objective`をオーバーライドしてペナルティを目的関数値に加算する一方で、`compute_cv`は制約違反として`0`を返す実装にすれば、ペナルティ関数法だけで実行可能性を扱わない（全解を実行可能とみなす）構成にもできます。

## 関連コンポーネント

- [制約付き最適化](../tutorials/constraints.md) — 制約の定義と組み込みハンドラの基本的な使い方
- [Problem](problem.md) — `handler`引数の渡し方
- [Comparator](comparators.md) — `feasibility_threshold`が`eps_cv`として同期される先

## 参照

- {py:class}`saealib.ConstraintHandler`
- {py:class}`saealib.StaticToleranceHandler`
- {py:class}`saealib.EpsilonConstraintHandler`
- {py:class}`saealib.GradientRepairHandler`
- {py:class}`saealib.InequalityConstraint`
- {py:class}`saealib.EqualityConstraint`
- {py:func}`saealib.linear_epsilon_schedule`
- {py:func}`saealib.exponential_epsilon_schedule`
