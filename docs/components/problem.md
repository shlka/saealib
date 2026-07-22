# Problem

`saealib`の全コンポーネントは、`Problem`を通じて解くべき最適化問題を共有します。
目的関数そのものだけでなく、設計変数の範囲、目的の方向、制約、解の比較方法までを1つのオブジェクトにまとめる役割を持ちます。

## Problemの役割

`Problem`のコンストラクタは次の引数を受け取ります。

```python
Problem(
    func, dim, n_obj, direction, lb=None, ub=None,
    comparator=None, constraints=None, *,
    eps_cv=1e-6, eps_obj=1e-6, handler=None, variables=None,
)
```

**`func`**：設計変数の配列を受け取り評価値を返す目的関数。
**`dim`**：設計変数の次元数。
**`n_obj`**：目的関数の数。

**`direction`**：目的ごとの最適化方向を表す`shape (n_obj,)`の配列で、各要素は`+1`（最大化）または`-1`（最小化）でなければなりません。
それ以外の値を渡すと構築時に例外になります。

**`lb`**/**`ub`**：設計変数の下限と上限。
`variables`を指定しない場合は必須で、指定した場合は`variables`側の範囲から自動的に導出されます。

**`comparator`**：解を比較する[Comparator](comparators.md)。
省略した場合、`n_obj == 1`なら`SingleObjectiveComparator`、`n_obj > 1`なら`NSGA2Comparator`が自動的に選ばれます。
渡した`Comparator`の`direction`が未設定（`None`）であれば、`Problem`の`direction`がそのまま注入されます。

**`constraints`**：不等式制約(`InequalityConstraint`)のリスト。
定義方法は[制約付き最適化](../tutorials/constraints.md)で扱います。

**`eps_cv`**/**`eps_obj`**：それぞれ実行可能性判定の許容誤差、目的関数値の同値判定の許容誤差を表します。
`eps_cv`はコンストラクタ実行時に既定の`handler`/`comparator`へ引き継がれるだけで、構築後に`problem.eps_cv`を直接書き換えても実行時の挙動には反映されません。
実際に使われる閾値は`handler.feasibility_threshold`であり、`Optimizer`の実行中は毎世代`comparator`/`pareto_archive`へ同期されます。

**`handler`**：制約違反の集約や目的関数の補正を担う[ConstraintHandler](constraints.md)。
省略時は`StaticToleranceHandler(eps_cv=eps_cv)`が使われます。

**`variables`**：設計変数ごとの型を`Variable`のリストで指定します。
連続変数だけの問題では省略してよく、その場合は全次元が`ContinuousVariable`として扱われます。
整数変数とカテゴリ変数を混在させたい場合にここへ`IntegerVariable`/`CategoricalVariable`を含めると、[Crossover](crossover.md)/[Mutation](mutation.md)が変数の型ごとに異なる演算子を自動的に割り当てます。

```{note}
旧バージョンのチュートリアルには`weight=`という引数を使う例があるが、現行の`Problem`にこの引数は存在しません。
`weight=`を渡すと`TypeError`になります。
```

## directionとweightの役割分担

`direction`は`saealib`全体で統一された、符号だけを表す`±1`の配列です。
一方、`WeightedSumComparator`や`DecompositionComparator`が受け取る`weights`は、複数目的をスカラー値へ集約する際の非負の重みであり、方向とは独立した別の概念です。

この役割分担のもとでは、目的の重要度そのもの（スケーリング）を`weights`で表現することはできません。
目的関数値の大きさを調整したい場合は`func`の内部でスケーリングします。
`direction`は符号だけを、`weights`は集約の重み配分だけを表す、という2軸に整理されています。

## 独自Variableの実装方法

`Variable`(ABC)は、`lb`/`ub`という2つのプロパティと`repair(x)`というメソッドだけを要求します。
組み込みの`ContinuousVariable`/`IntegerVariable`/`CategoricalVariable`は、いずれも自分の定義域へ値を射影するだけの薄い実装であり、これ以外の変数型（周期変数、対数スケール変数など）が必要な場合は`Variable`を直接継承すればよいです。

次の例は、値を切り詰めるのではなく境界で折り返す変数です。
角度のように、上限を超えた値が下限側から連続しているとみなしたい設計変数に使えます。

```python
import numpy as np
from saealib import Variable


class PeriodicVariable(Variable):
    def __init__(self, lb: float, ub: float):
        self._lb = float(lb)
        self._ub = float(ub)

    @property
    def lb(self) -> float:
        return self._lb

    @property
    def ub(self) -> float:
        return self._ub

    def repair(self, x):
        span = self._ub - self._lb
        return self._lb + np.mod(np.asarray(x, dtype=float) - self._lb, span)
```

`ContinuousVariable.repair()`が`np.clip`で範囲外の値を境界に留めるのに対し、この実装は`np.mod`で範囲外の値を反対側の境界から巻き戻します。
`Variable`が表す値は、`Population`配列上で扱われる「エンコード済みfloat空間」上の値である点に注意してください。
`CategoricalVariable`のようにカテゴリ値と内部インデックスの対応が必要な変数型では、`repair()`はインデックス空間上で完結させ、実際のカテゴリ値への変換は別のメソッドで行います。

## 関連コンポーネント

- [Comparator](comparators.md) — `comparator`引数で解の比較方法を差し替える
- [ConstraintHandler](constraints.md) — `handler`引数で制約違反の扱い方を差し替える
- [Crossover](crossover.md) / [Mutation](mutation.md) — `variables`で定義した変数の型ごとに使い分けられる演算子
- [制約付き最適化](../tutorials/constraints.md) — 制約の定義方法と組み込み`ConstraintHandler`の選び方

## 参照

- {py:class}`saealib.Problem`
- {py:class}`saealib.Variable`
- {py:class}`saealib.ContinuousVariable`
- {py:class}`saealib.IntegerVariable`
- {py:class}`saealib.CategoricalVariable`
