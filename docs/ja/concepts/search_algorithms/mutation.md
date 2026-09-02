---
primary_layer: layer3
page_type: concept
---

# Mutation

`GA`(`saealib.GA`)は、交叉で生成した子個体に摂動を加える処理を、`Mutation`という差し替え可能な演算子に委ねています。 突然変異の方式を変えたいときは、`GA`本体ではなくこの`Mutation`だけを差し替えればよいです。

## Mutationの役割

`Mutation`が実装する必要があるメソッドは`mutate_batch(candidates_batch, mutate_range, rng=...)`の1つだけです。形状`(n, dim)`の候補個体を受け取り、同じ形状の変異後個体を返します。`mutate_range`には設計変数の下限と上限が`(lb, ub)`タプルとして渡されます。基底クラスは`mutate_batch()`を1行で呼び出すことで、便利メソッド`mutate(p, mutate_range, rng=...)`を導出します。`GA`は`variation_execution="sequential"`のときこのメソッドを使います。`mutate()`だけを上書きしても既定のバッチモードには影響しません。`variation_execution`については[Algorithm](algorithm.md)と`GA`のAPIリファレンスを参照してください。

個体レベルの変異確率はクラス属性`prob`が保持します。`Crossover`では交叉を実行するかどうかを`GA`が判定しますが、各`Mutation`実装は変異の判定を自ら行います。すべての`mutate_batch()`実装は行ごとに1つのゲートを抽選し、通過しなかった行を変更せずに残す必要があります。これが、継承した1行用の`mutate()`に対する自己ゲートにもなります。そのため`GA`は候補を事前に絞らず、選択された変異メソッドを無条件に呼びます。独自の`Mutation`でも、この`prob`ゲートを実装する必要があります。

もう1つのクラス属性`prob_var`は変数レベルの変異確率です。`None`の場合、組み込みMutationは`mutate_batch()`の呼び出し時に`min(0.5, 1/dim)`へ解決します。

## 組み込みMutation

| クラス | パラメータ | 特徴 |
|---|---|---|
| `MutationUniform` | `prob, *, prob_var=None` | 選ばれた次元を`[lb, ub]`の一様乱数で置き換える |
| `MutationGaussian` | `prob, *, sigma, prob_var=None` | 選ばれた次元に`N(0, sigma)`のガウス摂動を加える{cite}`rechenberg1973es` |
| `MutationPolynomial` | `prob, *, eta, prob_var=None` | 多項式変異{cite}`deb2001mooea`。`eta`が大きいほど摂動が小さくなる |
| `MutationIntegerUniform` | `prob, *, prob_var=None` | `[lb[i], ub[i]]`（両端含む）の一様整数乱数で置き換える。整数変数向け |
| `MutationCategorical` | `prob, *, prob_var=None` | `{0, ..., n_categories-1}`の一様整数乱数でカテゴリインデックスを置き換える。カテゴリ変数向け |

連続変数だけの問題であれば、この中から1つを選んで`GA(mutation=..., ...)`に渡せばよいです。 探索の荒さを変数の値域そのものに合わせたいなら`MutationUniform`、既存の値の近傍だけを摂動したいなら`MutationGaussian`か`MutationPolynomial`を選ぶ、という判断が基本になります。 `MutationGaussian`は`sigma`で摂動の大きさを直接指定するのに対し、`MutationPolynomial`は`eta`（分布指数）で摂動の集中度を指定し、値域からの相対的な摂動幅が自動的に決まります。 `MutationIntegerUniform`と`MutationCategorical`は、乱数の生成方法（値域内の一様整数乱数への置き換え）を共通の非公開実装で共有しているだけで、公開APIとしては別クラスとして扱います。

設計変数に整数変数とカテゴリ変数が混在する問題では、`GA` は変数型ごとに異なる `Mutation` インスタンスを使います。`GA` コンストラクタの `integer_mutation`/`categorical_mutation` 引数を省略すると、`MutationIntegerUniform`/`MutationCategorical` が自動的に指定されます（`prob_var` は連続変数用 `mutation` から継承します）。`GA.ask()` は変数型ごとに個体を列へ分け、対応する列だけに各 `Mutation` を適用して結果を再構成します。既定のバッチモードは各型の列で `mutate_batch()` を呼び、逐次モードは各型の `mutate()` を呼んで、個体ごとの従来の振り分けと乱数列を保持します。変数型と `Mutation` の対応は [Problem](../problem_and_ranking/problem.md) の `variables` 引数で決まります。

```{note}
`MutationUniform`のみ`@register()`済みで、他の4クラスは現状Registry未登録です。Registry経由でクラスを文字列から解決する場合はこの違いに注意してください。
```

### 外部ライブラリアダプタ

`PymooMutation(operator, *, prob=1.0)`は、構築済みの[pymoo](https://pymoo.org/)突然変異演算子（例：`PM()`）をラップし、既存のpymooベースの研究コードを`GA`内でそのまま再利用できるようにします。

`PymooCrossover`と異なり、`prob`は`PymooMutation`自身が適用します。これは、すべての組み込み`Mutation`で個体レベルの確率判定を`mutate_batch()`内で行う規約に沿ったものです。`prob_var`はラップしたpymoo演算子から意図的に引き継がず、ここでは`None`のままです。そのため、`GA`の混合変数ルーティングは外部の`float`とは限らない値ではなく、独自の既定値にフォールバックします。`GA`の既定バッチモードでは、`PymooMutation.mutate_batch()`はゲート済み部分集合に対してラップした演算子の`_do()`を最大1回呼びます。`variation_execution="sequential"`では、継承した`mutate()`が1個体をそのバッチ実装に渡すため、直接`mutate()`を呼ぶ場合と同様に、ゲートを通過した個体ごとに`_do()`が1回呼ばれます。`rng`はpymooの`random_state`引数を介して渡されます。

`pymoo` extraのインストールについては[インストール](../../getting_started/installation.md)を参照してください。

## 拡張フック

境界外に出た値を丸めるといった後処理だけを足したい場合、サブクラスを新設せずに`with_post(fn)`で既存の`Mutation`インスタンスへ処理を追加できます。 `with_post`は元のインスタンスを変更せず、`fn`を追加したコピーを返します。

```python
import numpy as np
from saealib import MutationUniform


def clip_offspring(offspring, mutate_range, rng, ctx=None):
    lb, ub = mutate_range
    return np.clip(offspring, lb, ub)


base = MutationUniform(prob=1.0)
clipped = base.with_post(clip_offspring)
```

`fn`のシグネチャは`fn(offspring, mutate_range, rng, ctx) -> np.ndarray`で、既存のフック（既定では何もしない恒等関数）の結果を受け取って追加の変換を返します。 複数回`with_post`を呼べば、フックは呼び出した順に連結されます。

バッチモードでは、`GA`が`post_mutation_batch(offspring_batch, mutate_range, rng, ctx)`を呼びます。既定の実装は個体ごとに`post_mutation()`を呼ぶため、`with_post()`で追加したフックも引き続き動作します。後処理自体をベクトル化する場合は`post_mutation_batch()`を上書きします。この上書きでは、維持したい`with_post()`フックも自ら合成する必要があります。

## 独自Mutationの実装方法

独自の変異方式が必要な場合は、`Mutation`を継承して`mutate_batch()`を実装します。次の例は、選ばれた次元を値域の中点へ置き換えます。

```python
import numpy as np
from saealib import Mutation


class MidpointMutation(Mutation):
    def __init__(self, prob: float = 1.0, *, prob_var: float | None = None):
        super().__init__()
        self.prob = prob
        self.prob_var = prob_var

    def mutate_batch(self, candidates_batch, mutate_range, rng=np.random.default_rng()):
        candidates_batch = np.asarray(candidates_batch, dtype=float)
        n, dim = candidates_batch.shape
        p_var = self.prob_var if self.prob_var is not None else min(0.5, 1.0 / dim)
        gate = rng.random(n) < self.prob
        result = candidates_batch.copy()
        if not np.any(gate):
            return result

        selected = result[gate]
        var_gate = rng.random(selected.shape) < p_var
        lb, ub = mutate_range
        midpoint = (np.asarray(lb) + np.asarray(ub)) / 2.0
        selected[var_gate] = np.broadcast_to(midpoint, selected.shape)[var_gate]
        result[gate] = selected
        return result
```

個体レベルのゲートでは行ごとに1つの値を抽選し、通過しなかったすべての行を維持する必要があります。変数レベルの`prob_var`ゲートは演算子ごとに異なりますが、ここでは組み込みクラスと同じ規約に従います。これらの判定を省略すると、常に全次元が変異する実装になります。

## 関連コンポーネント

- [Algorithm](algorithm.md)：`GA`が`Mutation`をどう組み合わせるか
- [Crossover](crossover.md)：変異の前に呼ばれる、対になる演算子
- [Problem](../problem_and_ranking/problem.md)：整数変数とカテゴリ変数、および混合変数Mutationとの対応の定義
- [拡張ガイドライン](../extension_guidelines.md)：`with_post` 型フックの一般的な設計思想

## 参照

- {py:class}`saealib.Mutation`
- {py:class}`saealib.MutationUniform`
- {py:class}`saealib.MutationGaussian`
- {py:class}`saealib.MutationPolynomial`
- {py:class}`saealib.MutationIntegerUniform`
- {py:class}`saealib.MutationCategorical`
- {py:class}`saealib.PymooMutation`
