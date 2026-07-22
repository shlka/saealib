# Mutation

`GA`(`saealib.GA`)は、交叉で生成した子個体に摂動を加える処理を、`Mutation`という差し替え可能な演算子に委ねている。
突然変異の方式を変えたいときは、`GA`本体ではなくこの`Mutation`だけを差し替えればよい。

## Mutationの役割

`Mutation`が実装を要求するメソッドは`mutate(p, mutate_range, rng=...)`の1つだけである。
`p`には変異対象の個体が`shape (dim,)`で渡され、変異後の個体を同じ`shape (dim,)`で返す。
`mutate_range`には設計変数の下限と上限が`(lb, ub)`のタプルで渡される。

個体レベルの変異確率は、クラス属性`prob`が保持する。
`Crossover`では交叉を実行するかどうかの判定を`GA`側が担っていたが、`Mutation`ではこの判定を`mutate()`の実装自身が行う。
組み込みクラスはいずれも`mutate()`の冒頭で`rng.random() >= self.prob`を確認し、満たさない場合は個体をそのまま複製して返す。
そのため`GA`は個体レベルの確率を意識せず、選択された全個体に対して無条件に`mutate()`を呼び出すだけでよい。
独自の`Mutation`を実装する際は、この`prob`判定を自分で書く必要がある点に注意する。

もう1つのクラス属性`prob_var`は、変数レベルの変異確率である。
`None`の場合、`mutate()`の呼び出し時に`min(0.5, 1/dim)`へ解決される。

## 組み込みMutation

| クラス | パラメータ | 特徴 |
|---|---|---|
| `MutationUniform` | `prob, *, prob_var=None` | 選ばれた次元を`[lb, ub]`の一様乱数で置き換える |
| `MutationGaussian` | `prob, *, sigma, prob_var=None` | 選ばれた次元に`N(0, sigma)`のガウス摂動を加える{cite}`rechenberg1973es` |
| `MutationPolynomial` | `prob, *, eta, prob_var=None` | 多項式変異{cite}`deb2001mooea`。`eta`が大きいほど摂動が小さくなる |
| `MutationIntegerUniform` | `prob, *, prob_var=None` | `[lb[i], ub[i]]`（両端含む）の一様整数乱数で置き換える。整数変数向け |
| `MutationCategorical` | `prob, *, prob_var=None` | `{0, ..., n_categories-1}`の一様整数乱数でカテゴリインデックスを置き換える。カテゴリ変数向け |

連続変数だけの問題であれば、この中から1つを選んで`GA(mutation=..., ...)`に渡せばよい。
探索の荒さを変数の値域そのものに合わせたいなら`MutationUniform`、既存の値の近傍だけを摂動したいなら`MutationGaussian`か`MutationPolynomial`を選ぶ、という判断が基本になる。
`MutationGaussian`は`sigma`で摂動の大きさを直接指定するのに対し、`MutationPolynomial`は`eta`（分布指数）で摂動の集中度を指定し、値域からの相対的な摂動幅が自動的に決まる。
`MutationIntegerUniform`と`MutationCategorical`は、乱数の生成方法（値域内の一様整数乱数への置き換え）を共通の非公開実装で共有しているだけで、公開APIとしては別クラスとして扱う。

設計変数に整数変数やカテゴリ変数が混在する問題では、`GA`は変数の型ごとに異なる`Mutation`インスタンスを使い分ける。
`GA`コンストラクタの`integer_mutation`/`categorical_mutation`引数を省略すると、それぞれ`MutationIntegerUniform`/`MutationCategorical`が自動的に補われる(`prob_var`は連続変数用の`mutation`から引き継がれる)。
`GA.ask()`は個体を変数の型ごとの列に分割し、各`Mutation`を該当する列だけに適用してから結果を組み立て直す。
変数の型と`Mutation`の対応づけは[Problem](problem.md)の`variables`引数で決まる。

```{note}
`MutationUniform`のみ`@register()`済みで、他の4クラスは現状Registry未登録である。
Registry経由でクラスを文字列から解決する使い方をする場合はこの違いに注意する。
```

## 拡張フック

境界外に出た値を丸めるといった後処理だけを足したい場合、サブクラスを新設せずに`with_post(fn)`で既存の`Mutation`インスタンスへ処理を追加できる。
`with_post`は元のインスタンスを変更せず、`fn`を追加したコピーを返す。

```python
import numpy as np
from saealib import MutationUniform


def clip_offspring(offspring, mutate_range, rng, ctx=None):
    lb, ub = mutate_range
    return np.clip(offspring, lb, ub)


base = MutationUniform(prob=1.0)
clipped = base.with_post(clip_offspring)
```

`fn`のシグネチャは`fn(offspring, mutate_range, rng, ctx) -> np.ndarray`で、既存のフック（既定では何もしない恒等関数）の結果を受け取って追加の変換を返す。
複数回`with_post`を呼べば、フックは呼び出した順に連結される。

## 独自Mutationの実装方法

独自の突然変異方式が必要な場合は、`Mutation`を継承して`mutate()`だけを実装すればよい。
次の例は、選ばれた次元を値域の中点へ置き換える単純な変異である。

```python
import numpy as np
from saealib import Mutation


class MidpointMutation(Mutation):
    def __init__(self, prob: float = 1.0, *, prob_var: float | None = None):
        super().__init__()
        self.prob = prob
        self.prob_var = prob_var

    def mutate(self, p, mutate_range, rng=np.random.default_rng()):
        if rng.random() >= self.prob:
            return p.copy()
        dim = len(p)
        p_var = self.prob_var if self.prob_var is not None else min(0.5, 1.0 / dim)
        c = p.copy()
        lb, ub = mutate_range
        for i in range(dim):
            if rng.random() < p_var:
                c[i] = (lb[i] + ub[i]) / 2.0
        return c
```

`prob`による個体レベルの判定と`prob_var`による変数レベルの判定は、いずれも組み込みクラスと同じ流儀で自分で書く必要がある。
これらの判定を省略すると、常に全次元が変異する実装になる。

## 関連コンポーネント

- [Algorithm](algorithm.md) — `GA`が`Mutation`をどう組み合わせるか
- [Crossover](crossover.md) — 変異の前に呼ばれる、対になる演算子
- [Problem](problem.md) — 整数変数とカテゴリ変数の定義と、混合変数向けMutationの対応関係
- [拡張のガイドライン](extension_guidelines.md) — `with_post`系フックの一般的な設計思想

## 参照

- {py:class}`saealib.Mutation`
- {py:class}`saealib.MutationUniform`
- {py:class}`saealib.MutationGaussian`
- {py:class}`saealib.MutationPolynomial`
- {py:class}`saealib.MutationIntegerUniform`
- {py:class}`saealib.MutationCategorical`
