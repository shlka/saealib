# Crossover

`GA`(`saealib.GA`)は、選択した親個体から子個体を生成する処理を、`Crossover`という差し替え可能な演算子に委ねている。
交叉の方式を変えたいときは、`GA`本体ではなくこの`Crossover`だけを差し替えればよい。

## Crossoverの役割

`Crossover`が実装を要求するメソッドは`crossover(parent, bounds=None, rng=...)`の1つだけである。
`parent`には交叉に使う親個体の配列が渡される。
既定では`n_parents = 2`個の親をshape `(n_parents, dim)`で受け取り、`n_children = 2`個の子をshape `(n_children, dim)`で返す。
2親2子以外の交叉方式を実装する場合は、`n_parents`/`n_children`というクラス属性をサブクラス側で上書きする。
`bounds`には設計変数の下限と上限が`(lb, ub)`のタプルで渡され、`None`の場合は無制限として扱う。

交叉を実行するかどうかを決める個体レベルの確率は、`crossover()`の内部ではなく呼び出し側が判定する。
`GA.ask()`は`ctx.rng.random() < self.crossover.prob`を満たしたペアだけに対して`crossover()`を呼び、満たさないペアは親をそのまま子として複製する。
つまり`prob`というクラス属性自体は`Crossover`が保持しているが、それを読んで交叉を実行するかどうかを決めるのは`GA`側の責務であり、`crossover()`の実装は常に交叉が起こる前提で書いてよい。

## 組み込みCrossover

| クラス | パラメータ | 特徴 |
|---|---|---|
| `CrossoverBLXAlpha` | `prob, alpha` | BLX-α交叉(Eshelman & Schaffer, 1993が導入)。`alpha`が大きいほど、子が親の値の範囲外まで広がりうる |
| `CrossoverSBX` | `prob, eta, *, prob_var=0.5` | Simulated Binary Crossover{cite}`deb1995sbx`。`bounds`が有限値なら境界付き版に自動で切り替わる。`eta`が大きいほど子は親に近づく |
| `CrossoverUniform` | `prob, swap_rate=0.5` | 各次元を独立に`swap_rate`の確率で親同士を入れ替える(Syswerda, 1989が導入) |
| `CrossoverOnePoint` | `prob` | 1点交叉 |
| `CrossoverTwoPoint` | `prob` | 2点交叉 |
| `CrossoverIntegerSBX` | `prob, eta, *, prob_var=0.5` | `CrossoverSBX`{cite}`deb1995sbx`と同じ計算をしたのち整数に丸める。整数変数向け |
| `CrossoverCategorical` | `prob` | 各次元を50/50でどちらかの親の値をそのままコピーする。カテゴリ変数向け |

連続変数だけの問題であれば、この中から1つを選んで`GA(crossover=..., ...)`に渡せばよい。
`CrossoverBLXAlpha`/`CrossoverUniform`は無制約な問題で素直に使え、境界を活かした交叉が必要なら`CrossoverSBX`を選ぶ、という判断が基本になる。

設計変数に整数変数やカテゴリ変数が混在する問題では、`GA`は変数の型ごとに異なる`Crossover`インスタンスを使い分ける。
`GA`コンストラクタの`integer_crossover`/`categorical_crossover`引数を省略すると、それぞれ`CrossoverIntegerSBX`/`CrossoverCategorical`が自動的に補われる(`eta`や`prob`は連続変数用の`crossover`から引き継がれる)。
`GA.ask()`は親個体を変数の型ごとの列に分割し、各`Crossover`を該当する列だけに適用してから結果を組み立て直す。
この仕組み上、`integer_crossover`/`categorical_crossover`に独自のクラスを渡す場合は、`n_children`/`n_parents`を連続変数用の`crossover`と一致させる必要がある。
一致しない場合は`ConfigurationError`になる。

変数の型と`Crossover`の対応づけは[Problem](problem.md)の`variables`引数で決まる。

```{note}
`CrossoverBLXAlpha`のみ`@register()`済みで、他の6クラスは現状Registry未登録である。
Registry経由でクラスを文字列から解決する使い方をする場合はこの違いに注意する。
```

## 拡張フック

境界外に出た値を丸めるといった後処理だけを足したい場合、サブクラスを新設せずに`with_post(fn)`で既存の`Crossover`インスタンスへ処理を追加できる。
`with_post`は元のインスタンスを変更せず、`fn`を追加したコピーを返す。

```python
import numpy as np
from saealib import CrossoverBLXAlpha

base = CrossoverBLXAlpha(prob=1.0, alpha=0.5)


def clip_to_bounds(offspring, parents, rng, ctx=None):
    return np.clip(offspring, -1.0, 1.0)


repaired = base.with_post(clip_to_bounds)
```

`fn`のシグネチャは`fn(offspring, parents, rng, ctx) -> np.ndarray`で、既存のフック（既定では何もしない恒等関数）の結果を受け取って追加の変換を返す。
複数回`with_post`を呼べば、フックは呼び出した順に連結される。

## 独自Crossoverの実装方法

独自の交叉方式が必要な場合は、`Crossover`を継承して`crossover()`だけを実装すればよい。
次の例は、2つの親の平均を2個の子としてそのまま返す単純な交叉である。

```python
import numpy as np
from saealib import Crossover


class AverageCrossover(Crossover):
    def __init__(self, prob: float = 1.0):
        super().__init__()
        self.prob = prob

    def crossover(self, parent, bounds=None, rng=np.random.default_rng()):
        mean = parent.mean(axis=0)
        return np.array([mean, mean])
```

`n_parents`/`n_children`を2以外にしたい場合や、`bounds`を使う独自の丸め処理を加えたい場合は、クラス属性のオーバーライドと`crossover()`内での`bounds`の参照を追加すればよい。

## 関連コンポーネント

- [Algorithm](algorithm.md) — `GA`が`Crossover`をどう組み合わせるか
- [Mutation](mutation.md) — 交叉の次に呼ばれる、対になる演算子
- [ParentSelection](parent_selection.md) — `Crossover`に渡す親個体を選ぶ演算子
- [Problem](problem.md) — 整数変数とカテゴリ変数の定義と、混合変数向けCrossoverの対応関係
- [拡張のガイドライン](extension_guidelines.md) — `with_post`系フックの一般的な設計思想

## 参照

- {py:class}`saealib.Crossover`
- {py:class}`saealib.CrossoverBLXAlpha`
- {py:class}`saealib.CrossoverSBX`
- {py:class}`saealib.CrossoverUniform`
- {py:class}`saealib.CrossoverOnePoint`
- {py:class}`saealib.CrossoverTwoPoint`
- {py:class}`saealib.CrossoverIntegerSBX`
- {py:class}`saealib.CrossoverCategorical`
