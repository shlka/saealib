---
primary_layer: layer3
page_type: concept
---

# Crossover

`GA`(`saealib.GA`)は、選択した親個体から子個体を生成する処理を、`Crossover`という差し替え可能な演算子に委ねています。 交叉の方式を変えたいときは、`GA`本体ではなくこの`Crossover`だけを差し替えればよいです。

## Crossoverの役割

`Crossover`が実装する必要があるメソッドは`crossover_batch(parents_batch, bounds=None, rng=...)`の1つだけです。形状`(n_pair, n_parents, dim)`の`n_pair`個の親グループを受け取り、形状`(n_pair, n_children, dim)`の子を返します。既定では各グループが`n_parents = 2`個の親を含み、`n_children = 2`個の子を生成します。2親2子以外の交叉方式を実装するには、サブクラスで`n_parents`/`n_children`クラス属性を上書きします。`bounds`には設計変数の下限と上限が`(lb, ub)`タプルとして渡され、`None`の場合は上限・下限なしです。

基底クラスは、`crossover_batch()`を1つの親グループで呼び出すことで、簡便メソッド`crossover(parent, bounds=None, rng=...)`を提供します。`GA`は`variation_execution="sequential"`でこのメソッドを使います。`crossover()`だけを上書きしても既定のバッチモードには影響しません。`variation_execution`については[Algorithm](algorithm.md)と`GA`のAPIリファレンスを参照してください。

交叉を実行するかどうかを決める個体レベルの確率は、どちらの交叉メソッド内でもなく`GA`が判定します。既定のバッチモードでは`GA.ask()`がゲートを抽選し、通過した親グループだけを`crossover_batch()`へ渡します。逐次モードでは、通過したグループごとに`crossover()`を呼びます。ゲートを通過しなかったグループは親をそのまま子として複製するため、どちらのメソッドも交叉が実行される前提で実装できます。

## 組み込みCrossover

| クラス | パラメータ | 特徴 |
|---|---|---|
| `CrossoverBLXAlpha` | `prob, alpha` | BLX-α交叉（Eshelman & Schaffer, 1993が導入）。`alpha`が大きいほど、子が親の値の範囲外まで広がりうる |
| `CrossoverSBX` | `prob, eta, *, prob_var=0.5` | シミュレーテッドバイナリ交叉{cite}`deb1995sbx`。`bounds`が有限値なら境界付き版に自動で切り替わる。`eta`が大きいほど子個体は親に近づく |
| `CrossoverUniform` | `prob, swap_rate=0.5` | 各次元を独立に`swap_rate`の確率で親同士を入れ替える（Syswerda, 1989が導入） |
| `CrossoverOnePoint` | `prob` | 1点交叉 |
| `CrossoverTwoPoint` | `prob` | 2点交叉 |
| `CrossoverIntegerSBX` | `prob, eta, *, prob_var=0.5` | `CrossoverSBX`{cite}`deb1995sbx`と同じ計算をしたのち整数に丸める。整数変数向け |
| `CrossoverCategorical` | `prob` | 各次元を50/50でどちらかの親の値をそのままコピーする。カテゴリ変数向け |

連続変数だけの問題であれば、この中から1つを選んで`GA(crossover=..., ...)`に渡せばよいです。 `CrossoverBLXAlpha`/`CrossoverUniform`は無制約な問題で素直に使え、境界を活かした交叉が必要なら`CrossoverSBX`を選ぶ、という判断が基本になります。

設計変数に整数変数やカテゴリ変数が混在する問題では、`GA`は変数の型ごとに異なる`Crossover`インスタンスを使い分けます。`GA`コンストラクタの`integer_crossover`/`categorical_crossover`引数を省略すると、`CrossoverIntegerSBX`/`CrossoverCategorical`が自動的に補われます（`eta`/`prob`は連続変数用の`crossover`から引き継がれます）。`GA.ask()`は親個体を変数型ごとの列に分割し、各`Crossover`を該当する列だけに適用してから結果を組み立て直します。既定のバッチモードでは型ごとの`crossover_batch()`を各列に適用し、逐次モードでは型ごとの`crossover()`を呼び出して従来のペア単位のルーティングと乱数列を保ちます。この仕組み上、`integer_crossover`/`categorical_crossover`へ独自クラスを渡す場合、その`n_children`/`n_parents`を連続変数用の`crossover`と一致させる必要があります。一致しない場合は`ConfigurationError`になります。

変数の型と`Crossover`の対応づけは[Problem](../problem_and_ranking/problem.md)の`variables`引数で決まります。

```{note}
`CrossoverBLXAlpha`のみ`@register()`済みで、他の6クラスは現状Registry未登録です。Registry経由でクラスを文字列から解決する場合はこの違いに注意してください。
```

### 外部ライブラリアダプタ

`PymooCrossover(operator, *, prob=None, n_parents=None, n_children=None)`は、構築済みの[pymoo](https://pymoo.org/)交叉演算子（例：`SBX()`）をラップし、既存のpymooベースの研究コードを`GA`内でそのまま再利用できるようにします。 `prob`/`n_parents`/`n_children`は、ラップした演算子自身の値が既定になります。

`GA`の既定のバッチモードでは、`PymooCrossover.crossover_batch()`がゲートを通過したバッチ全体に対して、ラップした演算子の`_do()`を1回だけ呼び、pymooの集団単位のベクトル化を維持します。`variation_execution="sequential"`では、継承した`crossover()`が1つの親グループをバッチ実装へ渡すため、直接`crossover()`を呼ぶ場合と同じくグループごとに`_do()`が1回呼ばれます。大きな集団でこのグループ単位のオーバーヘッドを避けられることが、既定のバッチモードでアダプタを使う主な利点です。`rng`はpymooの`random_state`引数を介してラップした演算子へ渡され、saealibのシード管理下で再現性が保たれます。

`pymoo` extraのインストールについては[インストール](../../getting_started/installation.md)を参照してください。

## 拡張フック

境界外に出た値を丸めるといった後処理だけを足したい場合、サブクラスを新設せずに`with_post(fn)`で既存の`Crossover`インスタンスへ処理を追加できます。 `with_post`は元のインスタンスを変更せず、`fn`を追加したコピーを返します。

```python
import numpy as np
from saealib import CrossoverBLXAlpha

base = CrossoverBLXAlpha(prob=1.0, alpha=0.5)


def clip_to_bounds(offspring, parents, rng, ctx=None):
    return np.clip(offspring, -1.0, 1.0)


repaired = base.with_post(clip_to_bounds)
```

`fn`のシグネチャは`fn(offspring, parents, rng, ctx) -> np.ndarray`で、既存のフック（既定では何もしない恒等関数）の結果を受け取って追加の変換を返します。 複数回`with_post`を呼べば、フックは呼び出した順に連結されます。

バッチモードでは、`GA`が`post_crossover_batch(offspring_batch, parents_batch, rng, ctx)`を呼びます。既定の実装は親グループごとに`post_crossover()`を呼ぶため、`with_post()`で追加したフックも引き続き動作します。後処理自体をベクトル化する場合は`post_crossover_batch()`を上書きします。この上書きでは、維持したい`with_post()`フックも自ら合成する必要があります。

## 独自Crossoverの実装方法

独自の交叉方式が必要な場合は、`Crossover`を継承して`crossover_batch()`を実装します。次の例は、各親ペアの平均を2個の子として返します。

```python
import numpy as np
from saealib import Crossover


class AverageCrossover(Crossover):
    def __init__(self, prob: float = 1.0):
        super().__init__()
        self.prob = prob

    def crossover_batch(self, parents_batch, bounds=None, rng=np.random.default_rng()):
        mean = parents_batch.mean(axis=1)
        return np.repeat(mean[:, np.newaxis, :], self.n_children, axis=1)
```

`n_parents`/`n_children`を2以外にする場合や、`bounds`を使う独自の丸め処理が必要な場合は、クラス属性を上書きして`crossover_batch()`内で`bounds`を使います。継承した`crossover()`が1つの親グループを自動的に処理します。`crossover()`だけの上書きが有効なのは、実装済みの具体的な交叉を`variation_execution="sequential"`専用に拡張する場合です。

## 関連コンポーネント

- [Algorithm](algorithm.md)：`GA`が`Crossover`をどう組み合わせるか
- [Mutation](mutation.md)：交叉の次に呼ばれる、対になる演算子
- [ParentSelection](parent_selection.md)：`Crossover`に渡す親個体を選ぶ演算子
- [Problem](../problem_and_ranking/problem.md)：整数変数とカテゴリ変数の定義と、混合変数向けCrossoverの対応関係
- [拡張のガイドライン](../extension_guidelines.md)：`with_post`系フックの一般的な設計思想

## 参照

- {py:class}`saealib.Crossover`
- {py:class}`saealib.CrossoverBLXAlpha`
- {py:class}`saealib.CrossoverSBX`
- {py:class}`saealib.CrossoverUniform`
- {py:class}`saealib.CrossoverOnePoint`
- {py:class}`saealib.CrossoverTwoPoint`
- {py:class}`saealib.CrossoverIntegerSBX`
- {py:class}`saealib.CrossoverCategorical`
- {py:class}`saealib.PymooCrossover`
