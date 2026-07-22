# Algorithm

`saealib`は、候補解の生成と個体群の更新をAsk-Tellという2つの手続きに分解し、その実装を`Algorithm`という差し替え可能なコンポーネントに委ねている。
`Optimizer.set_algorithm(algorithm)`で差し替える。

## Algorithmの役割

`Algorithm`が実装を要求するメソッドは2つある。

**`ask(ctx, provider, n_offspring=None) -> Population`**：候補解を生成する。
`n_offspring`が`None`の場合、通常は現在の個体群サイズと同数の候補を生成する。

**`tell(ctx, provider, offspring) -> None`**：`ask()`が生成した候補群`offspring`を使って`ctx.population`を更新する。

このAsk-Tellという分割により、「どの候補を真に評価するか」という判断を`Algorithm`の外側（[OptimizationStrategy](strategies.md)）に切り出せる。

## 組み込みAlgorithm

| クラス | 探索方式 |
|---|---|
| `GA` | 交叉と突然変異による遺伝的アルゴリズム |
| `PSO` | 速度と位置の更新による粒子群最適化 |

### GA：演算子を組み合わせるコンテナ

`GA`は、それ自体が探索ロジックを持つのではなく、`crossover`/`mutation`/`parent_selection`/`survivor_selection`という4つの演算子を注入して組み立てるコンテナである。

```python
GA(crossover, mutation, parent_selection, survivor_selection, *,
   duplicate_elimination=None,
   integer_crossover=None, integer_mutation=None,
   categorical_crossover=None, categorical_mutation=None)
```

各演算子の挙動とパラメータの詳細は、それぞれ独立したページで扱う。

- [Crossover](crossover.md) — 親個体から子個体を生成する
- [Mutation](mutation.md) — 子個体に摂動を加える
- [ParentSelection](parent_selection.md) — 交叉に使う親個体を選ぶ
- [SurvivorSelection](survivor_selection.md) — 次世代に残す個体を選ぶ

`GA.tell()`は、現世代の個体群と`ask()`が生成した子個体群を1つのプールへ統合し、`survivor_selection`でそのプールから生存個体を選ぶ、という(μ+λ)方式で個体群を更新する。
プールに何を含めるか（親を含めるかどうか）は`GA`側の責務であり、`SurvivorSelection`のインターフェース自体には現れない。

### GA：混合変数問題への対応

設計変数に整数変数とカテゴリ変数が混在する問題では、`GA`は連続変数用の`crossover`/`mutation`とは別に、型ごとの演算子を使い分ける。
`integer_crossover`/`integer_mutation`/`categorical_crossover`/`categorical_mutation`を省略すると、それぞれ`CrossoverIntegerSBX`/`MutationIntegerUniform`/`CrossoverCategorical`/`MutationCategorical`が自動的に補われる。
補われる演算子の確率パラメータ（`prob`/`prob_var`）は、連続変数用の`crossover`/`mutation`から引き継がれる。

型ごとに補う演算子であっても、`n_children`/`n_parents`は連続変数用の`crossover`と一致していなければならない。
一致しない場合は`ConfigurationError`になる。
これは、`GA`が親個体を変数の型ごとの列に分割し、各演算子を該当する列だけに適用してから結果を1つの個体へ組み立て直すという実装上、子の個体数と親の個体数が型ごとにずれてはならないためである。

変数の型は[Problem](problem.md)の`variables`引数で定義する。

### GA：補助ユーティリティ

**`duplicate_elimination`**引数に`DuplicateElimination(atol=1e-16, rtol=0.0, max_retries=100)`を渡すと、現在の個体群と重複する子個体を再生成で置き換える。
重複判定の許容誤差は`atol`/`rtol`で、再生成の試行上限は`max_retries`で指定する。
省略時（既定の`None`）は重複除去を行わない。

`saealib.repair_clipping(candidates, bounds)`は、候補群を`(lb, ub)`の範囲へ`np.clip`するだけの独立したユーティリティ関数である。
`GA`自体は候補の修復を[ConstraintHandler](constraints.md)の`repair()`（既定はやはり`np.clip`）と`Problem.repair()`（`Variable`ごとの射影）を通じて行っており、`repair_clipping`はGAの内部処理に自動的に組み込まれているわけではない。
`ConstraintHandler`を経由しない独自の評価パイプラインを書く場合など、同等のクリッピング処理を単体で使いたい場面向けに公開されている。

### PSO

`PSO(w=0.7, c1=1.5, c2=1.5, v_max=None)`は、慣性項`w`、個体最良解への追従`c1`、群最良解への追従`c2`という重みで速度を更新し、その速度で位置を進める。
`v_max`を指定すると、各次元の速度の大きさをその値でクランプする。

`GA`とは異なり演算子を注入する構成ではなく、`ask()`/`tell()`の中で速度と位置の更新とpbest（個体ごとの最良解）の追跡を直接行う。
群最良解（リーダー）は、全粒子のpbestから`ctx.comparator`を使って選ばれるため、単目的の`Comparator`であればどれを使っても自動的に対応する。
多目的PSO（MOPSO）には非劣解集合を管理する専用のサブクラスが必要であり、組み込みの`PSO`は単目的問題を対象にしている。

## 独自Algorithmの実装方法

独自の探索アルゴリズムが必要な場合は、`Algorithm`を継承して`ask()`/`tell()`を実装する。
`get_required_attrs()`（`Population`に追加で必要な属性。無ければ空リスト）と`population_class`/`archive_class`プロパティも実装する必要がある。

次の例は、各個体を独立にガウス摂動し、親より良ければ置き換えるだけの単純なアルゴリズムである。

```python
import numpy as np
from saealib import Algorithm, Population, Archive


class RandomWalkAlgorithm(Algorithm):
    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def get_required_attrs(self, problem):
        return []

    @property
    def population_class(self):
        return Population

    @property
    def archive_class(self):
        return Archive

    def ask(self, ctx, provider, n_offspring=None):
        x = ctx.population.get_array("x")
        noise = ctx.rng.normal(0.0, self.sigma, size=x.shape)
        new_x = np.clip(x + noise, ctx.problem.lb, ctx.problem.ub)
        offspring = ctx.population.empty_like(capacity=len(new_x))
        offspring.extend({"x": new_x})
        return offspring

    def tell(self, ctx, provider, offspring):
        parent_f = ctx.population.get_array("f")
        child_f = offspring.get_array("f")
        better = (child_f[:, 0] * ctx.direction[0]) > (parent_f[:, 0] * ctx.direction[0])
        ctx.population.get_array("x")[better] = offspring.get_array("x")[better]
        ctx.population.get_array("f")[better] = offspring.get_array("f")[better]
```

`tell()`は`offspring`に真の目的関数値`f`が既に設定されている前提で呼ばれる（`OptimizationStrategy`が`ask()`と`tell()`の間で評価を行う）。
PSOのように個体ごとの付加情報（pbestなど）が必要な場合は、`get_required_attrs()`でその属性を宣言する。

## 関連コンポーネント

- [Crossover](crossover.md) / [Mutation](mutation.md) / [ParentSelection](parent_selection.md) / [SurvivorSelection](survivor_selection.md) — `GA`が組み合わせる4つの演算子
- [Problem](problem.md) — `variables`による混合変数の定義
- [ConstraintHandler](constraints.md) — `GA`が候補の修復に使う`repair()`
- [OptimizationStrategy](strategies.md) — `ask()`/`tell()`の間で真の評価を行うかどうかを判断する
- [Population](population.md) — `ask()`/`tell()`が読み書きする個体群

## 参照

- {py:class}`saealib.Algorithm`
- {py:class}`saealib.GA`
- {py:class}`saealib.PSO`
- {py:class}`saealib.DuplicateElimination`
- {py:func}`saealib.repair_clipping`
