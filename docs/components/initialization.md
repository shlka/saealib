# Initializer

`Optimizer`は実行を始める前に、初期`Population`/`Archive`/`ParetoArchive`と最初の`OptimizationState`を一括で構築する処理を、`Initializer`という差し替え可能なコンポーネントに委ねている。
サンプリング方法を変えたいときは、`Optimizer`本体ではなくこの`Initializer`だけを差し替えればよい。

## Initializerの役割

`Initializer`が実装を要求するメソッドは`initialize(provider, problem) -> OptimizationState`の1つだけである。
`provider`は`Algorithm`/`Evaluator`など構築済みの他コンポーネントへアクセスするための`ComponentProvider`、`problem`は解くべき[Problem](problem.md)である。

## 組み込みInitializer

| クラス | サンプリング方法 |
|---|---|
| `LHSInitializer` | `scipy.stats.qmc.LatinHypercube` |
| `RandomInitializer` | `rng.uniform` |
| `SobolInitializer` | `scipy.stats.qmc.Sobol(scramble=True)` |

3クラスとも、コンストラクタは`(n_init_archive, n_init_population, seed=None)`で共通している。
`n_init_archive`件をサンプリングして評価し、その中から`comparator`でソートした上位`n_init_population`件を初期`Population`へ投入する、という流れも共通である。

```
サンプリング(n_init_archive件)
  -> provider.evaluator.evaluate_batch で評価
  -> archive / pareto_archive へ add
  -> problem.comparator.sort_population でソート
  -> 上位 n_init_population 件を population へ投入
```

3クラスの実装はサンプリング方法の1行以外ほぼ重複しているが、これは意図的な単純さの選択であり、共通処理を過度に抽象化しない設計になっている。

## 基底クラスのヘルパーメソッド

`Initializer`基底には、独自実装で再利用できる2つのヘルパーメソッドが用意されている。

**`_create_attrs(problem, provider)`**：`Population`/`Archive`用の`PopulationAttribute`一覧を組み立てる。
`x`/`f`/`g`/`cv`という標準属性に、`provider.algorithm.get_required_attrs(problem)`が返すアルゴリズム固有の属性（PSOの速度など）を追加する。

**`_create_context(problem, archive, pareto_archive, population, rng)`**：`OptimizationState`を構築する。
`comparator`が`NSGA3Comparator`で、まだ内部rngを持たない場合はここで`rng.spawn(1)[0]`を注入する。

## 独自Initializerの実装方法

独自のサンプリング方法が必要な場合は、`Initializer`を継承して`initialize()`を実装する。
組み込み3クラスの実装がそのままテンプレートになり、内部で担う責務は次の9ステップに整理できる。

1. `provider.algorithm.population_class`/`archive_class`/`create_pareto_archive`で`Population`/`Archive`/`ParetoArchive`を構築する
2. `_create_context`で`OptimizationState`を構築する
3. 設計変数空間からサンプリングする
4. `provider.dispatch(InitialEvaluationStartEvent(...))`を発火する
5. `provider.evaluator.evaluate_batch(x, problem)`で評価する
6. 結果を`archive`/`pareto_archive`へ`add`する
7. `ctx.count_fe(...)`で評価回数を加算する
8. `provider.dispatch(InitialEvaluationEndEvent(...))`を発火する
9. `problem.comparator.sort_population`でソートし、上位件数を`population`へ投入する

次の例は、`scipy.stats.qmc.Halton`で初期サンプルを生成する`Initializer`である。

```python
import numpy as np
import scipy.stats
from saealib import Initializer, InitialEvaluationStartEvent, InitialEvaluationEndEvent


class HaltonInitializer(Initializer):
    def __init__(self, n_init_archive, n_init_population, seed=None):
        self.n_init_archive = n_init_archive
        self.n_init_population = n_init_population
        self.seed = seed

    def initialize(self, provider, problem):
        provider_seed = getattr(provider, "seed", None)
        rng = np.random.default_rng(
            provider_seed if provider_seed is not None else self.seed
        )
        attrs = self._create_attrs(problem, provider)

        population = provider.algorithm.population_class(
            attrs=attrs, init_capacity=self.n_init_population
        )
        archive = provider.algorithm.archive_class(
            attrs=attrs, init_capacity=self.n_init_archive
        )
        pareto_archive = provider.algorithm.create_pareto_archive(
            attrs=attrs, init_capacity=self.n_init_archive, problem=problem
        )

        ctx = self._create_context(problem, archive, pareto_archive, population, rng)

        archive_x = scipy.stats.qmc.Halton(d=problem.dim, seed=rng).random(
            self.n_init_archive
        )
        archive_x = scipy.stats.qmc.scale(archive_x, problem.lb, problem.ub)

        provider.dispatch(InitialEvaluationStartEvent(ctx=ctx, candidates_x=archive_x))
        result = provider.evaluator.evaluate_batch(archive_x, problem)

        for i in range(self.n_init_archive):
            data = {
                "x": archive_x[i], "f": result.f[i], "g": result.g[i],
                "cv": float(result.cv[i]),
            }
            archive.add(data)
            pareto_archive.add(data)

        ctx.count_fe(self.n_init_archive)
        provider.dispatch(InitialEvaluationEndEvent(ctx=ctx, archive=archive))

        sorted_idx = problem.comparator.sort_population(archive)
        archive_sorted = archive.extract(sorted_idx)
        archive.clear()
        archive.extend(archive_sorted)
        population.extend(archive[: self.n_init_population])
        return ctx
```

`InitialEvaluationStartEvent`/`InitialEvaluationEndEvent`は[CallbackManager](callbacks.md)経由で観察できるイベントである。
評価そのものの詳細は[Evaluator](evaluation.md)を参照する。

`Optimizer.set_initializer(initializer)`で差し替える。

## 関連コンポーネント

- [OptimizationState](optimization_state.md) — `initialize()`が最終的に返す状態オブジェクト
- [Population](population.md) — 構築対象の`Population`/`Archive`/`ParetoArchive`
- [Evaluator](evaluation.md) — 初期サンプルの評価に使う
- [CallbackManager](callbacks.md) — `InitialEvaluationStartEvent`/`InitialEvaluationEndEvent`の観察

## 参照

- {py:class}`saealib.Initializer`
- {py:class}`saealib.LHSInitializer`
- {py:class}`saealib.RandomInitializer`
- {py:class}`saealib.SobolInitializer`
