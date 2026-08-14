---
primary_layer: layer4
---

# Initializer

Runの開始時には、`Optimizer` が `Initializer` に初期Population、Archive、ParetoArchive、OptimizationStateの構築を委譲します。
サンプリング方法を変えるときは、Optimizer全体ではなくInitializerを差し替えます。

現行の標準経路では、`GenomeInitializer` が `Problem.space` からGenomeをサンプリングします。
`LHSInitializer`、`RandomInitializer`、`SobolInitializer` は、ベクトルの `x` 配列を使う互換性用Initializerとして残っています。

## Initializerの役割

`Initializer` の実装境界は `initialize(provider, problem) -> OptimizationState` です。
`provider` は、AlgorithmやEvaluatorなど、すでに構築されたComponentへアクセスするためのProviderです。
`problem` は対象となる [Problem](../problem_and_ranking/problem.md) です。

## Built-in Initializers

| Class | Sampling method |
|---|---|
| `GenomeInitializer` | `Problem.space.sample()` によるGenome生成 |
| `LHSInitializer` | `scipy.stats.qmc.LatinHypercube` |
| `RandomInitializer` | `rng.uniform` |
| `SobolInitializer` | `scipy.stats.qmc.Sobol(scramble=True)` |

`GenomeInitializer` は、SearchSpaceが提供するGenomeをそのままArchiveへ登録します。
他の3つは同じコンストラクタ `(n_init_archive, n_init_population, seed=None)` を持ち、ベクトル表現の初期点を生成します。
いずれも初期Archiveを評価し、Comparatorで並べた上位候補を初期Populationへ渡します。

```
Sample (n_init_archive points)
  -> Evaluate via provider.evaluator.evaluate_batch
  -> add to archive / pareto_archive
  -> Sort via problem.comparator.sort_population
  -> Feed the top n_init_population into population
```

The three classes' implementations are nearly identical except for the one line doing the sampling — this is a deliberate choice for simplicity, a design that doesn't over-abstract the shared processing.

## Base-class helper methods

The `Initializer` base provides two helper methods reusable in custom implementations.

**`_create_attrs(problem, provider)`**：`Population` と `Archive` の `PopulationAttribute` を組み立てます。
旧来のベクトル経路では `x`、`f`、`g`、`cv` を使い、Algorithmが必要とする補助属性を追加します。
`GenomeInitializer` ではGenomeが専用の列として管理されるため、標準属性に `x` を含めません。

**`_create_context(problem, archive, pareto_archive, population, rng)`**：`OptimizationState` を構築します。
Comparatorが `NSGA3Comparator` で内部乱数生成器をまだ持たない場合は、ここで `rng.spawn(1)[0]` を設定します。

## カスタムInitializerの実装

独自のベクトルサンプリングを追加する場合は `Initializer` を継承して `initialize()` を実装します。
新しいGenome表現を使う場合は、`Problem.space.sample()` と `Problem.space.validate()` を使うInitializerを実装します。

Genome-nativeな初期化では、次の順序を保ちます。

1. Algorithmの `population_class`、`archive_class`、`create_pareto_archive()` で入れ物を作る。
2. `OptimizationState` を構築する。
3. `problem.space.sample(n, rng)` でGenomeを生成し、`problem.space.validate()` で検証する。
4. `provider.evaluator.evaluate_batch(genomes, problem)` で評価する。
5. 候補ID、Genome、評価結果をArchiveとParetoArchiveへ登録する。
6. 評価回数を更新し、Comparatorで並べた候補をPopulationへ渡す。

次の骨格は、Genome-nativeなInitializerが使う入力と出力の境界を示しています。

```python
from saealib import GenomeInitializer


class CustomGenomeInitializer(GenomeInitializer):
    def initialize(self, provider, problem):
        # problem.space.sample() を独自のサンプリング処理へ置き換える。
        # 生成したGenomeはspace.validate()で検証し、Evaluatorへ渡す。
        ...
```

初期評価の開始と終了は `CallbackManager` から観測できます。
評価Requestの候補IDとGenomeを維持する必要があるため、Genomeを `x` 配列へ暗黙に変換して管理しません。

`Optimizer.set_initializer(initializer)` でInitializerを差し替えます。

## Related components

- [OptimizationState](../observation_and_state/optimization_state.md): The state object `initialize()` ultimately returns
- [Population](../observation_and_state/population.md): The `Population`/`Archive`/`ParetoArchive` being constructed
- [Evaluator](evaluation.md): Used to evaluate the initial samples
- [CallbackManager](../observation_and_state/callbacks.md): Observing `InitialEvaluationStartEvent`/`InitialEvaluationEndEvent`

## References

- {py:class}`saealib.Initializer`
- {py:class}`saealib.LHSInitializer`
- {py:class}`saealib.RandomInitializer`
- {py:class}`saealib.SobolInitializer`
