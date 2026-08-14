---
primary_layer: layer3
---

# Algorithm

`saealib` は候補生成とフィードバックの消費を分離し、その契約を `Algorithm` にまとめます。
`Algorithm` は `Optimizer.set_algorithm(algorithm)` で差し替えられます。

現行の `Algorithm` は、候補を生成するProposerと、評価結果を消費するFeedbackConsumerを兼ねる契約です。
通常の利用では、利用者がこの契約を意識する必要はありません。
独自の探索アルゴリズムを実装するときだけ、次の状態境界を実装します。

## Algorithmの役割

`Algorithm` は、読み取り専用の `StateView` を受け取り、状態を直接変更せずに結果を返します。

**`ask(request, state) -> ProposalBatch`**：`ProposalRequest` と `StateView` から候補のバッチを生成します。
`request.n_offspring` が `None` の場合の候補数は、アルゴリズムが現在のPopulationから決定します。

**`tell(feedback, state) -> StatePatch`**：`FeedbackBatch` を消費し、適用すべき状態変更を `StatePatch` として返します。
`tell()` は `OptimizationState` や `Population` を直接変更しません。

この分離によって、どの候補を高コストな真の評価へ送るかを `Algorithm` の外側に置けます。
その判断は [OptimizationStrategy](../execution_and_evaluation/strategies.md) と評価計画、FeedbackPolicyの組み合わせが担います。

## Built-in Algorithms

| Class | Search method |
|---|---|
| `GA` | Genetic algorithm via crossover and mutation |
| `PSO` | Particle swarm optimization via velocity and position updates |
| `GenomeGA` | SearchSpaceのGenomeを直接扱う遺伝的アルゴリズム |

`GenomeGA` は、`GenomeBatch` と `SearchSpace` の契約を使うGenome-native経路のAlgorithmです。
固定幅のdense vectorを前提とする `GA` と同じ `x` 列の経路へ暗黙に変換せず、Genomeの表現と空間サービスを維持します。

### GA: a container that combines operators

`GA` itself has no search logic; it's a container assembled by injecting four operators: `crossover`/`mutation`/`parent_selection`/`survivor_selection`.

```python
GA(crossover, mutation, parent_selection, survivor_selection, *,
   duplicate_elimination=None,
   variation_execution="batch",
   integer_crossover=None, integer_mutation=None,
   categorical_crossover=None, categorical_mutation=None)
```

The behavior and parameters of each operator are covered on their own dedicated pages.

- [Crossover](crossover.md): Generates offspring from parent individuals
- [Mutation](mutation.md): Adds perturbation to offspring
- [ParentSelection](parent_selection.md): Chooses the parent individuals used for crossover
- [SurvivorSelection](survivor_selection.md): Chooses the individuals kept for the next generation

`variation_execution` selects batch or sequential execution for crossover and mutation, and a dictionary can control them independently.
Batch is the default; for the full operation/hook ordering and reproducibility semantics, see the `GA` API reference below.

`GA.tell()` は現在のPopulationと `ask()` が生成した候補を一つのプールにまとめ、`survivor_selection` で次のPopulationを選びます。
これは $(\mu+\lambda)$ 型の更新に相当します。
What goes into the pool (whether parents are included) is `GA`'s own responsibility, and doesn't appear in `SurvivorSelection`'s interface itself.

### GA: handling mixed-variable problems

For problems whose design variables mix integer and categorical variables, `GA` uses type-specific operators in addition to the `crossover`/`mutation` for continuous variables.
If `integer_crossover`/`integer_mutation`/`categorical_crossover`/`categorical_mutation` are omitted, `CrossoverIntegerSBX`/`MutationIntegerUniform`/`CrossoverCategorical`/`MutationCategorical` are supplied automatically, respectively.
The probability parameters (`prob`/`prob_var`) of the automatically supplied operators are inherited from the continuous-variable `crossover`/`mutation`.

Even for operators supplied per type, `n_children`/`n_parents` must match the continuous-variable `crossover`.
A mismatch raises `ConfigurationError`.
This is because `GA` splits parent individuals into columns by variable type, applies each operator only to its corresponding columns, and then reassembles the results into a single individual — an implementation under which the number of children and parents must not diverge between types.
The default batch mode calls each type-specific operator's batch method on its columns.
`variation_execution="sequential"` uses the earlier per-pair/per-individual routing and preserves its random-number sequence.

Variable types are defined via [Problem](../problem_and_ranking/problem.md)'s `variables` argument.

### GA: auxiliary utilities

Passing `DuplicateElimination(atol=1e-16, rtol=0.0, max_retries=100)` to the **`duplicate_elimination`** argument replaces offspring that duplicate the current population by regenerating them.
The tolerance for duplicate detection is given by `atol`/`rtol`, and the maximum number of regeneration attempts by `max_retries`.
If omitted (the default `None`), no duplicate elimination is performed.

`saealib.repair_clipping(candidates, bounds)` is a standalone utility function that simply `np.clip`s a set of candidates into `(lb, ub)`.
`GA` itself repairs candidates via [ConstraintHandler](../problem_and_ranking/constraints.md)'s `repair()` (which also defaults to `np.clip`) and `Problem.repair()` (per-`Variable` projection); `repair_clipping` is not automatically wired into GA's internal processing.
It's exposed for cases where you want the same clipping behavior standalone — for example, when writing a custom evaluation pipeline that doesn't go through `ConstraintHandler`.

### PSO

`PSO(w=0.7, c1=1.5, c2=1.5, v_max=None)` updates velocity using weights for the inertia term `w`, attraction to the personal best `c1`, and attraction to the swarm best `c2`, then advances position by that velocity.
Specifying `v_max` clamps each dimension's velocity magnitude to that value.

Unlike `GA`, it isn't assembled by injecting operators — `ask()`/`tell()` directly perform the velocity and position updates and track each particle's pbest (personal best).
The swarm best (leader) is chosen from all particles' pbests using `ctx.comparator`, so any single-objective `Comparator` is automatically supported.
Multi-objective PSO (MOPSO) requires a dedicated subclass that manages a non-dominated solution set; the built-in `PSO` targets single-objective problems.

### External library adapters

`PymooAlgorithm(pymoo_algorithm, *, allow_partial_tell=False)` wraps an already-constructed [pymoo](https://pymoo.org/) algorithm (e.g. `NSGA2()`, `DE()`) so an existing pymoo algorithm can drive saealib's ask-tell loop and surrogate-assisted strategies unchanged.

Unlike `GA`/`PSO`, which treat `ctx.population` as the source of truth, `PymooAlgorithm` runs in "engine mode": the wrapped pymoo algorithm owns its own population and internal survival state, and `ctx.population` is refreshed from it at the end of every `tell()` — a mirror, not the source of truth.
This is the only way to reuse a pymoo algorithm's own tested survival logic unchanged, but it comes with real limits worth knowing before reaching for it:

- No checkpoint/resume — `OptimizationState.save()` only serializes saealib's own arrays, not the wrapped algorithm's internal state.
- `n_offspring` passed to `ask()` is ignored; the offspring count is fixed by the wrapped algorithm's own configuration (e.g. its `pop_size`).
- `PreSelectionStrategy`'s top-k truncation hands `tell()` only a subset of what `ask()` produced. By default this raises `ConfigurationError`, since index-coupled algorithms (e.g. differential evolution) would silently misalign parents and offspring under a partial `tell()`; pass `allow_partial_tell=True` to opt in anyway.

See [Installation](../../getting_started/installation.md) for the `pymoo` extra.

## Implementing a custom Algorithm

独自の探索アルゴリズムを実装するときは、`Algorithm` を継承して `ask()` と `tell()` を実装します。
`get_required_attrs()`、`population_class`、`archive_class` も、既存のPopulation互換経路を使う場合には実装が必要です。

新しい実装では、`ask()` は `ProposalBatch` を返し、`tell()` は `StatePatch` を返します。
`state` は宣言された状態キーだけを参照できる `StateView` です。
候補数、候補ID、評価結果の対応付けは `ProposalBatch` と `FeedbackBatch` の契約に従います。

次の骨格は、独自Algorithmが実装する境界だけを示しています。

```python
from saealib import Algorithm
from saealib.algorithms import ProposalRequest
from saealib.core.contracts import FeedbackBatch, ProposalBatch
from saealib.core.state import StatePatch, StateView
from saealib.population import Archive, Population


class CustomAlgorithm(Algorithm):
    """候補生成とFeedbackの消費を実装するAlgorithmの骨格。"""

    def get_required_attrs(self, problem):
        return []

    @property
    def population_class(self):
        return Population

    @property
    def archive_class(self):
        return Archive

    def ask(self, request: ProposalRequest, state: StateView) -> ProposalBatch:
        # state.context と宣言済みのStateKeyから現在のPopulationを読む。
        # 候補IDとFeedbackRequirementを含むProposalBatchを返す。
        ...

    def tell(self, feedback: FeedbackBatch, state: StateView) -> StatePatch:
        # FeedbackBatchを読み、Populationなどへの変更をStatePatchにまとめる。
        ...
```

`tell()` が受け取るFeedbackは、どの評価経路から届いたかと、候補IDとの対応を保持します。
真の評価だけを必要とするか、予測値を受け入れるかは、Algorithmの契約とFeedbackPolicyで宣言します。
PSOのpbestのような永続的な補助情報は、Population属性または宣言済みのStateKeyとして扱います。

## Related components

- [Crossover](crossover.md) / [Mutation](mutation.md) / [ParentSelection](parent_selection.md) / [SurvivorSelection](survivor_selection.md): The four operators `GA` combines
- [Problem](../problem_and_ranking/problem.md): Defining mixed variables via `variables`
- [ConstraintHandler](../problem_and_ranking/constraints.md): The `repair()` that `GA` uses to repair candidates
- [OptimizationStrategy](../execution_and_evaluation/strategies.md): Decides whether to perform true evaluation between `ask()`/`tell()`
- [Population](../observation_and_state/population.md): Algorithmの互換性経路で利用するPopulationとGenomeBatch

## References

- {py:class}`saealib.Algorithm`
- {py:class}`saealib.GA`
- {py:class}`saealib.PSO`
- {py:class}`saealib.PymooAlgorithm`
- {py:class}`saealib.DuplicateElimination`
- {py:func}`saealib.repair_clipping`
