# Initializer

Before starting a run, `Optimizer` delegates constructing the initial `Population`/`Archive`/`ParetoArchive` and the first `OptimizationState` all at once to `Initializer`, a swappable component.
To change the sampling method, you only need to swap out this `Initializer`, not `Optimizer` itself.

## Initializer's role

`Initializer` requires only one method, `initialize(provider, problem) -> OptimizationState`, to be implemented.
`provider` is a `ComponentProvider` for accessing other already-constructed components such as `Algorithm`/`Evaluator`, and `problem` is the [Problem](problem.md) to solve.

## Built-in Initializers

| Class | Sampling method |
|---|---|
| `LHSInitializer` | `scipy.stats.qmc.LatinHypercube` |
| `RandomInitializer` | `rng.uniform` |
| `SobolInitializer` | `scipy.stats.qmc.Sobol(scramble=True)` |

All three classes share the same constructor, `(n_init_archive, n_init_population, seed=None)`.
They also share the same flow: sample `n_init_archive` points and evaluate them, then feed the top `n_init_population` (sorted by `comparator`) into the initial `Population`.

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

**`_create_attrs(problem, provider)`**: Assembles the list of `PopulationAttribute`s for `Population`/`Archive`.
Adds algorithm-specific attributes (such as PSO's velocity) returned by `provider.algorithm.get_required_attrs(problem)` on top of the standard attributes `x`/`f`/`g`/`cv`.

**`_create_context(problem, archive, pareto_archive, population, rng)`**: Constructs `OptimizationState`.
If `comparator` is an `NSGA3Comparator` and doesn't yet have an internal rng, this is where `rng.spawn(1)[0]` is injected.

## Implementing a custom Initializer

If you need a custom sampling method, subclass `Initializer` and implement `initialize()`.
The three built-in classes' implementations serve directly as a template; the responsibility they carry internally can be organized into the following 9 steps.

1. Construct `Population`/`Archive`/`ParetoArchive` via `provider.algorithm.population_class`/`archive_class`/`create_pareto_archive`
2. Construct `OptimizationState` via `_create_context`
3. Sample from the design-variable space
4. Fire `provider.dispatch(InitialEvaluationStartEvent(...))`
5. Evaluate via `provider.evaluator.evaluate_batch(x, problem)`
6. `add` the results to `archive`/`pareto_archive`
7. Add to the evaluation count via `ctx.count_fe(...)`
8. Fire `provider.dispatch(InitialEvaluationEndEvent(...))`
9. Sort via `problem.comparator.sort_population` and feed the top entries into `population`

The following example is an `Initializer` that generates initial samples with `scipy.stats.qmc.Halton`.

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

`InitialEvaluationStartEvent`/`InitialEvaluationEndEvent` are events observable via [CallbackManager](callbacks.md).
See [Evaluator](evaluation.md) for the details of evaluation itself.

Swap it via `Optimizer.set_initializer(initializer)`.

## Related components

- [OptimizationState](optimization_state.md): The state object `initialize()` ultimately returns
- [Population](population.md): The `Population`/`Archive`/`ParetoArchive` being constructed
- [Evaluator](evaluation.md): Used to evaluate the initial samples
- [CallbackManager](callbacks.md): Observing `InitialEvaluationStartEvent`/`InitialEvaluationEndEvent`

## References

- {py:class}`saealib.Initializer`
- {py:class}`saealib.LHSInitializer`
- {py:class}`saealib.RandomInitializer`
- {py:class}`saealib.SobolInitializer`
