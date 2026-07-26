# ParentSelection

`GA` (`saealib.GA`) delegates choosing the parent individuals used for crossover from the population to `ParentSelection`, a swappable operator.
To change the selection pressure (how much better individuals are favored), you only need to swap out this `ParentSelection`, not `GA` itself.

## ParentSelection's role

`ParentSelection` requires only one method, `select(ctx, population, n_pair, n_parents, rng=...)`, to be implemented.
It returns the indices of `n_pair` groups of parents, each consisting of `n_parents` individuals, as an array of shape `(n_pair, n_parents)`.

## Built-in ParentSelections

| Class | Parameters | Characteristics |
|---|---|---|
| `TournamentSelection` | `tournament_size` | Randomly draws `tournament_size` individuals and picks the best, repeated `n_pair * n_parents` times {cite}`miller1995tournament` |
| `SequentialSelection` | None | Performs no comparison; simply assigns population indices in order |
| `RouletteWheelSelection` | None | Roulette-wheel selection with a linear rank-based probability derived from rank |

`TournamentSelection` selects the best individual within each tournament by repeating 1-vs-1 comparisons via `ctx.comparator.compare_population` {cite}`blickle1996selection`.
Because `compare_population` is also defined on Comparators that can't use a direct two-point comparison (`compare()`), such as [SPEA2Comparator](comparators.md) or [HypervolumeComparator](comparators.md), it works correctly regardless of which `Comparator` it's paired with.

`SequentialSelection` performs no comparison at all, making it the simplest selection scheme, with no notion of selection pressure.
It can be paired with any `Comparator`.

`RouletteWheelSelection` converts the rank returned by `ctx.comparator.sort_population` into probabilities, rather than using raw fitness.
This lets it compute selection probabilities without numerical issues, even for problems where objective values can be negative or vary wildly in scale.

```{note}
Only `SequentialSelection` is currently `@register()`ed; `TournamentSelection`/`RouletteWheelSelection` are not yet registered with the Registry.
Keep this difference in mind if you resolve classes from strings via the Registry.
```

## Implementing a custom ParentSelection

If you need a custom selection scheme, subclass `ParentSelection` and implement only `select()`.
The following example is a selection scheme that chooses parents completely at random, with no comparison.

```python
import numpy as np
from saealib import ParentSelection


class RandomPairSelection(ParentSelection):
    """A selection scheme that chooses parent individuals completely at random."""

    def select(self, ctx, population, n_pair, n_parents, rng=np.random.default_rng()):
        n_pop = len(population)
        return rng.integers(0, n_pop, size=(n_pair, n_parents))
```

Implementing it to reference `ctx.comparator` lets you build a custom scheme with selection pressure, like `TournamentSelection`.

## Related components

- [Algorithm](algorithm.md): How `GA` combines `ParentSelection`
- [Crossover](crossover.md): The operator that receives the parents chosen by `ParentSelection`
- [Comparator](comparators.md): Used by `TournamentSelection`/`RouletteWheelSelection` to compare individuals

## References

- {py:class}`saealib.ParentSelection`
- {py:class}`saealib.TournamentSelection`
- {py:class}`saealib.SequentialSelection`
- {py:class}`saealib.RouletteWheelSelection`
