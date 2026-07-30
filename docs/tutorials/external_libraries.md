# Integrating External Libraries

`saealib` provides adapters that thinly wrap external machine learning and evolutionary-computation libraries behind its own abstract base classes.

The adapter only translates `saealib`-side data representations such as `Problem`/`Population`/`ctx`; the learning algorithm or search operator itself uses the external library's implementation as-is.

Currently, surrogate model adapters (scikit-learn, XGBoost, LightGBM, PyTorch), pymoo adapters (`Crossover`/`Mutation`/`Algorithm`/`Problem`), DEAP adapters (`Crossover`/`Mutation`/`Algorithm`), and a Nevergrad `Algorithm` adapter are implemented. For users who would rather remove the runtime dependency on an external evolutionary-computation library, [Porting operators to native saealib code](#porting-operators-to-native-saealib-code) below covers rewriting a pymoo- or DEAP-style operator as a native `Crossover`/`Mutation` subclass instead.

## Installation

Each adapter can only be used once the corresponding `extra` is installed.

```bash
pip install "saealib[sklearn]"
```

See [Installation](../getting_started/installation.md) for how to install and the full list of extras.

Importing an adapter without the corresponding `extra` installed raises `ImportError`.

## Surrogate adapters

Each adapter implements `saealib`'s `Surrogate` base class, and can be passed to the `surrogate` argument just like the built-in `RBFSurrogate`.

| Class | Required `extra` | Wrapped model |
|--------|--------|--------|
| `SklearnGPRSurrogate` | `sklearn` | Gaussian Process Regressor |
| `SklearnRFRSurrogate` | `sklearn` | Random Forest Regressor |
| `SklearnSVMSurrogate` | `sklearn` | Support Vector Regression |
| `SklearnNNSurrogate` | `sklearn` | Multi-layer Perceptron |
| `SklearnXGBSurrogate` | `xgboost` | XGBoost regression |
| `SklearnLGBMSurrogate` | `lightgbm` | LightGBM regression |
| `TorchSurrogate` | `torch` | Any PyTorch `nn.Module` |

Keyword arguments to the constructor are passed straight through to the corresponding library's model.

```python
import numpy as np
from saealib import minimize, SklearnGPRSurrogate


def expensive_func(x):
    return np.sum(x**2)


DIM = 10

result = minimize(
    expensive_func,
    dim=DIM,
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
    surrogate=SklearnGPRSurrogate(),
    max_fe=300,
    seed=0,
)
```

Passing a `Surrogate` instance to the `surrogate` argument works the same as the `RBFSurrogate` example in "Switching components" in [Single-Objective Optimization](single_objective.md) — internally it is wrapped in a `LocalSurrogateManager`.

See [Surrogate](../components/surrogate.md) for adapters aimed at classification problems (e.g. feasibility classification) and the detailed arguments of each adapter.

## pymoo adapters

Each adapter wraps an already-constructed pymoo object and implements the corresponding `saealib` base class, so it can be passed anywhere that base class is expected.

| Class | Wraps |
|--------|--------|
| `PymooCrossover` | A pymoo `Crossover` (e.g. `SBX()`) |
| `PymooMutation` | A pymoo `Mutation` (e.g. `PM()`) |
| `PymooAlgorithm` | A pymoo `Algorithm` (e.g. `NSGA2()`, `DE()`) |
| `PymooProblem` | A pymoo `Problem` (a benchmark, or existing research code) |

`PymooCrossover`/`PymooMutation` drop straight into `GA`:

```python
import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from saealib import GA, TournamentSelection, TruncationSelection, minimize
from saealib.operators import PymooCrossover, PymooMutation


def expensive_func(x):
    return np.sum(x**2)


DIM = 10

result = minimize(
    expensive_func,
    dim=DIM,
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
    algorithm=GA(
        crossover=PymooCrossover(SBX(eta=15)),
        mutation=PymooMutation(PM(eta=20)),
        parent_selection=TournamentSelection(2),
        survivor_selection=TruncationSelection(),
    ),
    surrogate="rbf",
    max_fe=300,
    seed=0,
)
```

`PymooAlgorithm` reuses a whole pymoo algorithm's own search and survival logic, and `PymooProblem` reuses an existing pymoo problem definition unchanged:

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from saealib import minimize
from saealib.algorithms import PymooAlgorithm
from saealib.problem import PymooProblem
from saealib.strategies.direct import DirectStrategy

problem = PymooProblem(get_problem("zdt1"))
result = minimize(
    problem,
    algorithm=PymooAlgorithm(NSGA2(pop_size=20)),
    surrogate="rbf",
    strategy=DirectStrategy(),
    max_fe=200,
    pop_size=20,
    seed=0,
)
```

`PymooAlgorithm` runs in "engine mode": the wrapped pymoo algorithm owns its own population and internal survival state, and `ctx.population` is refreshed from it after every generation rather than being the source of truth.
See [Algorithm](../components/algorithm.md) for what this means in practice (no checkpoint/resume, `n_offspring` is ignored, and `PreSelectionStrategy`'s partial `tell()` needs an explicit opt-in).

(porting-operators-to-native-saealib-code)=
## Porting operators to native saealib code

Wrapping an external operator, as in the previous section, keeps a runtime dependency on that library.
The pymoo adapters already preserve population-level vectorization in `GA`'s default batch mode, calling the wrapped operator once for the whole batch or gated subset, so porting is not normally needed to remove per-item calling overhead.
Port an operator when you want to remove the runtime dependency, or when its logic should be directly auditable and citable as native saealib code for a software paper.
The core logic usually stays the same; the main changes are adapting the array shape where necessary and using saealib's RNG.

The one detail that is easy to get wrong when porting is *where* the individual-level probability gate lives.
saealib's `GA` checks `Crossover.prob` itself and passes only gated parent groups to `crossover_batch()`, but `Mutation.mutate_batch()` must draw one `self.prob` gate per row and leave ungated rows unchanged.
Both pymoo's built-in mutations and DEAP's `toolbox.mutate()` (gated externally by `algorithms.varAnd`'s `mutpb`) leave this gate outside the operator, so a ported mutation usually needs a gate array that the original code did not have.

### From pymoo

A custom pymoo `Mutation` typically vectorizes over the whole batch `X` (shape `(n_individuals, dim)`):

```python
import numpy as np
from pymoo.core.mutation import Mutation as PymooMutationBase


class MyPymooMutation(PymooMutationBase):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def _do(self, problem, X, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng()
        Xp = X.copy()
        mask = rng.random(Xp.shape) < 0.5
        Xp[mask] += rng.normal(0, self.sigma, size=Xp.shape)[mask]
        return Xp
```

Ported to a native saealib `Mutation`, the batch axis stays in place.
The `_do()` body maps almost directly to `mutate_batch()`; the main addition is saealib's per-row `prob` gate:

```python
import numpy as np
from saealib.operators import Mutation


class MyMutation(Mutation):
    def __init__(self, prob=1.0, *, sigma=1.0, prob_var=0.5):
        super().__init__()
        self.prob = prob
        self.sigma = sigma
        self.prob_var = prob_var

    def mutate_batch(self, candidates_batch, mutate_range, rng=np.random.default_rng()):
        candidates_batch = np.asarray(candidates_batch, dtype=float)
        n, dim = candidates_batch.shape
        gate = rng.random(n) < self.prob
        result = candidates_batch.copy()
        if not np.any(gate):
            return result

        selected = result[gate]
        mask = rng.random((len(selected), dim)) < self.prob_var
        noise = rng.normal(0, self.sigma, size=selected.shape)
        selected[mask] += noise[mask]
        result[gate] = selected
        return result
```

### From DEAP

DEAP operators work on one individual or parent pair at a time, so a native saealib port must add and vectorize over a leading batch axis.
A custom crossover:

```python
import random


def my_cx(ind1, ind2, swap_rate=0.5):
    for i in range(len(ind1)):
        if random.random() < swap_rate:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2
```

becomes a native `Crossover` by vectorizing over both the parent pairs and dimensions.
It also replaces `random.random()` with the `rng` saealib passes in, dropping DEAP's implicit global RNG state in favor of saealib's per-run `np.random.Generator` (kept reproducible via `minimize(..., seed=...)`):

```python
import numpy as np
from saealib.operators import Crossover


class MyCrossover(Crossover):
    def __init__(self, prob, swap_rate=0.5):
        super().__init__()
        self.prob = prob
        self.swap_rate = swap_rate

    def crossover_batch(self, parents_batch, bounds=None, rng=np.random.default_rng()):
        n_pair, _, dim = parents_batch.shape
        p1, p2 = parents_batch[:, 0, :], parents_batch[:, 1, :]
        mask = rng.random((n_pair, dim)) < self.swap_rate
        c1 = np.where(mask, p2, p1)
        c2 = np.where(mask, p1, p2)
        return np.stack((c1, c2), axis=1)
```

`toolbox.mate` is gated externally by `varAnd`'s `cxpb`, matching saealib's `GA`, so — unlike mutation — no extra gate is needed here.

### Algorithm- and Problem-level code

Porting a whole search algorithm (survival selection, archive management, index-coupled state such as DE) is not a mechanical rewrite in the same way: it means reimplementing `Algorithm.ask()`/`tell()` from scratch. For pymoo specifically, the engine-mode `PymooAlgorithm` adapter above covers this case without a rewrite; for other libraries, see [Algorithm](../components/algorithm.md) for what a native `Algorithm` subclass needs to implement.

## References

- {py:class}`saealib.Surrogate`
- {py:class}`saealib.SklearnGPRSurrogate` / {py:class}`saealib.SklearnRFRSurrogate` / {py:class}`saealib.SklearnSVMSurrogate` / {py:class}`saealib.SklearnNNSurrogate`
- {py:class}`saealib.SklearnXGBSurrogate` / {py:class}`saealib.SklearnLGBMSurrogate`
- {py:class}`saealib.TorchSurrogate`
- {py:class}`saealib.PymooCrossover` / {py:class}`saealib.PymooMutation`
- {py:class}`saealib.PymooAlgorithm`
- {py:class}`saealib.PymooProblem`
- {py:func}`saealib.minimize`
