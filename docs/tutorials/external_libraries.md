# Integrating External Libraries

`saealib` provides adapters that thinly wrap external machine learning and evolutionary-computation libraries behind its own abstract base classes.

The adapter only translates `saealib`-side data representations such as `Problem`/`Population`/`ctx`; the learning algorithm or search operator itself uses the external library's implementation as-is.

Currently, surrogate model adapters (scikit-learn, XGBoost, LightGBM, PyTorch) and pymoo adapters (`Crossover`/`Mutation`/`Algorithm`/`Problem`) are implemented. For users who would rather remove the runtime dependency on an external evolutionary-computation library, [Porting operators to native saealib code](#porting-operators-to-native-saealib-code) below covers rewriting a pymoo- or DEAP-style operator as a native `Crossover`/`Mutation` subclass instead.

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

Wrapping an external operator, as in the previous section, keeps a runtime dependency on the external library and pays a calling-granularity cost: saealib calls `crossover()`/`mutate()` once per parent group or individual, while most external libraries vectorize their operators over an entire population in one call. When that overhead matters, or when the runtime dependency itself is unwanted, porting the operator's logic into a native `Crossover`/`Mutation` subclass is usually a mechanical rewrite rather than a redesign — the core logic stays the same; only the batch loop and the RNG source change.

The one detail that is easy to get wrong when porting is *where* the individual-level probability gate lives. saealib's `GA` checks `Crossover.prob` itself before calling `crossover()`, but `Mutation.mutate()` must check `self.prob` itself — the same asymmetry already noted for `PymooMutation` above. Both pymoo's built-in mutations and DEAP's `toolbox.mutate()` (gated externally by `algorithms.varAnd`'s `mutpb`) leave this gate outside the operator, so a ported mutation usually needs a guard line that the original code did not have.

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

Ported to a native saealib `Mutation`, the batch dimension disappears; the only additions are the `prob` gate described above and the per-individual shape `(dim,)`:

```python
import numpy as np
from saealib.operators import Mutation


class MyMutation(Mutation):
    def __init__(self, prob=1.0, *, sigma=1.0, prob_var=0.5):
        super().__init__()
        self.prob = prob
        self.sigma = sigma
        self.prob_var = prob_var

    def mutate(self, p, mutate_range, rng=np.random.default_rng()):
        if rng.random() >= self.prob:
            return p.copy()
        c = p.copy()
        mask = rng.random(len(p)) < self.prob_var
        c[mask] += rng.normal(0, self.sigma, size=len(p))[mask]
        return c
```

### From DEAP

DEAP operators already work on one individual (or a pair) at a time, so porting is even more direct. A custom crossover:

```python
import random


def my_cx(ind1, ind2, swap_rate=0.5):
    for i in range(len(ind1)):
        if random.random() < swap_rate:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2
```

becomes a native `Crossover` by replacing the Python loop with a vectorized mask and `random.random()` with the `rng` saealib passes in, dropping DEAP's implicit global RNG state in favor of saealib's per-run `np.random.Generator` (kept reproducible via `minimize(..., seed=...)`):

```python
import numpy as np
from saealib.operators import Crossover


class MyCrossover(Crossover):
    def __init__(self, prob, swap_rate=0.5):
        super().__init__()
        self.prob = prob
        self.swap_rate = swap_rate

    def crossover(self, parent, bounds=None, rng=np.random.default_rng()):
        p1, p2 = parent[0], parent[1]
        mask = rng.random(len(p1)) < self.swap_rate
        c1 = np.where(mask, p2, p1)
        c2 = np.where(mask, p1, p2)
        return np.array([c1, c2])
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
