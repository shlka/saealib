# Integrating External Libraries

`saealib` provides adapters that thinly wrap external machine learning and evolutionary-computation libraries behind its own abstract base classes.

The adapter only translates `saealib`-side data representations such as `Problem`/`Population`/`ctx`; the learning algorithm or search operator itself uses the external library's implementation as-is.

Currently, surrogate model adapters (scikit-learn, XGBoost, LightGBM, PyTorch) and pymoo adapters (`Crossover`/`Mutation`/`Algorithm`/`Problem`) are implemented.

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

## References

- {py:class}`saealib.Surrogate`
- {py:class}`saealib.SklearnGPRSurrogate` / {py:class}`saealib.SklearnRFRSurrogate` / {py:class}`saealib.SklearnSVMSurrogate` / {py:class}`saealib.SklearnNNSurrogate`
- {py:class}`saealib.SklearnXGBSurrogate` / {py:class}`saealib.SklearnLGBMSurrogate`
- {py:class}`saealib.TorchSurrogate`
- {py:class}`saealib.PymooCrossover` / {py:class}`saealib.PymooMutation`
- {py:class}`saealib.PymooAlgorithm`
- {py:class}`saealib.PymooProblem`
- {py:func}`saealib.minimize`
