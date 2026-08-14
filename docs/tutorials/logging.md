---
primary_layer: layer2
---

# Logging Progress

Record optimization progress with the standard `logging` module.

## Default behavior

`Optimizer` registers a handler (`logging_generation`) that records progress at the start of every generation, but only when `minimize`/`maximize`'s `verbose=True` (the default).

However, this handler only calls `logging.getLogger(__name__).info(...)`, so nothing is displayed unless INFO-level output is enabled on Python's `logging` module.

```python
import numpy as np
from saealib import minimize


def expensive_func(x):
    return np.sum(x**2)


DIM = 5

# nothing is printed here since logging.basicConfig has not been called yet
result = minimize(
    expensive_func, dim=DIM, lb=[-5.0] * DIM, ub=[5.0] * DIM, max_fe=100, seed=0
)
```

To display progress, enable INFO level with `logging.basicConfig`.

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

result = minimize(
    expensive_func, dim=DIM, lb=[-5.0] * DIM, ub=[5.0] * DIM, max_fe=100, seed=0
)
# Generation 0 started. fe: 25. Best f: [14.04274116]
# Generation 1 started. fe: 27. Best f: [14.04274116]
# ...
```

`logging_generation` switches what it records based on the number of objectives: for single-objective it records the best objective value, and for multi-objective it records the size of the first non-dominated front and the range of values per objective.

If you don't want progress recorded at all, specify `verbose=False` to stop the handler from being registered.

```python
result = minimize(
    expensive_func,
    dim=DIM,
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
    max_fe=100,
    seed=0,
    verbose=False,
)
```

## Writing to a file

Adding a `FileHandler` to the `saealib.callback.handlers` logger writes progress out to a file.

```python
import logging

file_handler = logging.FileHandler("optimization.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

saealib_logger = logging.getLogger("saealib.callback.handlers")
saealib_logger.addHandler(file_handler)
saealib_logger.setLevel(logging.INFO)

result = minimize(
    expensive_func, dim=DIM, lb=[-5.0] * DIM, ub=[5.0] * DIM, max_fe=100, seed=0
)
```

## Hypervolume logging for multi-objective problems

For multi-objective problems, registering the handler returned by `logging_generation_hv(reference_point)` records the hypervolume for each generation.

```python
import numpy as np
from saealib import (
    Optimizer,
    Termination,
    max_fe,
    GenerationStartEvent,
    logging_generation,
    logging_generation_hv,
)
from saealib.benchmarks import zdt1

problem = zdt1(n_var=5)
optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(200)))

# remove the default logging_generation and swap in the HV-based one
optimizer.cbmanager.unregister(GenerationStartEvent, logging_generation)
optimizer.cbmanager.register(
    GenerationStartEvent, logging_generation_hv(reference_point=np.array([1.1, 1.1]))
)

ctx = optimizer.run()
# Generation 0. fe: 25. HV: 0.612345
# ...
```

By minimization convention, `reference_point` should be a value larger than the best achievable value for each objective.

## Warning-level logging

Some components record numerical issues via `logger.warning(...)`.

For example, `RBFSurrogate` issues a warning when the kernel matrix becomes ill-conditioned.

Even without calling `logging.basicConfig`, WARNING-level and above logs are shown on standard error via Python's "handler of last resort".

Note that, unlike INFO-level progress logs, this kind of warning is visible even without any configuration.

## Custom logging

If you want to log something other than what `logging_generation`/`logging_generation_hv` record, register your own handler on `CallbackManager`.

See [CallbackManager](../concepts/observation_and_state/callbacks.md) for the underlying mechanism.

## References

- {py:func}`saealib.logging_generation` / {py:func}`saealib.logging_generation_hv`
- {py:class}`saealib.CallbackManager` / {py:class}`saealib.GenerationStartEvent`
- {py:func}`saealib.minimize`
