---
primary_layer: layer1
---

# Quickstart
Have you finished installing saealib?
See [here](./installation.md) for installation instructions.

## Your first run
Let's solve an optimization problem with the least amount of code.
```python
from saealib import minimize
from saealib.benchmarks import rastrigin

problem = rastrigin(n_var=10)
result = minimize(func=problem)
print(f"objective: {result.f}")
print(f"solution: {result.x}")
print(f"evaluated: {result.fe}")
print(f"generation: {result.gen}")
```
Here we solve a minimization problem for the 10-dimensional Rastrigin function (`saealib.benchmarks.rastrigin`) from the benchmark package provided by `saealib`.
`minimize()` / `maximize()` is a high-level API that runs an optimization just by specifying parameters.

## Optimizing an arbitrary function
The previous section used a benchmark problem; here, let's look at an example of optimizing an arbitrary function.
When using `saealib`'s benchmark package, the required parameters are passed to the API automatically, but when passing an arbitrary function (`callable`), you need to specify a few parameters yourself.
```python
import numpy as np
from saealib import minimize


def rastrigin(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


result = minimize(func=rastrigin, dim=10, lb=[-5.12] * 10, ub=[5.12] * 10)
print(f"objective: {result.f}")
print(f"solution: {result.x}")
```
Here we define the 10-dimensional Rastrigin function and solve it as a minimization problem.
The `func` parameter can be any `callable` object that takes a numpy array and returns an evaluation value.
By specifying an expensive-to-evaluate objective function here — such as a simulation (CAE) or the training of a machine learning model — this can be applied to efficient parameter search.

## Next steps
What's shown here is only a part of `saealib`.
See the following pages for detailed guides.

- [Layerの案内](index.md#利用方法を選ぶ)：目的に合うLayerと関連ページを選びます
- [Tutorials](../tutorials/index.md): Guides for specific usage scenarios
- [Concepts](../concepts/index.md): Detailed usage of each component
- [API Reference](../api/index.md): Reference for all parameters
