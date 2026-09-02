<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo-dark.svg">
  <img alt="saealib" src="docs/_static/logo-light.svg" height="60">
</picture>

![Status: Beta](https://img.shields.io/badge/Status-Beta-orange)
![ci-tests](https://img.shields.io/github/actions/workflow/status/saealib/saealib/test.yml?branch=main&label=tests)
[![codecov](https://codecov.io/gh/saealib/saealib/branch/main/graph/badge.svg)](https://codecov.io/gh/saealib/saealib)
![pypi-saealib-v](https://img.shields.io/pypi/v/saealib)
![pypi-python-v](https://img.shields.io/pypi/pyversions/saealib)
[![Apache-2.0](https://custom-icon-badges.herokuapp.com/badge/license-Apache%202.0-8BB80A.svg?logo=law&logoColor=white)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/saealib)](https://pepy.tech/project/saealib)

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)
![ty](https://custom-icon-badges.demolab.com/badge/ty-261230.svg?logo=ty-astral-logo)
[![NumPy](https://custom-icon-badges.herokuapp.com/badge/NumPy-9C8AF9.svg?logo=NumPy&logoColor=white)](https://numpy.org/)

</div>

**Status: Active Development (Beta)**
> **Warning**: This project is under active development. APIs are subject to change without notice. Operation is not guaranteed in production environments.

A comprehensive library for **Surrogate-Assisted Evolutionary Algorithms (SAEAs)** in Python.  
Designed for expensive optimization problems where function evaluations are costly, `saealib` provides a modular framework to combine evolutionary algorithms, surrogate models, and model management strategies.

## Table of Contents

- [Why saealib](#why-saealib)
- [Documents](#documents)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Contributing](#contributing)
- [License](#license)

## Why saealib

`saealib` is designed around composable components for surrogate-assisted evolutionary optimization.
Surrogates, acquisition functions, model-management strategies, and optimization pipelines can be replaced
independently, making it suitable for researchers and practitioners who want to experiment with SAEA
structure rather than treat an algorithm as monolithic.

General-purpose libraries such as [pymoo](https://github.com/anyoptimization/pymoo) provide broad evolutionary
optimization coverage. `saealib` complements them by focusing on modular surrogate-assisted optimization and
the surrogate/strategy layer.

## Documents
See the [official documentation](https://saealib.github.io/saealib/index.html) for the architecture,
component reference, tutorials, and API.

## Key Features

- **Composable SAEA architecture**: major optimization components — Algorithm, Surrogate, SurrogateManager, AcquisitionFunction, OptimizationStrategy, EvaluationPlanner, Evaluator, and more — are independently replaceable and can be assembled fluently with `Optimizer.set_*()`.

- **First-class evaluation decisions**: separate candidate generation, surrogate scoring, and true-evaluation planning. New surrogate-assisted strategies or batch-selection ideas can often be implemented by replacing a single component without rewriting the optimization loop.

- **Validated composition**: component contracts and the graph compiler check structural compatibility — including data flow, required services, lifecycle, state access, and runtime capabilities — before optimization begins, with structured diagnostics for invalid configurations.

- **Surrogate and model-management toolkit**: built-in RBF plus adapters for scikit-learn, XGBoost, LightGBM, and PyTorch, with regression, feasibility-classification, and pairwise-comparison workflows.

- **Evolutionary and multi-objective optimization**: GA and PSO with configurable operators, mixed-variable support, Pareto/epsilon-dominance ranking, NSGA-II/III and R-NSGA-II comparators, hypervolume indicators, and decomposition methods for MOEA/D-style setups.

- **Constraint handling**: equality and inequality constraints, epsilon-constraint tolerance scheduling, feasibility surrogates, and gradient-based repair.

- **Interoperability**: reuse implementations from other optimization ecosystems through adapters for pymoo and DEAP, from individual crossover/mutation operators to complete algorithms and problems.

- **Experimentation and observability**: typed pipeline callbacks, execution history, result visualization, asynchronous evaluation, and experiment tooling for multi-trial sweeps, parallel execution, checkpointing, and resume.

## Installation

### Requirements
- Python >= 3.10

```bash
pip install saealib
# or
uv add saealib
```

saealib is still pre-1.0, so the public API may change between minor versions. The command
above installs the latest stable release; pre-releases need `pip install --pre saealib`.

### Optional extras

| Extra | Adds |
|---|---|
| `sklearn` | scikit-learn-based surrogates |
| `xgboost` | XGBoost surrogate |
| `lightgbm` | LightGBM surrogate |
| `torch` | PyTorch-based components |
| `parallel` | joblib-based parallel evaluation |
| `pymoo` | pymoo adapters for crossover, mutation, algorithm, and problem |
| `deap` | DEAP adapters for crossover, mutation, and generate/update algorithms |
| `viz` | Matplotlib-based result and history plots |
| `tqdm` | tqdm progress bars for experiment sweeps |
| `rich` | Rich progress bars for experiment sweeps |
| `hdf5` | HDF5 trial-result storage and loading via h5py |
| `all` | All optional dependencies listed above |

```bash
pip install "saealib[sklearn,parallel]"
# or install everything
pip install "saealib[all]"
```

### Install from source

For contributing or a development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick Start

`minimize()` / `maximize()` run a surrogate-assisted optimization end-to-end with sensible defaults
(GA algorithm, RBF surrogate, individual-based strategy):

```python
import numpy as np
from saealib import minimize


def sphere(x):
    return np.sum(x**2)


result = minimize(
    sphere,
    dim=5,
    lb=[-5.0] * 5,
    ub=[5.0] * 5,
    max_fe=500,
    seed=0,
)

print(result.x, result.f)
```

Every default is overridable — pass `algorithm='PSO'`, `surrogate='rbf'`, `strategy='gb'`/`'ps'`, or
your own component instances (`Algorithm`, `Surrogate`/`SurrogateManager`, `OptimizationStrategy`).

For per-generation inspection, custom pipelines, or research-style control loops, build an `Optimizer`
directly and drive it with `.iterate()` instead of `.run()`. See
[the documentation](https://saealib.github.io/saealib/index.html) for the low-level API and full component
reference.

## Contributing

Contributions are welcome! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

[Apache License 2.0](LICENSE)
