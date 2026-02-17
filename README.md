# saealib
![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange)

**Status: Active Development (Alpha)**
> **Warning**: This project is under active development. APIs are subject to change without notice. Operation is not guaranteed in production environments.

A comprehensive library for **Surrogate-Assisted Evolutionary Algorithms (SAEAs)** in Python.  
Designed for expensive optimization problems where function evaluations are costly, `saealib` provides a modular framework to combine evolutionary algorithms, surrogate models, and model management strategies.

## Key Features

- **Modular Architecture**: Easily mix and match Algorithms (e.g., GA), Surrogates (e.g., RBF), and Management Strategies.
- **Method Chaining**: The `Optimizer` class allows for fluent and readable configuration.
- **Customizable Components**:
  - **Algorithms**: Genetic Algorithms (GA) with various operators.
  - **Surrogates**: Radial Basis Function (RBF) networks with Gaussian kernels.
  - **Strategies**: Individual-based management strategies (e.g., generation-based or pre-selection).
  - **Operators**: Includes BLX-Alpha Crossover, Uniform Mutation, and various Selection methods.

## Installation

Since the package is in active development, it is recommended to install from the source.

### Requirements
- Python >= 3.10
- numpy
- scipy

### Install via uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Run active environment
uv run <command>
```

### Install via pip

```bash
git clone https://github.com/yourusername/saealib.git
cd saealib
pip install .
```

Or for development (editable mode):

```bash
pip install -e .
```

## Quick Start

Here is a simple example of how to use `saealib` to minimize a Sphere function using a Surrogate-Assisted GA (SAGA) with an RBF model.

```python
import numpy as np
from saealib import (
    GA,
    Optimizer,
    Problem,
    RBFsurrogate,
    LHSInitializer,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
    IndividualBasedStrategy,
    Termination,
    max_fe,
    gaussian_kernel
)

# 1. Define the Objective Function (e.g., Sphere Function)
def sphere(x):
    return np.sum(x**2)

# 2. Setup the Optimization Problem
dim = 5
problem = Problem(
    func=sphere,
    dim=dim,
    n_obj=1,
    weight=np.array([-1.0]),  # -1.0 implies minimization
    lb=[-5.0] * dim,          # Lower bounds
    ub=[5.0] * dim,           # Upper bounds
)

# 3. Configure Components

# Initialization: Latin Hypercube Sampling
initializer = LHSInitializer(
    n_init_archive=5 * dim,      # Initial samples for the surrogate
    n_init_population=4 * dim,   # Initial population size
    seed=42,
)

# Algorithm: Genetic Algorithm
algorithm = GA(
    crossover=CrossoverBLXAlpha(crossover_rate=0.7, gamma=0.4),
    mutation=MutationUniform(mutation_rate=0.3),
    parent_selection=SequentialSelection(),
    survivor_selection=TruncationSelection(),
)

# Surrogate Model: RBF with Gaussian Kernel
surrogate = RBFsurrogate(gaussian_kernel, dim)

# Strategy: Individual-Based Management
# knn=50: K-Nearest Neighbors for local modeling
# rsm=0.1: Ratio of surrogate model usage
strategy = IndividualBasedStrategy(knn=50, rsm=0.1)

# Termination Criterion
termination = Termination(max_fe(100))  # Stop after 100 function evaluations

# 4. Build and Run the Optimizer
opt = (
    Optimizer(problem)
    .set_initializer(initializer)
    .set_algorithm(algorithm)
    .set_termination(termination)
    .set_surrogate(surrogate)
    .set_strategy(strategy)
)

print("Starting optimization...")
opt.run()
print("Optimization finished.")
```

## Architecture Overview

`saealib` is built around the `Optimizer` class which orchestrates the interaction between:

- **Problem**: Defines the objective function, constraints, and bounds.
- **Algorithm**: The evolutionary search engine (e.g., GA).
- **Surrogate**: The approximate model used to replace expensive evaluations.
- **Strategy**: Decides when to use the surrogate and when to use the real function (e.g., Pre-selection, Generation control).
- **Initializer**: Generates the initial dataset.

## Contributing

Contributions are welcome! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

[MIT License](LICENSE) (Assuming standard open source license, please verify)
