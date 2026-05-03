# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0b1] - 2026-05-03

Initial beta release of saealib.

### Added

- **High-level API**: `minimize()` and `maximize()` functions for quick setup with sensible defaults
- **Low-level API**: `Optimizer` builder with swappable components via `set_algorithm()`, `set_surrogate_manager()`, and `set_strategy()`
- **Algorithms**: Genetic Algorithm (`GA`) with crossover, mutation, and selection operators
- **Surrogate models**: RBF (Radial Basis Function) surrogate with configurable kernels
- **Acquisition functions**: pluggable acquisition function interface with built-in implementations
- **Optimization strategies**: `IndividualBasedStrategy`, `GenerationBasedStrategy`, `PreSelectionStrategy`
- **Multi-objective support**: Pareto-based archive and multi-objective problem interface
- **Constraint handling**: feasibility-aware selection and result reporting
- **Callbacks**: extensible callback interface for monitoring and early stopping
- **Archive**: solution archive with sorting and feasibility tracking
- **Problem definition**: `Problem` class supporting minimization (`weight=-1`) and maximization (`weight=+1`)
- **Termination criteria**: configurable stopping conditions
- **Type hints**: fully typed public API, PEP 561 compliant (`py.typed` marker included)

[0.1.0b1]: https://github.com/shlka/saealib/releases/tag/v0.1.0b1
