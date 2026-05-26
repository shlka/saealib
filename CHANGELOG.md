# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0b2] - 2026-05-21

Surrogate API flexibility — concentrated breaking changes to decouple surrogate output from objective semantics.

### Added

- **`GPSurrogate`**: Gaussian Process surrogate providing mean and uncertainty estimates; `SurrogatePrediction.std` is populated and usable with EI, LCB, PoF, and MaxUncertainty acquisition functions ([#78](https://github.com/shlka/saealib/issues/78))
- **Sklearn / torch adapter**: `SklearnSurrogate` wraps any scikit-learn estimator; `SVMSurrogate`, `NNSurrogate` (MLP), and `DTSurrogate` (Random Forest) added ([#72](https://github.com/shlka/saealib/issues/72))
- **`ArchiveBasedManager`**: abstract base for archive-direct scoring without a surrogate model; concrete implementations: `NoveltyManager` (nearest-neighbor distance), `DensityManager` (inverse ε-NN density), `NichingManager` (inter-candidate distance) ([#84](https://github.com/shlka/saealib/issues/84))
- **`Optimizer.validate()`**: pre-run configuration consistency check — validates operator presence, strategy/surrogate/algorithm compatibility, and reports misconfigurations with actionable error messages ([#73](https://github.com/shlka/saealib/issues/73))
- **Training data abstraction**: `TrainingData` / `TrainingSet` builder with `PairwiseComparisonSet`, `TopKBipartitionSet`, `LevelBasedSet`, and `FeasibilityClassificationSet` labelling strategies for flexible surrogate training data construction ([#67](https://github.com/shlka/saealib/issues/67))

### Fixed

- Intercept `NaN` surrogate predictions at `SurrogateManager` boundary and `OptimizationStrategy` to prevent silent downstream propagation

### Changed

- **`SurrogatePrediction.mean` renamed to `SurrogatePrediction.value`** (breaking): decouples raw surrogate output from the objective value written to offspring, preventing pbest corruption when non-regression surrogates (e.g., novelty scores, classification probabilities) are used ([#69](https://github.com/shlka/saealib/issues/69))
- **`SurrogatePrediction.tell_f` added** (breaking): strategies now assign objective function values via `tell_f` rather than `mean`, separating surrogate output semantics from tell semantics ([#69](https://github.com/shlka/saealib/issues/69))

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

[0.1.0b2]: https://github.com/shlka/saealib/compare/v0.1.0b1...v0.1.0b2
[0.1.0b1]: https://github.com/shlka/saealib/releases/tag/v0.1.0b1
