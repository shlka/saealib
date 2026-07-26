# Components

Detailed usage of each component and guidelines for extending it.
First we show how the components fit together in the overall pipeline, then move into page-by-page detail.

## Overall pipeline structure

`Optimizer` bundles the following components and drives the generation loop until `Termination` decides to stop.
`OptimizationStrategy` orchestrates the processing for a single generation.
`Algorithm` generates candidate solutions (ask), `SurrogateManager` scores them cheaply, the strategy decides which candidates receive an expensive true evaluation, and the result is reflected in both `Algorithm`'s population (tell) and the `Archive`.
`Archive` also doubles as the surrogate's training data.

```{mermaid}
flowchart TD
    INIT["Initializer<br/>(generates the initial population)"] --> STEP
    subgraph STEP["OptimizationStrategy.step() (one generation)"]
        direction TB
        ASK["Algorithm.ask()<br/>Generate candidates"] --> SCORE["SurrogateManager<br/>score_candidates()"]
        SCORE --> SEL["Select candidates<br/>for true evaluation"]
        SEL --> EVAL["Evaluator →<br/>Problem (expensive)"]
        EVAL --> TELL["Algorithm.tell()<br/>Update population"]
    end
    STEP --> TERM{"Termination?"}
    TERM -- "Continue" --> STEP
    TERM -- "Done" --> RESULT([Result])
    subgraph SM["SurrogateManager"]
        direction TB
        SUR["Surrogate<br/>fit / predict"] --> ACQ["AcquisitionFunction<br/>Prediction → scalar score"]
    end
    SCORE -.-> SM
    EVAL -- "Evaluated points" --> ARC[("Archive")]
    ARC -. "Training data" .-> SUR
```

Each stage fires typed events via `CallbackManager`, so you can observe or intervene in the pipeline's progress without subclassing (see [CallbackManager](callbacks.md)).

The role of each component is as follows.

| Components | Role |
|---|---|
| [Problem](problem.md) | Defines the objective function, design variables, search range, constraints, and optimization direction |
| [Initializer](initialization.md) | Generates the initial population and archive before the loop starts |
| [Algorithm](algorithm.md) | The evolutionary search itself (GA/PSO). `ask()` generates candidates, `tell()` updates the population |
| [OptimizationStrategy](strategies.md) | Orchestrates one generation's pipeline and decides which candidates receive a true evaluation |
| [SurrogateManager](surrogate_manager.md) | Bridges surrogate fitting and scoring, exposing `score_candidates()` |
| [Surrogate](surrogate.md) | Fits and predicts using the archive's data. Knows nothing about how scoring works |
| [AcquisitionFunction](acquisition_functions.md) | Converts predictions into a scalar score (higher is better). Knows nothing about the model's details |
| [Evaluator](evaluation.md) | Runs true evaluation (sequentially, or in parallel via the `parallel` extra) |
| [Archive](population.md) | Accumulates every truly evaluated point. Also doubles as the surrogate's training dataset |
| [Termination](termination.md) | Judges the loop's termination condition (default: maximum evaluations) |
| [CallbackManager](callbacks.md) | Observes and records events throughout the pipeline, and also enables swapping components at runtime |

## Assembling with the low-level API

Using `Optimizer` directly lets you swap out each component individually via method chaining.

```python
from saealib import GA, Optimizer, Problem, IndividualBasedStrategy
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.surrogate import GlobalSurrogateManager
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel
from saealib.acquisition import MeanPrediction
from saealib.termination import Termination, max_fe

problem = Problem(func, dim=5, lb=[-5] * 5, ub=[5] * 5, n_obj=1, direction=[-1])

opt = (
    Optimizer(problem)
    .set_algorithm(GA(
        CrossoverBLXAlpha(prob=0.7, alpha=0.4),
        MutationUniform(prob_var=0.3),
        SequentialSelection(),
        TruncationSelection(),
    ))
    .set_surrogate_manager(
        GlobalSurrogateManager(RBFSurrogate(gaussian_kernel, dim=5), MeanPrediction())
    )
    .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.1))
    .set_termination(Termination(max_fe(100)))
)

ctx = opt.run()
```

The high-level API (`minimize()`/`maximize()`) is this same pipeline auto-configured with sensible defaults.
For research use cases that need per-generation inspection or custom loop control, use `.iterate()` instead of `.run()`.
Worked examples using `.iterate()` can be found in tutorials such as [Single-Objective Optimization](../tutorials/single_objective.md).

See [OptimizationStrategy](strategies.md) for how each strategy configures this pipeline internally, and [Stage](stage.md) for the contract of the individual stages that make up the pipeline.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`wand-magic-sparkles;sd-mr-1` Extension guidelines
:link: extension_guidelines
:link-type: doc

For when subclassing is too heavy: `with_post`/`with_post_fit`, `Pipeline`/`Stage`, `CallbackManager`, `Registry`.
:::

::::

## Problem definition and ranking

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`compass;sd-mr-1` Problem
:link: problem
:link-type: doc

Defines the objective function, variables, direction, and constraints.
:::

:::{grid-item-card} {fa}`filter;sd-mr-1` ConstraintHandler
:link: constraints
:link-type: doc

How to implement your own constraint-violation aggregation, feasibility judgment, and repair strategy.
:::

:::{grid-item-card} {fa}`arrow-down-up-across-line;sd-mr-1` Comparator
:link: comparators
:link-type: doc

Ranking solutions. When to use NSGA2/SPEA2/HV and the like.
:::

:::{grid-item-card} {fa}`crown;sd-mr-1` Dominator
:link: dominance
:link-type: doc

How to implement your own dominance relations, such as Pareto dominance or ε-dominance.
:::

:::{grid-item-card} {fa}`arrow-down-wide-short;sd-mr-1` NonDominatedSorter
:link: nondominated_sorting
:link-type: doc

Swapping the non-dominated sorting algorithm. Also covers computing crowding distance and SPEA2 fitness.
:::

:::{grid-item-card} {fa}`scissors;sd-mr-1` Decomposition
:link: decomposition
:link-type: doc

MOEA/D-style scalarizing (decomposition) functions and `DecompositionComparator`.
:::

::::

## Search algorithms

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`dna;sd-mr-1` Algorithm
:link: algorithm
:link-type: doc

GA and PSO, and how to implement a custom `Algorithm`.
:::

:::{grid-item-card} {fa}`code-fork;sd-mr-1` Crossover
:link: crossover
:link-type: doc

BLX-α, SBX, uniform crossover, and more. When to use each for mixed-variable problems.
:::

:::{grid-item-card} {fa}`bolt-lightning;sd-mr-1` Mutation
:link: mutation
:link-type: doc

Uniform, Gaussian, polynomial mutation, and more. When to use each for mixed-variable problems.
:::

:::{grid-item-card} {fa}`hand-pointer;sd-mr-1` ParentSelection
:link: parent_selection
:link-type: doc

Parent-selection schemes such as tournament and roulette-wheel selection.
:::

:::{grid-item-card} {fa}`user-check;sd-mr-1` SurvivorSelection
:link: survivor_selection
:link-type: doc

Survivor-selection (generational replacement) schemes such as truncation selection.
:::

::::

## Surrogate modeling

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`brain;sd-mr-1` Surrogate
:link: surrogate
:link-type: doc

Built-in surrogates, external-library adapters, and how to implement a custom `Surrogate`.
:::

:::{grid-item-card} {fa}`sitemap;sd-mr-1` SurrogateManager
:link: surrogate_manager
:link-type: doc

Bridges the surrogate's predictions and the acquisition function's scoring.
:::

:::{grid-item-card} {fa}`database;sd-mr-1` TrainingSet
:link: training_set
:link-type: doc

Where the surrogate's training data comes from, and with which labels.
:::

:::{grid-item-card} {fa}`calculator;sd-mr-1` AcquisitionFunction
:link: acquisition_functions
:link-type: doc

Scores candidates from the surrogate's predictions.
:::

:::{grid-item-card} {fa}`toggle-on;sd-mr-1` AccuracyBasedSurrogateSwitcher
:link: surrogate_switching
:link-type: doc

Accuracy metrics, evaluation methods, and dynamic component switching inside an `iterate()` loop.
:::

::::

## Execution and evaluation strategy

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`play;sd-mr-1` Initializer
:link: initialization
:link-type: doc

How the initial population and archive are generated (LHS/Random/Sobol), and custom implementations.
:::

:::{grid-item-card} {fa}`microchip;sd-mr-1` Evaluator
:link: evaluation
:link-type: doc

Sequential and parallel backends for objective function evaluation.
:::

:::{grid-item-card} {fa}`chess-knight;sd-mr-1` OptimizationStrategy
:link: strategies
:link-type: doc

Decides which candidates receive a true evaluation.
:::

:::{grid-item-card} {fa}`stop;sd-mr-1` Termination
:link: termination
:link-type: doc

How to combine termination conditions (`|`/`&`/`~`) and write your own.
:::

::::

## Observation and core data structures

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`eye;sd-mr-1` CallbackManager
:link: callbacks
:link-type: doc

Observe the optimization pipeline, for logging and reaching into its internals.
:::

:::{grid-item-card} {fa}`bars-staggered;sd-mr-1` Stage
:link: stage
:link-type: doc

The stages that make up the generation loop inside `OptimizationStrategy`. How to implement a custom `Stage`.
:::

:::{grid-item-card} {fa}`hard-drive;sd-mr-1` OptimizationState
:link: optimization_state
:link-type: doc

The state (ctx) that flows through the pipeline. Its main fields and checkpointing.
:::

:::{grid-item-card} {fa}`users;sd-mr-1` Population
:link: population
:link-type: doc

The data structure and API for the population and archive. How Archive/ParetoArchive work.
:::

::::

```{toctree}
:hidden:

extension_guidelines
problem
constraints
comparators
dominance
nondominated_sorting
decomposition
algorithm
crossover
mutation
parent_selection
survivor_selection
surrogate
surrogate_manager
training_set
acquisition_functions
surrogate_switching
initialization
evaluation
strategies
termination
callbacks
stage
optimization_state
population
```
