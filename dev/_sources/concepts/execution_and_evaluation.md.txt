---
primary_layer: cross
related_layers: [layer2, layer3]
page_type: entry
---

# Execution and evaluation

On the existing Stage-based path, the Initializer creates the initial population and the OptimizationStrategy advances the work for each generation.
The Algorithm generates candidates, the SurrogateManager and AcquisitionFunction predict and score them, and the Evaluator performs the expensive true evaluation.
The results update the Algorithm's population and Archive, and generation processing repeats until Termination meets its stopping condition.

## Pipeline flow

The diagram shows the data flow from initialization through the termination check.
A configuration without a Surrogate can omit prediction and scoring.

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Create initial population"] --> STEP
    subgraph STEP["OptimizationStrategy.step()<br/>Process one generation"]
        direction TB
        ASK["Algorithm.ask()<br/>Generate candidates"] --> PREDICT["SurrogateManager.predict()<br/>Predict"]
        PREDICT --> SCORE["AcquisitionFunction<br/>Score candidates"]
        SCORE --> SEL["Select candidates for true evaluation"]
        SEL --> EVAL["Evaluator → Problem<br/>Expensive evaluation"]
        EVAL --> TELL["Algorithm.tell()<br/>Update population"]
    end
    STEP --> TERM{"Termination?"}
    TERM -- "No" --> STEP
    TERM -- "Yes" --> RESULT([Result])
    EVAL -- "Evaluated points" --> ARC[("Archive")]
    ARC -. "Training data" .-> PREDICT
```

Each Stage emits typed events through the CallbackManager.
This lets you observe progress or swap behavior at runtime without subclassing a Stage.

## Component roles

| Component | Role |
|---|---|
| [Problem](problem_and_ranking/problem.md) | Defines the objective function, design variables, search bounds, constraints, and optimization direction. |
| [Initializer](execution_and_evaluation/initialization.md) | Creates the initial population and Archive before the loop starts. |
| [Algorithm](search_algorithms/algorithm.md) | Generates candidates with `ask()` and applies evaluation results to the population with `tell()`. |
| [OptimizationStrategy](execution_and_evaluation/strategies.md) | Builds one generation's processing and decides which candidates receive true evaluation. |
| [SurrogateManager](surrogate_modeling/surrogate_manager.md) | Coordinates Surrogate training and prediction. |
| [Surrogate](surrogate_modeling/surrogate.md) | Predicts from training data; it does not define scoring. |
| [AcquisitionFunction](surrogate_modeling/acquisition_functions.md) | Converts predictions into scores for candidate selection. |
| [Evaluator](execution_and_evaluation/evaluation.md) | Evaluates the objective function sequentially or in parallel. |
| [Population](observation_and_state/population.md) | Holds evaluated individuals and population data. |
| [Termination](execution_and_evaluation/termination.md) | Determines when generation processing ends. |
| [CallbackManager](observation_and_state/callbacks.md) | Observes and records events from the entire pipeline. |

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`play;sd-mr-1` Initialization (Initializer)
:link: execution_and_evaluation/initialization
:link-type: doc
Choose how to create the initial population and Archive.
:::

:::{grid-item-card} {fa}`microchip;sd-mr-1` Evaluator
:link: execution_and_evaluation/evaluation
:link-type: doc
Run sequential and parallel objective evaluations.
:::

:::{grid-item-card} {fa}`chess-knight;sd-mr-1` OptimizationStrategy
:link: execution_and_evaluation/strategies
:link-type: doc
Assemble candidate generation, prediction, evaluation planning, and state updates.
:::

:::{grid-item-card} {fa}`stop;sd-mr-1` Termination
:link: execution_and_evaluation/termination
:link-type: doc
Combine stopping conditions such as evaluation and generation limits.
:::

::::

```{toctree}
:hidden:

execution_and_evaluation/initialization
execution_and_evaluation/evaluation
execution_and_evaluation/strategies
execution_and_evaluation/termination
```
