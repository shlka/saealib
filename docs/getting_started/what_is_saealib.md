# What is saealib

saealib is a general-purpose Python library for surrogate-assisted evolutionary algorithms (SAEA).
Evolutionary algorithms (EA), surrogate models, and model management strategies are modularized, and you build and run an algorithm by combining them.

## What is SAEA

An evolutionary algorithm (EA) is an optimization algorithm that mimics biological evolution.
However, because the evolutionary process requires repeatedly evaluating individuals, it faces a challenge with expensive optimization problems.

SAEA addresses this challenge by substituting evaluation with a lightweight mathematical model, reducing the number of expensive evaluations.

### A typical EA (GA)

<details>
<summary><b>Typical EA process flow (click to expand)</b></summary>

```{mermaid}
flowchart TD
    A[Generate initial population] --> B["Evaluate with objective function"]
    B --> C[Select parents]
    C --> D[Selection, crossover, mutation]
    D --> E["Evaluate with objective function"]
    E --> F[Select next generation]
    F --> G{Termination condition?}
    G -- No --> C
    G -- Yes --> H[Return best solution]

    style B fill:#e57373,color:#fff,stroke:#c62828
    style D fill:#e57373,color:#fff,stroke:#c62828
```

</details>

Because the entire population is evaluated every generation, the evaluation cost grows large.

### SAEA (individual-based GA)

<details>
<summary><b>SAEA process flow (click to expand)</b></summary>

```{mermaid}
flowchart TD
    A[Generate initial population] --> B["Evaluate with objective function"]
    B --> D[Select parents]
    D --> E[Selection, crossover, mutation]
    E --> F[Fit surrogate model]
    F --> G["Score with acquisition function"]
    G --> H[Select promising candidates]
    H --> I["Evaluate with objective function"]
    I --> J{Termination condition?}
    J -- No --> D
    J -- Yes --> K[Return best solution]

    style B fill:#e57373,color:#fff,stroke:#c62828
    style H fill:#e57373,color:#fff,stroke:#c62828
    style F fill:#81c784,color:#fff,stroke:#2e7d32
```

</details>

By narrowing down the number of individuals subjected to true evaluation, the overall evaluation cost is significantly reduced.

## Why saealib

When using EAs in Python, the standard choice for multi-objective search is **pymoo**.
Surrogate-assisted optimization specialized for expensive evaluations has been handled by **pysamoo**, a sister project of pymoo.
Neither library treats the decision of "which candidate solutions receive an expensive true evaluation" as a swappable component.
pymoo always evaluates generated candidates with the true function, and pysamoo writes this decision directly into each algorithm class.

saealib factors this decision out into **OptimizationStrategy**, a first-class swappable component.
The four built-in strategies — individual-based, generation-based, pre-selection, and direct — are all implementations of this abstraction, and if you need your own decision criteria, you can swap it out simply by subclassing `OptimizationStrategy`.
On top of this, saealib separates **Surrogate**, which does only fitting and prediction, from **AcquisitionFunction**, which converts predictions into a score, with **SurrogateManager** mediating between the two.
Changing the surrogate implementation does not affect the acquisition function code, and vice versa.

| Comparison | saealib | pymoo | pysamoo |
|---|---|---|---|
| Model management strategy is a swappable component | Yes | No (always evaluates all candidates) | Hardcoded per algorithm class |
| Swapping components at runtime via typed events | Yes | "Not intended for customizing algorithms" ([official docs](https://pymoo.org/interface/callback.html)) | No |
| Separation of surrogate and acquisition function | Yes | No (delegated to pysamoo) | Partial |

This swappability is underpinned by a design in which `Algorithm`/`OptimizationStrategy`/`Surrogate`/`AcquisitionFunction`/`SurrogateManager` all have abstract bases, and can be swapped at construction time via `Optimizer.set_*()` (see the [component overview](../components/index.md) for details).
For lighter changes that don't warrant subclassing, three lightweight mechanisms are also available: adding hooks via `with_post`/`with_post_fit`, rearranging stages via `Pipeline`/`Stage`, and observing and swapping components at runtime via `CallbackManager` (see the [extension guidelines](../components/extension_guidelines.md) for details).

Candidate generation and population updates are split into two procedures, **Ask-Tell**, and the decision of which of the candidates in between receive true evaluation is delegated to `OptimizationStrategy.step()` (see [Algorithm](../components/algorithm.md) for details).
This separation makes it possible to swap out only the evaluation strategy while leaving the search algorithm itself unchanged.

The pipeline assembled this way has two entry points: `minimize()`/`maximize()` is a boilerplate-free high-level API, while the `Optimizer` builder and `.iterate()` generator are a low-level API for research use cases that need per-generation inspection or custom loop control.
Both are built on the same pipeline, and the low-level API has no functionality of its own that the high-level API lacks (see the [component overview](../components/index.md) for how it's assembled).

Across the entire pipeline, scores are consistently unified under the convention that "higher is better" (see [Problem](../components/problem.md) for details).

The current trade-offs are also worth noting.
The built-in algorithms saealib provides are only GA and PSO, and it doesn't cover as many algorithms as pymoo or PlatEMO.
This is a trade-off from focusing the design on the surrogate and strategy layers.
