---
primary_layer: layer2
related_layers: []
page_type: concept
---

# CORS-RBF (Constrained Optimization using Response Surfaces)

CORS-RBF is the radial-basis-function (RBF) implementation of **Constrained
Optimization using Response Surfaces (CORS)** proposed by Regis and Shoemaker
(2005) {cite}`regis2005cors`.  CORS is intended for expensive black-box
objectives: it fits a response surface to the evaluations already paid for and
uses a distance constraint to keep the next evaluation away from those points.
The distance requirement cycles between global exploration and local
surrogate-based search.

`saealib` implements the distance-constrained criterion as
`CORSDistance`.  The library's evolutionary algorithm supplies a finite
candidate pool, so `CORSDistance` ranks that pool rather than solving the
continuous CORS auxiliary problem directly.  The highest-scoring candidates
are then sent to the true objective according to the configured evaluation
strategy.

## Overview

Let $S_i$ be the set of points evaluated by the true objective before CORS
step $i$, and let $\hat f_i$ be the response surface fitted to those points.
The CORS auxiliary problem (CORS-AP) is

$$
\begin{aligned}
\operatorname{minimize}_{x \in \mathcal{D}}\quad & \hat f_i(x) \\
\text{subject to}\quad & \lVert x-x_j\rVert \geq \beta_i\Delta_i,
\quad x_j \in S_i.
\end{aligned}
$$

Here, $\beta_i \in [0,1]$ controls the exploration--exploitation trade-off.
A value near one favors points far from the evaluated set; zero removes the
additional distance constraint and leaves ordinary surrogate-mean search.
The source paper defines the scale as the distance of a domain-wide maximin
point from its closest evaluated point:

$$
\Delta_i = \max_{\tilde{x} \in \mathcal{D}}
           \min_{x_j \in S_i}\lVert \tilde{x}-x_j\rVert.
\tag{2}
$$

The constraint is the central part of CORS, not an optional post-processing
step.  Regis and Shoemaker prove a global-convergence result when the search
pattern contains a positive value infinitely often; the result is independent
of the response-surface model and of the initial evaluated points
{cite}`regis2005cors`.

`saealib` follows its general acquisition convention that a **higher score is
better**.  `CORSDistance` first scalarizes the predicted mean (using
`direction`, `weights`, or the first objective), then assigns `-np.inf` to
candidates that violate the current distance constraint.

## Search patterns and $\beta$

The paper uses a periodic sequence
$\langle\beta_1,\ldots,\beta_{N+1}\rangle$ and repeats it after each cycle.
Its general description uses a non-increasing sequence ending in zero, with
large values for global search and values close to zero for local search.  In
the Dixon--Szego benchmark experiments it reports:

- **SP1**: $\langle 0.95,\ 0.25,\ 0.05,\ 0.03,\ 0\rangle$;
- **SP2**: $\langle 0.9,\ 0.75,\ 0.25,\ 0.05,\ 0.03\rangle$.

These are experimental choices, not separate acquisition classes.  SP1 is
`CORSDistance`'s default and has length $L=5$.  The source also observes that
the ordering and terminal-zero conditions are heuristics: having at least one
non-zero entry is the condition used by its convergence corollary
{cite}`regis2005cors`.

For `CORSDistance`, the beta index is explicitly

```text
ctx.decision_count % len(search_pattern)
```

The index is zero-based, so `decision_count == 0` selects
`search_pattern[0]`.  This is deliberately different from the scheduled
parameter in `LowerConfidenceBound`, which uses
`ctx.decision_count + 1` as a one-based round number.  `prepare(archive, ctx)`
packages the evaluated design vectors and the selected beta into the
read-only reference consumed by `score()`; scoring does not mutate a private
cycle counter.

### Source-faithful decisions and batch extensions

The source advances one CORS step for **one true evaluation point**.  The
source-faithful `saealib` configuration therefore uses a candidate pool with
`PreSelectionStrategy(..., n_select=1)` or an explicit `TopKEvaluation(1)`.
Candidate generation may still produce a batch; only one candidate proceeds to
true evaluation for each decision.

`saealib` also supports planners that send multiple distinct candidates to the
true objective in one `EvaluationPlan`.  `CORSDistance` prepares one beta for
that decision, and the runtime applies the same beta to every candidate in the
plan.  This is a supported batch extension, not the source's sequential CORS
procedure.  The compiler emits a `cors_nonsequential_evaluation` warning when
this configuration is statically visible; a custom planner is checked at
runtime by counting distinct candidate IDs, and the warning is emitted once per
optimizer runtime.

Repeated evaluations of one candidate do not count as a multi-candidate batch,
because the runtime compares unique candidate IDs.  The runtime does not warn
merely because an asynchronous scheduler is configured.  `max_pending=1`
keeps the source-faithful sequential boundary.  A larger capacity receives a
compiler warning when statically configured, while a runtime warning for
asynchronous non-sequential behavior requires actual overlap between distinct
decisions.

The beta index is `ctx.decision_count % len(search_pattern)`.  A
`decision_count` that starts at zero and remains paired with `CORSDistance`
from the beginning of a run matches the source iteration index.  Replacing a
component with `CORSDistance` during a run starts its phase at the current
`decision_count`; it does not reconstruct the earlier source iterations.

## CORS-RBF procedure

The following pseudocode separates the paper's one-point procedure from the
candidate-pool approximation used by `saealib`.

```{prf:algorithm} CORS-RBF with a candidate pool
:label: alg-rbf-cors

**Inputs** objective function $f$, search domain $\mathcal{D}$, initial evaluated set $S_1$, RBF surrogate, search pattern, and evaluation budget
**Output** best solution found in the true-evaluation archive

1. Evaluate the initial set $S_1$ with $f$.
2. Fit the RBF surrogate $\hat f_i$ to the evaluated archive.
3. Generate a candidate pool $C_i$ with the configured evolutionary algorithm.
4. Prepare $\beta_i$ from the current decision count and compute the distance of each candidate in $C_i$ to the evaluated archive.
5. Compute the candidate-pool approximation $\widehat{\Delta}_i = \max_{x \in C_i} \min_{x_j \in S_i} \lVert x-x_j\rVert$ unless a fixed `delta` was supplied.
6. Score the predicted mean. Set the score to $-\infty$ for each candidate with minimum distance below $\beta_i\widehat{\Delta}_i$ (or below $\beta_i\,\texttt{delta}$ for the fixed-scale mode).
7. Send the highest-scoring candidate or batch of candidates to the true objective, add the observations to the archive, and repeat from step 2 until termination.
```

## Flowchart

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Sample initial population<br/>→ true evaluation<br/>(L1)"] --> ASK
    subgraph GEN["One generation (PreSelectionStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>Generate candidate pool"] --> SCORE["GlobalSurrogateManager<br/>Fit RBF and predict<br/>Prepare beta once<br/>Score with CORSDistance"]
        SCORE --> DIST["Compute candidate-pool<br/>maximin Delta_i<br/>and apply beta_i Delta_i"]
        DIST --> SORT["Select the top n_select=1<br/>candidate by score"]
        SORT --> EVAL["True evaluation of one candidate<br/>all rows in the decision share beta_i<br/>(L4)"]
        EVAL --> TELL["GA.tell()<br/>Update population"]
    end
    GEN --> TERM{"Evaluation budget<br/>reached?"}
    TERM -- "Not yet (L5)" --> ASK
    TERM -- "Reached" --> RESULT(["Best solution x*"])
```

## Configuration in `saealib`

| Role | `saealib` implementation | Corresponding CORS step |
|---|---|---|
| Search algorithm | `GA` (the crossover, mutation, parent-selection, and survivor-selection choices are library configuration, not part of CORS) | 3 |
| Surrogate model | `RBFSurrogate` (the paper's experiment uses a thin-plate-spline kernel with a linear polynomial term) | 2 |
| Acquisition function | `CORSDistance` (predicted-mean score plus the $\beta_i\Delta_i$ distance constraint) | 4--6 |
| Surrogate management | `GlobalSurrogateManager` (fits the RBF on the complete evaluated archive) | 2 |
| Evaluation strategy | `PreSelectionStrategy(n_candidates=20, n_select=1)` (ranks the candidate pool and evaluates one candidate) | 6--7 |

The following is a runnable smoke example.  It uses the paper's SP1 and its
thin-plate-spline-plus-linear-polynomial RBF configuration.  `delta=None` is
explicit to show the paper-inspired candidate-pool approximation; it is also
the `CORSDistance` default.  The 40-FE budget keeps a documentation example
quick (the final batch can make the observed `fe` slightly larger than the
limit); use a larger budget for an experiment rather than treating this as a
benchmark.

```python
import numpy as np

from saealib import (
    CORSDistance,
    GA,
    PreSelectionStrategy,
    Optimizer,
    Problem,
    RBFSurrogate,
    ThinPlateSplineKernel,
)
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.surrogate import GlobalSurrogateManager
from saealib.termination import Termination, max_fe


def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)


SP1 = (0.95, 0.25, 0.05, 0.03, 0.0)
problem = Problem(
    sphere,
    dim=5,
    lb=[-5.0] * 5,
    ub=[5.0] * 5,
    n_obj=1,
    direction=[-1],
)

algorithm = GA(
    CrossoverBLXAlpha(prob=0.7, alpha=0.4),
    MutationUniform(prob_var=0.3),
    SequentialSelection(),
    TruncationSelection(),
)
surrogate_manager = GlobalSurrogateManager(
    RBFSurrogate(kernel=ThinPlateSplineKernel(), polynomial_degree=1)
)
acquisition = CORSDistance(
    delta=None,
    search_pattern=SP1,
    direction=problem.direction,
)
strategy = PreSelectionStrategy(n_candidates=20, n_select=1)

opt = (
    Optimizer(problem, seed=7)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_acquisition(acquisition)
    .set_strategy(strategy)
    .set_termination(Termination(max_fe(40)))
)
ctx = opt.run()
```

The example is intentionally explicit about `direction`: with
`direction=[-1]`, a lower objective value produces a higher acquisition score.
If a Gaussian RBF is preferred, use `RBFSurrogate(kernel=GaussianKernel())`
and omit `polynomial_degree`; a Gaussian kernel does not require a polynomial
term.

## Differences from the source

### Candidate-pool approximation of CORS-AP

The paper solves CORS-AP over the whole feasible domain and selects one point
that minimizes the response surface under the distance constraint.  `saealib`
uses the candidate pool generated by `GA` and selects the highest-scoring
feasible rows.

The canonical configuration evaluates one true candidate per decision, so it
matches the source's sequential decision structure.  `PreSelectionStrategy`
with `n_select=1` and `TopKEvaluation(1)` express that configuration while
leaving candidate generation batched.

`saealib` also permits multiple distinct candidates per decision.  Every
candidate in that `EvaluationPlan` receives the one beta prepared for the
plan.  The compiler or runtime warning identifies this supported extension,
but the configuration does not reproduce the source's sequential procedure.
The source paper's sequential convergence discussion must not be transferred
to that extension without an additional argument.

The GA operators and the choice of `GlobalSurrogateManager` are also
`saealib` configuration choices.  They should not be read as part of the CORS
definition.

### Beta progression and asynchronous boundaries

The source's $i$ advances after each true evaluation point.  `saealib`'s
`CORSDistance` reads the runtime's `ctx.decision_count` and selects one
zero-based pattern entry per prepared decision.  All candidates in one plan
share that entry because `prepare()` runs once for the plan and `score()` is
read-only.

A run that starts with `decision_count == 0` and continues to use
`CORSDistance` keeps the source iteration phase.  Introducing the acquisition
after a component replacement starts from the current decision count, so the
runtime does not infer the earlier phase.

Asynchronous scheduling follows the same boundary.  `max_pending=1` waits for
the current decision before preparing the next one.  A scheduler by itself
produces no warning.  A capacity greater than one is statically diagnosed when
it can allow overlapping decisions, and the runtime diagnoses only an overlap
that actually occurs.

### $\Delta_i$: paper definition versus `delta`

Equation (2) in the paper is a domain-wide maximin distance.  An acquisition
function does not receive the full feasible domain, so `CORSDistance(delta=None)`
computes the following approximation from the current candidate pool
`prediction.x` on every score call:

$$
\widehat{\Delta}_i = \max_{x \in C_i}\min_{x_j \in S_i}
                         \lVert x-x_j\rVert.
$$

This is the same cover-point idea described by the paper's implementation
section, with the generated candidate pool serving as the cover set.  It can
underestimate the domain-wide value if the pool covers only a promising
subregion, but it has an important feasibility property: for a non-empty pool
and $\beta_i\leq 1$, a maximin candidate satisfies the threshold (equality is
allowed), so at least one candidate remains executable.

Earlier documentation examples used the design-box diagonal as a fixed
`delta`.  That is a distance scale, not the iteration-specific $\Delta_i$ in
Eq. (2).  With a diagonal-length `delta`, a high value such as
$\beta_i=0.95$ can put every candidate in the current pool below the threshold
(the impossible region discussed in the paper), leaving no executable row.
The default `delta=None` path fixes that failure mode by deriving
$\widehat{\Delta}_i$ from the candidates each time.  Passing a finite positive
numeric `delta` remains supported for legacy fixed-scale experiments, but the
caller is responsible for choosing a scale that leaves feasible candidates.

### Search-pattern validation

The paper's non-increasing order and terminal zero are useful heuristics, not
requirements of the convergence argument.  The implementation therefore
accepts exactly the following input contract:

- `search_pattern` is non-empty;
- every value is finite; and
- every value satisfies $0\leq\beta\leq1$.

It does **not** require a trailing zero or a non-increasing sequence.  For
example, `(1.0,)` is valid and expresses pure exploration, and an all-zero
pattern is also accepted (although it gives up the paper's exploration
property).  The constructor normalizes the values to a tuple of floats.

## Parameters and variants

**`delta`**: The fixed distance scale used in the threshold
`beta * delta`.  Leave it as `None` (the default) to compute the candidate-pool
maximin approximation of $\Delta_i$ on each score call.  A numeric value must
be finite and greater than zero and opts into the legacy fixed-scale behavior.

**`search_pattern`**: The periodic beta sequence.  The default is SP1,
`(0.95, 0.25, 0.05, 0.03, 0.0)`.  SP2 from the paper can be supplied as
`(0.9, 0.75, 0.25, 0.05, 0.03)`.

**`weights` and `direction`**: Multi-objective scalarization options inherited
from `MeanPrediction`.  `direction` uses per-objective signs (`+1` for
maximization and `-1` for minimization) and takes precedence over `weights`.
If neither is supplied, the first objective is used; an unset direction is
normally injected from `problem.direction` at run start.

**RBF kernel**: `RBFSurrogate(kernel=...)` requires an explicit `RBFKernel`.
Use `ThinPlateSplineKernel()` with `polynomial_degree=1` to match the paper's
RBF experiment, or use a strictly positive-definite kernel such as
`GaussianKernel()` without a polynomial term.

**Candidate coordinates**: `CORSDistance` needs `prediction.x`, the candidate
design vectors aligned with `prediction.value`.  The surrogate's `predict()`
must populate this field.  Call the normal `evaluate(..., ctx=...)` path (or
call `prepare()` with a real context) so the beta for the current decision is
resolved before `score()` runs.

## Related

- [References](../references.md): Full bibliographic details for Regis & Shoemaker (2005)
- [Surrogate](../concepts/surrogate_modeling/surrogate.md): RBF surrogates and kernels
- [AcquisitionFunction](../concepts/surrogate_modeling/acquisition_functions.md): Acquisition lifecycle and score convention
- [SurrogateManager](../concepts/surrogate_modeling/surrogate_manager.md): `GlobalSurrogateManager`
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md): `PreSelectionStrategy` for one-candidate decisions and `IndividualBasedStrategy` for ratio-based extensions
