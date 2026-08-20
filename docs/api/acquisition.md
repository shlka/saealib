---
primary_layer: layer2
related_layers: [layer3]
---

# Acquisition Functions

## Base

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.AcquisitionFunction
```

## Implementations

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.ExpectedImprovement
   saealib.BatchExpectedImprovement
   saealib.LowerConfidenceBound
   saealib.MaxUncertainty
   saealib.MeanPrediction
   saealib.CORSDistance
   saealib.ProbabilityOfFeasibility
   saealib.ProductOfFeasibility
   saealib.EHVIAcquisition
   saealib.ParEGOAcquisition
   saealib.SMSEGOAcquisition
```

```{eval-rst}
.. autofunction:: saealib.gp_ucb_beta_schedule
```

## CORSDistance

`CORSDistance` is the CORS distance-constrained predicted-mean acquisition
function {cite}`regis2005cors`.  It gives each candidate a scalar predicted-mean
score and replaces the score with `-np.inf` when the candidate is too close to
an already evaluated design vector.  As with every `saealib` acquisition,
higher scores are preferred.

The public constructor is:

```python
from saealib import CORSDistance

acquisition = CORSDistance(
    delta=None,
    search_pattern=(0.95, 0.25, 0.05, 0.03, 0.0),
    weights=None,
    direction=None,
)
```

### Constructor arguments

- **`delta`** (`float | None`, default `None`): the distance scale in the
  threshold `beta * delta`.  With `None`, `CORSDistance` computes the paper's
  iteration-specific $\Delta_i$ approximately from the current candidate pool
  on every score call:
  `max_candidate min_evaluated distance`.  A numeric value must be finite and
  greater than zero and preserves the legacy fixed-scale behavior.  A design
  space's diagonal length is not the paper's Eq. (2) $\Delta_i$ and may make a
  high-beta batch infeasible; use `None` unless a fixed scale is intentional.
- **`search_pattern`** (`Sequence[float]`): the periodic beta sequence.  The
  default is the paper's SP1, `(0.95, 0.25, 0.05, 0.03, 0.0)`.  The
  implementation accepts any non-empty sequence whose values are finite and
  in `[0, 1]`.  It does not require the paper's trailing zero or non-increasing
  order, so `(1.0,)` is valid pure exploration.
- **`weights`** (`np.ndarray | None`): optional magnitude-aware weights for
  scalarizing multi-objective predicted means.  The shape is `(n_obj,)`.
- **`direction`** (`np.ndarray | None`): optional per-objective signs (`+1`
  for maximization and `-1` for minimization).  Direction-only scalarization
  takes precedence over `weights`; when unset, `problem.direction` is normally
  injected at run start.

### Prepared reference and return value

`CORSDistance` resolves the beta entry using
`ctx.decision_count % len(search_pattern)` (zero-based) in
`prepare(archive, ctx)`.  The normal call is through
`AcquisitionFunction.evaluate(..., ctx=...)`; calling `score()` directly
requires the prepared reference returned by `prepare()`, and `prediction.x`
must contain the candidate design vectors aligned row-for-row with
`prediction.value`.

`score(prediction, reference) -> np.ndarray` returns one `float` score per
candidate, with shape `(n_samples,)`.  Candidates violating

$$
\min_j\lVert x-x_j\rVert \geq \beta_i\Delta_i
$$

receive `-np.inf`; feasible candidates retain their scalarized predicted-mean
score.  With the default `delta=None`, $\Delta_i$ is the candidate-pool
maximin distance approximation.  All candidates evaluated in one prepared
batch share the same beta.  The source-faithful configuration uses one
true-evaluated candidate per decision, for example
`PreSelectionStrategy(..., n_select=1)` or `TopKEvaluation(1)`.
`CORSDistance` also supports multiple distinct candidates per `EvaluationPlan`.
That extension applies the one beta to the whole plan and does not reproduce the
source's sequential CORS procedure.

`CORSDistance.requires_sequential_decisions` is compiler metadata for this
semantic distinction.  The compiler emits the `cors_nonsequential_evaluation`
warning when a batch or overlapping asynchronous configuration is statically
visible.  A custom planner is checked at runtime by unique candidate ID, and a
runtime warning is emitted once per optimizer runtime when it selects multiple
distinct candidates or when distinct asynchronous decisions actually overlap.
Repeated evaluations of one candidate do not trigger the batch warning.
Configuring an asynchronous scheduler alone does not trigger a warning;
`max_pending=1` is the source-faithful boundary.

The beta phase starts at `search_pattern[0]` when `ctx.decision_count == 0` and
`CORSDistance` has been used continuously from the beginning of the run.  If a
component is replaced during a run and `CORSDistance` is introduced later, the
phase starts at the current `decision_count`.

```{note}
The paper advances beta after each single costly evaluation.  The runtime
advances `decision_count` once per confirmed `EvaluationPlan`.  A batch plan
therefore shares one beta, while a one-candidate plan corresponds to one source
iteration.  See [CORS-RBF](../algorithms/rbf_cors.md) for the full
paper-to-runtime comparison, including the candidate-pool approximation of
$\Delta_i$.
```
