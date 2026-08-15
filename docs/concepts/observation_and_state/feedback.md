---
primary_layer: layer4
related_layers: [layer2, layer3]
page_type: concept
---

# Feedback

Feedback is the contract that passes observations associated with candidates in a form the Algorithm can use.
It matches records by subject, candidate ID, proposal relation, sequence, status, source, and completion semantics rather than by row position.
Partial, out-of-order, and repeated observations for the same candidate are valid inputs.

## Feedback's role

Feedback passes observations associated with candidates in a form the Algorithm can use. It does not generate candidate representations or recalculate whether observations are correct.

## Proposals and observations

`ProposalBatch` collects candidates, relations among candidates, and the Feedback quantities they require.
`ObservationBatch` contains an observation schema and records, representing quantities such as objectives, constraints, features, and costs together with their source and status.
When the quantities requested by a proposal and provided by an observation batch are available, the proposal can proceed to later Feedback processing.

## Feedback on the graph-native path

`FeedbackBatch` is the observation batch delivered for one proposal.
`ObservationBatch` holds the subject, quantity, status, and source of each observation record, while `FeedbackBatch` adds the proposal ID, channel, sequence, and final flag.
The consumer declares ordering, completion, multiplicity, and grouping in `FeedbackContract`.
`FeedbackBuilder` is the boundary that projects observations into the dense `FeedbackResult` form.

## FeedbackResult on the Stage compatibility path

`FeedbackResult` is the dense compatibility data type passed to the Algorithm's `tell`.
In this type, `candidate_ids` are unique, and the `f`, `g`, `cv`, `evaluated_mask`, `source`, and artifacts arrays follow the same row-count and shape rules.
This constrains row consistency within `FeedbackResult`; it does not require row order to match an external `ObservationBatch`.

## What Feedback holds

| Type | Information held | Owning boundary |
|---|---|---|
| `ProposalBatch` | Proposal ID, candidates, relations among candidates, Feedback requirements, and metadata | Proposal side |
| `ObservationBatch` | Observation schema and records | Evaluation or observation side |
| `FeedbackBatch` | Proposal ID, observations, channel, completion status, and sequence | Delivery side |
| `FeedbackResult` | Candidate IDs, objective values, constraint values, evaluated mask, and value sources | Algorithm's tell boundary |

Each `FeedbackResult` row is identified by candidate ID, with `evaluated_mask` and `source` kept on the same row.
This lets later Algorithms and training data distinguish rows filled by predictions from rows directly evaluated.

## True, predicted, and mixed Feedback

`TrueOnlyFeedback` returns only rows completed by true evaluation.
`PredictedFeedback` returns the objective prediction channel for all candidates and supports an approximate update when true evaluation is unavailable.
`MixedFeedback` gives priority to true evaluation and fills the remaining rows with objective predictions.

These three are policies for selecting available observations and placing them in one Feedback result, not functions that decide which value is correct.
Sources such as `true`, `surrogate`, `human`, and `simulator` remain on the observation side, so later contracts and training data can validate usage conditions.

## Invariants

Feedback is responsible for matching subject, candidate ID, proposal relation, sequence, status, source, and completion semantics.
It does not generate candidate representations or recalculate whether observations are correct.
Do not prohibit repeated observations for the same candidate ID as a general Feedback invariant; distinguish quantities requested by proposals from those provided by observations, and preserve each value's source.

`ProposalBatch` and candidates are created when a proposal is made, while `ObservationBatch` is created when evaluation, prediction, or an external observation completes.
`FeedbackBuilder` matches them immediately before the next Algorithm uses them, and the Runtime passes the result to the next state boundary after collecting asynchronous evaluation.

## Extending Feedback

The Compiler verifies Feedback-related `PortContract`, schema bindings, required services, and lifecycle Feedback.
The Runtime checks candidate-ID matching, waits for incomplete observations, and prevents the same result from being applied twice.
An observation whose subject or proposal relation cannot be resolved, a delivery that violates declared ordering or completion contracts, missing requested quantities, or a choice that overwrites true evaluation with a prediction becomes a diagnostic or Feedback construction error.

When adding a Feedback policy, extend the Builder boundary while preserving observation sources and the evaluated mask.
For proposal or observation contract changes, see [Specs](../../framework/specs.md) and [Compiler](../../framework/compiler.md); for the asynchronous application boundary, see [Runtime](../../framework/runtime.md).
See the [API reference](../../api/index.md) for public import paths.

## Related components

- [Algorithm](../search_algorithms/algorithm.md): Consumes Feedback with `tell()`.
- [Evaluator](../execution_and_evaluation/evaluation.md): Produces evaluation results.
- [OptimizationState](optimization_state.md): Holds execution state.

## References

- {py:class}`saealib.FeedbackResult`
- {py:class}`saealib.FeedbackBuilder`
- {py:class}`saealib.core.StateView`
