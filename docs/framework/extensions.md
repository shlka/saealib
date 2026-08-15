---
primary_layer: layer4
related_layers: [layer3]
page_type: guide
---

# Extend the framework

Extend the contracts of the framework that inspects and executes components rather than swapping an existing component.

## Extension targets

| Target | Role | Public API |
|---|---|---|
| Contract | Declare ports, state, lifecycle, and execution capabilities | `ComponentContract` |
| SearchSpace | Provide Genome representation, sampling, validation, and space-specific services | `SearchSpace` |
| Graph | Describe component, data, control, and state relationships | `ComponentGraph` |
| Compiler rule | Check contract compatibility and transformations | `CompilationRule` |
| Feedback | Associate candidate IDs with observations | `FeedbackBatch`, `FeedbackBuilder` |
| Runtime | Advance the plan and manage state patches and waits | `ExecutionRuntime` |

For each target, start with the [framework overview](index.md), then read [Component](component.md), [ComponentContract](contract.md), [Specs](specs.md), [Graph](graph.md), and [Compiler](compiler.md).
For SearchSpace, Feedback, and Runtime extension boundaries, see [SearchSpace](../concepts/problem_and_ranking/search_space.md), [Feedback](../concepts/observation_and_state/feedback.md), and [Runtime](runtime.md), respectively.

## Implementation principles

Do not let a component access state or services that its contract does not declare.
Return state updates as `StatePatch` values and delegate their application to the Runtime.
Register representation and semantic conversions as explicit adapters and do not bypass port-compatibility checks.
Keep compilation-time checks separate from execution-time processing.

## Public API boundary

Obtain the public vocabulary for framework extensions from the `saealib.core`, `saealib.space`, `saealib.execution`, and `saealib.policies` facades.
Do not present individual implementation modules or internal Runtime classes as extension APIs for general users.
Check the release-specific API reference for public import paths for the Compiler and Compiler rules.

## Verification checklist

For every new extension, verify at least the following:

- Contract inputs and outputs match actual Graph connections.
- The component reads and writes no StateKey other than those it declares.
- Each Observation subject resolves to a valid candidate or proposal relation and satisfies the consumer's declared ordering/completion contract. Verification does not depend on source position.
- State boundaries remain intact for synchronous evaluation, asynchronous evaluation, and checkpoint resumption.
- The public API for ordinary users does not depend on framework-internal types.
