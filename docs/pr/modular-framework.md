# Modular framework

## Motivation

The framework needs a stable way to describe optimization components without
making the execution engine depend on every concrete algorithm, operator,
surrogate, or problem implementation. The modular architecture makes those
dependencies explicit: components publish contracts, a compiler checks and
resolves a graph, and a runtime executes the resulting plan against versioned
state. This keeps composition inspectable, gives configuration errors a useful
location and remedy, and leaves room for synchronous, asynchronous, and
profile-specific execution policies.

The public layering is intentionally small. The root `saealib` namespace remains
the convenience facade for established user-facing objects. Domain namespaces
such as `saealib.population`, `saealib.problem`, `saealib.algorithms`, and
`saealib.surrogate` own domain implementations. `saealib.core` owns the
framework vocabulary and compiler boundary; `saealib.execution` owns
evaluators, initializers, schedulers, and runtime registrations.

## Architecture overview

An optimization run has four logical boundaries:

1. Components provide a `ComponentContract` describing ports, held parts,
   lifecycle behavior, state access, runtime capabilities, and assumptions.
2. A `ComponentGraph` connects component nodes with data edges, control edges,
   state bindings, and entry points. A `StructuredGraph` additionally retains
   nested sequences, repeats, loops, and branches as explicit regions.
3. The compiler verifies the graph and applies registered resolution rules to
   produce an `ExecutablePlan` with resolved services and inserted adapters.
4. A runtime executes the plan, applies returned state patches, dispatches
   events, and handles commands such as termination, checkpointing, or
   recompilation.

The design separates description from execution. Contracts and graph values
are immutable snapshots at compilation time, while concrete components retain
their normal domain behavior. This allows diagnostics to be deterministic and
prevents a later service-resolution pass from silently changing the contract
being checked.

## Component/contract model

`saealib.core.Component` is the minimal extension protocol: it supplies a
`contract()` method returning `saealib.core.ComponentContract`. A contract
contains named `PortContract` values, optional named `PartSpec` subcomponents,
`LifecycleContract`, `StateContract`, `ExecutionContract`, and an
`AssumptionSet`. Ports describe the roles a component consumes or provides;
parts make constructor-held components visible to composition and diagnostics.

Contracts are declarative rather than a second execution interface. They state
what a component needs and offers: state keys it reads, writes, or exports;
events and feedback policy; required and offered runtime capabilities; and
schema or representation assumptions. A graph node caches one contract
snapshot for compilation. Verification can therefore report incompatible
ports, missing state, invalid lifecycle combinations, or unsupported runtime
capabilities before execution begins.

The framework core does not import concrete feature packages as a general rule.
The few retained implementation bridges are deliberate: state storage and
patching still carry the existing population values, and the graph builder
adapts the established `Stage`/`Pipeline` model. These bridges preserve the
working public API while the contract vocabulary remains independent of domain
implementations.

## Graph/compiler model

`ComponentGraph` is a self-contained description of nodes and relationships.
`DataEdge` connects a source port to a target port, `ControlEdge` expresses
ordering, and `StateBinding` maps a node role to a typed `StateKey`. Entry
points identify where execution may begin. `StructuredGraph` retains the
source-order operation tree and composes each region's state effect without
introducing graph cycles for repetition. Graph well-formedness checks catch
unknown endpoints, duplicate identities, missing entry points, and other
structural errors.

`saealib.core.compiler.Compiler` runs rule namespaces in two conceptual
phases. Verification rules observe the graph and return diagnostics. Resolution
rules may propose a graph rewrite, but must claim the canonical locations they
change; conflicting claims are diagnosed instead of being applied implicitly.
Resolution can bind services and insert adapters, including representation and
feedback adapters, while preserving the node contract snapshot. The result is
an immutable `ExecutablePlan` containing the resolved graph, diagnostics, and
adapter insertions. Compilation is performed at the run boundary and the
resulting plan is retained by the runtime until configuration requires a new
one.

The compiler also exposes `Compiler.compile_pipeline()`. It lowers the
structured DSL to a `StructuredGraph` and then applies the same resolution and
verification rules as direct graph construction. The older graph builder still
supports stage-based strategies during migration, but that compatibility path
is not a structured runtime execution target.

`Pipeline` is the high-level structured notation. `Pipeline` and nested
pipelines lower to sequence regions; `Repeat`, `Loop`, and `Branch` lower to
explicit region nodes. A loop condition is a component-like predicate with a
declared state contract, and a region effect is the composition of the effects
of its body and condition. Runtime region frames retain the next operation and
iteration, allowing a blocked or running component to resume without restarting
the enclosing region.

For structured pipelines, the compiler resolves each required input against
compatible outputs on control-ordered upstream components. A unique match is
materialized as a `DataEdge`; unresolved and ambiguous inputs are diagnostics.
Explicit graph construction remains available when a connection needs a
specific producer.

## State/runtime model

`saealib.core.StateStore` owns versioned values keyed by typed `StateKey`
objects. A component receives a `StateView` limited to the keys declared by its
contract. It cannot obtain undeclared state through the view, and the store
uses move semantics: applying a `StatePatch` consumes the current store and
returns the next generation. Patches support ordinary writes, deletes, and
specialized population row updates while keeping domain validation in the
population implementation.

Graph-native components execute against a restricted `StateView` and return a
`StatePatch` or a `NodeResult`. A `NodeResult` may also contain events,
runtime commands, and a status such as completed, blocked, failed, or
recompile-required. The runtime applies patches in order, dispatches events,
and interprets commands. The view's context is a read-only `RuntimeContext`
facade; it does not expose the complete `OptimizationState` or arbitrary state
mutation methods. `saealib.core.ExecutionRuntime` defines this plan and result
vocabulary; concrete lifecycle implementations and evaluator services are
provided through `saealib.execution`.

The existing optimizer environment remains supported. Stage nodes execute with
the established optimization state through the sequential compatibility plan,
while structured graph-native nodes use the restricted core view. A
`StructuredPlan` rejects stage adapters and `execute(OptimizationState)` nodes,
so the two execution contracts cannot be mixed accidentally.

The asynchronous provider preserves a structured plan and executes its regions
with the same resumable frames as the synchronous provider. Ordinary leaves use
their synchronous execution boundary; leaves with an asynchronous driver use
the scheduler seam, and a pending evaluation keeps the current operation
position until the next poll. A required asynchronous capability without a
driver is rejected rather than silently downgraded.

Structured plans refuse `RequestRecompile` while a region frame is active and
refresh the plan only at a safe boundary with no frames. Sequential plans retain
their existing step-boundary recompilation path.

## Proposal/observation/feedback model

Candidate generation is represented by `ProposalBatch`, identified proposals,
candidate identity, and optional `ProposalRelations`. A proposal can declare a
`FeedbackRequirement` describing required quantities, accepted observation
sources, minimum fidelity, and whether completion must be a complete batch or
may be partial. This makes the ask/tell boundary explicit instead of encoding
delivery assumptions in a particular algorithm or scheduler.

Observations are typed records and batches. They identify a subject, quantity,
source (`true`, `surrogate`, `human`, simulator, or imputed), status, and
portable value. Quantity kinds cover objectives, constraints, constraint
violations, features, behavior descriptors, and evaluation cost. Stable
vocabularies and schemas let policies consume true and predicted information
through the same model while retaining provenance and completion state.

`FeedbackBatch` and `FeedbackContract` define delivery grouping, ordering,
multiplicity, and completion semantics. Feedback policies such as true-only,
predicted, or mixed feedback select observations for a consumer; the compiler
and runtime then connect the policy to the component's declared requirements.
Asynchronous evaluation uses the same proposal and observation identities, so
out-of-order updates and partial feedback are explicit capabilities rather than
special cases in the main loop.

`CandidatePopulation`, imported from `saealib.core.contracts`, is the minimal
candidate collection contract used by `ProposalBatch`: it requires only length
and row extraction, so the proposal contract does not name a concrete
`Population` implementation.

## Representation/profile model

Representation is a contract-level schema variable, not a dependency on one
algorithm family. `RepresentationSpec` names a registered
`RepresentationKind` and supplies named `ParameterSpec` bindings. Core
unification uses the existing schema binding carriers (`Var`, `Fixed`,
`Contained`, and `Product`); a representation kind may provide custom
containment rules for parameters such as sequence alphabets or lengths.

Profiles supply domain-specific registrations and adapters at the edge of the
framework. The shipped vector profile registers the standard vector
representation and its lossless conversion behavior. This keeps generic
compiler rules concerned with nominal representation contracts while profile
code owns concrete genome codecs, operators, and conversions.

## Extension API

To add a graph-native component, implement `contract()` and an `execute(view)`
method. Declare every consumed or produced port and state key, expose held
components as `PartSpec` entries when they are part of composition, and return
`StatePatch` or `NodeResult` values. Use typed `StateKey` values and portable
observation payloads so checkpointing and profile adapters can reason about the
component.

The standard contract-building vocabulary is available without deep imports:

```python
from saealib.core import (
    AssumptionSet,
    ComponentContract,
    ExecutionContract,
    LifecycleContract,
    PartSpec,
    PortContract,
    StateContract,
)
```

The library owns compiler rule registration and the compiler engine. Extension
authors should express new behavior through component contracts, graph
vocabulary, and the supported facade types; compiler internals such as
`RuleContext`, `RuleRegistry`, and `saealib.core.compiler.compiler` are not
canonical extension imports. When an integration needs a new compiler rule or
adapter, it should first establish the corresponding facade contract and
registration surface.

For existing domain components, the established extension points remain
available: implement the relevant domain base class or protocol, use the
root/domain namespace imports, and use `saealib.execution` for evaluator or
initializer services. Existing `Stage` subclasses and `Pipeline` composition
remain valid; custom stages are discovered through the retained graph-builder
bridge and may progressively expose richer contracts.

For runtime providers, use the canonical surface
`from saealib.execution import RuntimeRegistry, RuntimeRegistration,
create_runtime`. The beta `RuntimeFactory` and `default_runtime_registry`
exports remain available as compatibility and advanced customization hooks;
`saealib.execution.runtime` is not the canonical import path.
Asynchronous provider boundaries report nonblocking poll progress through
`saealib.execution.PollResult` rather than relying on state-object identity.

## Compatibility/breaking changes

The root facade and established domain namespaces remain supported, while new
framework vocabulary should be imported from `saealib.core` and execution
services from `saealib.execution`. The public contract and state types are the
canonical boundary; concrete implementation modules are not interchangeable
facades.

Checkpoint support is retained through `CheckpointCallback` and the existing
optimizer checkpoint/resume behavior. `Pipeline` is the structured DSL for new
strategy descriptions; `Stage` remains available through the sequential
compatibility bridge.

The beta `OptimizationContext` name, `EvaluationRequest.x` view and `x=`
constructor input, and legacy Population constructor/column mappings remain
available where they protect existing Python users. They are compatibility
bridges, not the preferred vocabulary for new code.

The unreleased `get_readonly_array` seam is removed. The unreleased
`_deprecated` module is removed, and the scheduler's private wrapper is not a
supported compatibility surface. Consumers must use the public population
read-only value APIs, the contract-limited `StateView`, and public scheduler or
evaluator interfaces in `saealib.execution`.

## Testing

The test suite covers contract validation and diagnostics, graph well-formedness
and compilation, compiler rule claims and plan retention, state-view read
boundaries and patch application, proposal/observation/feedback validation,
representation unification, execution adapters, asynchronous scheduling,
checkpoint round trips, and compatibility imports. Architecture tests enforce
the core import boundary and verify that candidate identity remains identical
across canonical and compatibility exports. The documented checks are the
repository's Python test suite; this narrative does not make claims about
unrun Python versions.

## Migration notes

Existing users can continue to construct optimizers through the root facade.
For new strategy descriptions, use `Pipeline` and compile it through
`Compiler.compile_pipeline()`. Move framework imports to `saealib.core`
(`Component`, contracts, graph/compiler and state vocabulary) and execution
imports to `saealib.execution` (evaluators, initializers, schedulers, and
runtime factories). Keep domain-specific imports in their `saealib` domain
namespaces.

When introducing a component incrementally, keep its existing `Stage` wrapper
and add a contract first. Let the graph builder expose it through the retained
bridge, then replace the wrapper with a graph-native `execute(StateView)` node
when its state and lifecycle semantics are ready. Replace direct array seams
with declared state keys and patches, and replace scheduler-private access with
the public evaluator/scheduler contract. Checkpoint files remain managed by the
existing checkpoint API; applications should continue to treat checkpoint
schema/version validation as part of resume handling.
