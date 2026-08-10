# Pipeline

`Pipeline` is a structured description of an algorithm.  It does not execute
an `OptimizationState`; lower it to a semantic graph and compile that graph
before handing it to a runtime.

```python
from saealib import Branch, Loop, Pipeline, Repeat
from saealib.core.compiler import Compiler

pipeline = Pipeline(
    name="generation",
    steps=[
        Repeat(
            Pipeline(steps=[ask, acquire, tell], name="surrogate_generation"),
            count=10,
            name="surrogate_generations",
        ),
        Loop(evaluation, until=budget_reached, name="evaluation_loop"),
        Branch(route, then=fast_path, else_=safe_path, name="route"),
    ],
)
plan = Compiler().compile_pipeline(pipeline)
```

Nested pipelines and control values are retained as named structured regions.
`Repeat` is a compile-time or fixed-count repetition; `Loop` and `Branch`
evaluate a condition through its declared `StateContract`.  Regions are not
lowered to ordinary graph cycles, so the compiler and runtime can inspect
their state effects and preserve resumable region frames.

Components placed in a structured pipeline must provide the graph-native
`contract()` and `execute(StateView)` boundary.  The older `Stage` execution
surface remains available only to the compatibility graph builder.  Lowering a
bare `Stage` fails with guidance to wrap it in `stage_component(stage)`.

During compilation, each required input port is matched against compatible
outputs on control-ordered upstream components.  One match creates a
`DataEdge`; no match or multiple matches produce a compiler diagnostic.  Use
an explicit graph when a pipeline needs to disambiguate a connection.

`stage_component(stage)` is a migration adapter.  State writes made through its
transaction proxy become `StatePatch` values, while mutable objects exposed as
services or context capabilities remain adapter-owned.  New graph-native
components should keep persistent mutable state behind declared state keys and
patches.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Stage
   saealib.Pipeline
   saealib.Repeat
   saealib.Loop
   saealib.Branch
```
