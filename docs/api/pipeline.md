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
surface remains available only to the compatibility graph builder and is not
accepted by a `StructuredPlan`.

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
