# Extension Guidelines

`Algorithm`/`OptimizationStrategy`/`Surrogate`/`AcquisitionFunction`/`SurrogateManager` all have abstract bases, and can be swapped out wholesale via `Optimizer.set_*()`.
This is the standard approach — use this path when you need a custom search algorithm or surrogate.
On the other hand, four lightweight mechanisms are available for cases where a change doesn't warrant replacing an existing component entirely.

## with_post / with_post_fit

[Crossover](crossover.md)/[Mutation](mutation.md)'s `with_post(fn)` and [Surrogate](surrogate.md)'s `with_post_fit(fn)` add post-processing to an existing instance without subclassing.
They don't modify the original instance — they return a copy with the hook added.

Typical uses are adding a repair function to `Crossover`/`Mutation`, or post-fit processing on `Surrogate`.
Each component page's "Extension hooks" section has concrete examples per component.

## Pipeline / Stage

`OptimizationStrategy`'s internal generation loop is structured as a sequence of units called [Stage](stage.md), executed in order by `Pipeline`.
`pipeline.replace("name", stage)` replaces a specific stage with a different one, and `pipeline.find("name", recursive=False)` looks up a stage by `name`.

```python
from saealib import Pipeline, Stage
from saealib.stages import CountGenerationStage


class DoubleCountStage(Stage):
    name = "count_generation"

    def execute(self, state):
        return state.replace(gen=state.gen + 2)


pipeline = Pipeline([CountGenerationStage(), ...])
pipeline.replace("count_generation", DoubleCountStage())
```

See [Stage](stage.md) for the contract each Stage satisfies, the list of 11 built-in stages, and how to implement a custom Stage.
This page covers the operation of "rearranging and replacing stages" itself.

## CallbackManager

[CallbackManager](callbacks.md) is an observation mechanism that calls handlers when events fire.
Use `cbmanager.register/unregister/replace` to change the default pipeline's handlers at runtime.

The `candidates` field carried by `PostCrossoverEvent`/`PostMutationEvent`/`PostAskEvent` is for observation only; reassigning it inside a handler has no effect on the actual candidate array.
The distinction is: use `with_post` if you want to actually swap out the candidate array, and CallbackManager if you only need observation, logging, or conditional branching decisions.
See [CallbackManager](callbacks.md)'s "The candidates field is for observation only" section for details.

The `Optimizer` instance itself isn't passed as a handler argument.
To swap a component at runtime, capture the `Optimizer` instance directly in the handler's closure and reassign `optimizer.algorithm`/`strategy`/`surrogate_manager`/`termination`.
Because each Strategy rebuilds the pipeline on every call to `step()`, this kind of swap reliably takes effect from the next generation (or the next iteration of `iterate()`) onward.

## Registry

`saealib.registry` is a mechanism for constructing an actual instance from a name (string) or a spec (`{"type": "Name", "params": {...}}`).
Where `with_post`/`Pipeline`-`Stage`/`CallbackManager` are mechanisms for "changing behavior at runtime," Registry serves a different purpose: "assembling components from strings or a config file."
Use it in situations that don't directly import classes, such as config-driven construction via preset YAML (`Optimizer.set_preset()`) or checkpoint resumption.

**`register(name=None)`** (decorator): Registers a class or function with the registry.
Adding `@register()` to a custom `Algorithm`/`Surrogate` subclass, etc., lets it be referenced by a short name just like a built-in component.

```python
from saealib import register
from saealib.surrogate.base import Surrogate


@register()
class MyCustomSurrogate(Surrogate):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    ...
```

`get`/`build`/`to_spec` are not exposed from saealib's top level; import them directly from `saealib.registry`.

**`get(name)`**: Resolves a registered name, or, if unregistered, a dotted path in `"module.submodule.ClassName"` form.

**`build(spec)`**: Recursively builds a spec into an actual instance.
If a value inside the spec is itself a nested spec, it is built recursively.
The form `{"callable": "dotted.path"}` resolves a function or built-in function itself (without calling it).

**`to_spec(obj)`**: The inverse of `build()`.
It reflects the constructor signature, reads same-named attributes, and recursively serializes them into a spec.
Classes with a `_registry_spec` attribute (such as `TerminationCondition`) don't use this generic reflection — they return that attribute directly.
This is the path `Optimizer.save_preset()` uses.

```python
from saealib.registry import build, get, to_spec

obj = build({"type": "MyCustomSurrogate", "params": {"alpha": 2.0}})
get("MyCustomSurrogate")  # -> the MyCustomSurrogate class
to_spec(obj)  # -> {"type": "MyCustomSurrogate", "params": {"alpha": 2.0}}
```

Several component pages have a note saying "class X is not `@register()`ed" — this only matters if you resolve classes by name via the Registry.

## Which mechanism to use

| What you want to do | Mechanism to use |
|---|---|
| Just add post-processing to an existing operator or surrogate | `with_post` / `with_post_fit` |
| Change the order of stages itself | `Pipeline` / `Stage` |
| External observation, logging, conditional swapping | `CallbackManager` |
| Assemble from a config file or preset | `Registry` |

## Related components

- [Crossover](crossover.md) / [Mutation](mutation.md) / [Surrogate](surrogate.md): Components with `with_post`-style hooks
- [Stage](stage.md): The contract of each stage that `Pipeline` combines
- [CallbackManager](callbacks.md): The event list and observation mechanism
- [strategies](strategies.md): When the pipeline is rebuilt

## References

- {py:func}`saealib.register`
