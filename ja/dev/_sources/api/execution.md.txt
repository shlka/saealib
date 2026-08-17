---
primary_layer: layer4
related_layers: [layer3]
page_type: reference
---

# Execution API

The Execution API exposes evaluators, initializers, the asynchronous evaluation scheduler, and runtime registration mechanisms.
`Runner` is an internal implementation and is not included in this public listing.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.execution.AsyncEvaluationScheduler
   saealib.execution.PollResult
   saealib.execution.RuntimeFactory
   saealib.execution.RuntimeRegistration
   saealib.execution.RuntimeRegistry
   saealib.execution.create_runtime
   saealib.execution.default_runtime_registry
```

See [Evaluation](evaluation.md) and [Initialization](initialization.md) for details about evaluators, evaluation Requests, and initializers.
