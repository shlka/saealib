---
primary_layer: layer4
related_layers: [layer3]
page_type: reference
---

# Core API

The Core API is the public facade for component contracts, graphs, state, and compiled plans.
For ordinary extensions, import names from `saealib.core` rather than referring directly to implementation-module paths.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.core.Component
   saealib.core.ComponentContract
   saealib.core.PartSpec
   saealib.core.AssumptionSet
   saealib.core.DataSpec
   saealib.core.PortContract
   saealib.core.PortSpec
   saealib.core.StateContract
   saealib.core.LifecycleContract
   saealib.core.ExecutionContract
   saealib.core.ComponentGraph
   saealib.core.StructuredGraph
   saealib.core.StructuredRegion
   saealib.core.CompilationRule
   saealib.core.StateStore
   saealib.core.StateView
   saealib.core.StatePatch
   saealib.core.ExecutablePlan
   saealib.core.ExecutionRuntime
```

The public import policy for `Compiler` is still being organized.
This page therefore does not declare either `saealib.core` or `saealib.core.compiler` as the standard import path.
For code that uses the Compiler, check the target release's public API and release notes.

Graph composition values are available from `saealib.core.compiler`.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.core.compiler.ComponentNode
   saealib.core.compiler.NodeRef
   saealib.core.compiler.DataEdge
   saealib.core.compiler.ControlEdge
   saealib.core.compiler.StateBinding
   saealib.core.compiler.Diagnostic
   saealib.core.compiler.ContractPath
```
