---
primary_layer: cross
related_layers: [layer2, layer3, layer4]
page_type: concept
---

# Diagnostics and observation

Different kinds of problems and runtime information belong to different reporting mechanisms.
The following boundaries keep programming errors, user-actionable notices, runtime messages, structured data, and compiler results distinct.

| Purpose | Owner |
|---|---|
| Incorrect usage or an invalid configuration that indicates the developer should fix the code | Exception |
| Notice of a problem that the user should fix in code or configuration | `warnings.warn` |
| Diagnosis, progress, and numerical anomalies during execution | Standard `logging` with `saealib.*` loggers |
| Structured per-generation numerical records for analysis and visualization | History channels |
| Execution observation and extension points | `CallbackManager` |
| One-way handoff of Compiler analysis results | `Diagnostic` / `DiagnosticBag` |

## Runtime logging

saealib does not configure logging output handlers. It installs only a `NullHandler`.
Whether log messages are displayed therefore depends on the user's `logging` configuration.
See [Logging Progress](../../tutorials/logging.md) for examples of enabling levels and adding handlers.

`verbose=False` only prevents the progress callback handler from being registered.
It is independent of logging output configuration and does not install, remove, or reconfigure logging handlers.

Typical runtime messages include INFO entries for run start and finish, DEBUG entries for the resolved configuration and runtime activity, and WARNING entries from scheduler recovery or numerical issues.
The wording of logging messages and their fields is not guaranteed as a stable API; consumers should configure levels and logger namespaces rather than parse individual messages.

## Structured history and callbacks

Use History channels for structured numerical records collected across generations and intended as input to analysis or visualization.
This keeps data consumers from depending on human-readable log messages.

`CallbackManager` is the observation and extension boundary for execution events.
It can drive progress handlers, custom logging, and history collection, but it is not a substitute for exceptions or user-facing warnings.
See [CallbackManager](callbacks.md) for event registration and observation details.

## Compiler diagnostics

Compiler rules pass analysis results in one direction: they produce `Diagnostic` values, which are collected in a `DiagnosticBag` and returned with the compilation result.
The caller can then decide how to render or act on those diagnostics.
Compiler diagnostics are not runtime log messages and should not be routed through `logging`.
See [Compiler](../../framework/compiler.md) for the compilation and verification boundary.
