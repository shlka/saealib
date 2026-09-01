---
primary_layer: layer1
related_layers: [layer2, layer3]
---

# Problem

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Problem
   saealib.PymooProblem
   saealib.InequalityConstraint
   saealib.EqualityConstraint
   saealib.ConstraintHandler
   saealib.StaticToleranceHandler
   saealib.EpsilonConstraintHandler
   saealib.GradientRepairHandler
```

```{eval-rst}
.. autofunction:: saealib.linear_epsilon_schedule
```

```{eval-rst}
.. autofunction:: saealib.exponential_epsilon_schedule
```

`Problem` accepts a `SearchSpace` through `space` and can manage Genome representations and conversion to evaluation inputs.
See [Problem](../concepts/problem_and_ranking/problem.md) and [SearchSpace](../concepts/problem_and_ranking/search_space.md) for details.
