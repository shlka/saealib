# Problem

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Problem
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

```{note}
`saealib.Constraint` は `saealib.InequalityConstraint` の非推奨エイリアスです。新しいコードでは `InequalityConstraint` を使用してください。
```
