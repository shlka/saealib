---
primary_layer: layer1
related_layers: [layer2, layer3]
---

# 問題

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

`Problem`は`space`を通じて`SearchSpace`を受け取り、Genome表現と評価入力への変換を管理できます。
詳細は[Problem](../concepts/problem_and_ranking/problem.md)と[SearchSpace](../concepts/problem_and_ranking/search_space.md)を参照してください。
