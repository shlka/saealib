---
primary_layer: layer1
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

`Problem` は `space` に `SearchSpace` を受け取り、Genomeの表現と評価入力の変換を管理できます。
詳細は [Problem](../concepts/problem_and_ranking/problem.md) と [探索空間（SearchSpace）](../concepts/problem_and_ranking/search_space.md) を参照してください。
