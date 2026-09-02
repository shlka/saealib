---
primary_layer: layer2
related_layers: [layer3]
---

# 演算子

## Crossover

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Crossover
   saealib.CrossoverBLXAlpha
   saealib.CrossoverSBX
   saealib.CrossoverUniform
   saealib.CrossoverOnePoint
   saealib.CrossoverTwoPoint
   saealib.CrossoverIntegerSBX
   saealib.CrossoverCategorical
```

## Mutation

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Mutation
   saealib.MutationUniform
   saealib.MutationPolynomial
   saealib.MutationGaussian
   saealib.MutationIntegerUniform
   saealib.MutationCategorical
```

## 選択

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.ParentSelection
   saealib.TournamentSelection
   saealib.SequentialSelection
   saealib.LinearRankSelection
   saealib.SurvivorSelection
   saealib.TruncationSelection
```

## 修復

```{eval-rst}
.. autofunction:: saealib.repair_clipping
```

## 重複排除

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.DuplicateElimination
```

## pymoo / 外部ライブラリアダプター

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.PymooCrossover
   saealib.PymooMutation
```

