# Surrogate Models

## Base

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Surrogate
   saealib.RegressionSurrogate
   saealib.ComparisonSurrogate
   saealib.SurrogatePrediction
```

## Surrogate Managers

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.SurrogateManager
   saealib.GlobalSurrogateManager
   saealib.LocalSurrogateManager
   saealib.CompositeSurrogateManager
   saealib.PairwiseSurrogateManager
   saealib.product_combine
   saealib.rank_weighted_combine
```

## RBF Surrogate

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.RBFSurrogate
```

```{eval-rst}
.. autofunction:: saealib.gaussian_kernel
```

## Scikit-learn / External Library Adapters

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.SklearnSurrogate
   saealib.SklearnGPRSurrogate
   saealib.SklearnRFRSurrogate
   saealib.SklearnSVMSurrogate
   saealib.SklearnNNSurrogate
   saealib.SklearnXGBSurrogate
   saealib.SklearnLGBMSurrogate
   saealib.SklearnClassificationSurrogate
   saealib.SklearnRFCClassificationSurrogate
   saealib.SklearnSVCClassificationSurrogate
   saealib.TorchSurrogate
```

## Per-Objective Surrogate

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.PerObjectiveSurrogate
```

## Training Sets

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.TrainingData
   saealib.TrainingSet
   saealib.ArchiveObjectiveSet
   saealib.ConstraintObjectiveSet
   saealib.FeasibilityClassificationSet
   saealib.KNNObjectiveSet
   saealib.KNNConstraintObjectiveSet
   saealib.LevelBasedSet
   saealib.PairwiseComparisonSet
   saealib.ReferencePointComparisonSet
   saealib.TopKBipartitionSet
```

## Archive-Based Managers

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.ArchiveBasedManager
   saealib.DensityManager
   saealib.NichingManager
   saealib.NoveltyManager
```

## Surrogate Switching

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.AccuracyBasedSurrogateSwitcher
   saealib.GenCtrlSwitcher
   saealib.ManagerSwitcher
   saealib.StrategySwitcher
```

## Accuracy Evaluation

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.AccuracyEvaluator
   saealib.HeldOutAccuracyEvaluator
   saealib.KFoldAccuracyEvaluator
   saealib.LOOAccuracyEvaluator
   saealib.SurrogateAccuracy
   saealib.SurrogateAccuracyMetric
   saealib.RMSE
   saealib.R2Score
   saealib.SpearmanCorrelation
```
