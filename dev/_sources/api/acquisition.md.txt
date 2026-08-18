---
primary_layer: layer2
related_layers: [layer3]
---

# Acquisition Functions

## Base

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.AcquisitionFunction
```

## Implementations

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.ExpectedImprovement
   saealib.BatchExpectedImprovement
   saealib.LowerConfidenceBound
   saealib.MaxUncertainty
   saealib.MeanPrediction
   saealib.ProbabilityOfFeasibility
   saealib.ProductOfFeasibility
   saealib.EHVIAcquisition
   saealib.ParEGOAcquisition
   saealib.SMSEGOAcquisition
```

```{eval-rst}
.. autofunction:: saealib.gp_ucb_beta_schedule
```
