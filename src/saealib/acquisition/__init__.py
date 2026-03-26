"""
Acquisition functions package.

Provides acquisition (infill criterion) functions for surrogate-assisted
optimization. These functions convert surrogate predictions into scalar
scores used to rank candidates for true evaluation.
"""

from saealib.acquisition.base import AcquisitionFunction
from saealib.acquisition.ei import ExpectedImprovement
from saealib.acquisition.lcb import LowerConfidenceBound
from saealib.acquisition.mean import MeanPrediction
from saealib.acquisition.pof import ProbabilityOfFeasibility
from saealib.acquisition.uncertainty import MaxUncertainty

__all__ = [
    "AcquisitionFunction",
    "ExpectedImprovement",
    "LowerConfidenceBound",
    "MeanPrediction",
    "MaxUncertainty",
    "ProbabilityOfFeasibility",
]
