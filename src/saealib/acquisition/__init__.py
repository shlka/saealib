"""
Acquisition functions package.

Provides acquisition (infill criterion) functions for surrogate-assisted
optimization. These functions convert surrogate predictions into scalar
scores used to rank candidates for true evaluation.
"""

from saealib.acquisition.archive_based import (
    InverseDensityAcquisition,
    MaximinDistanceAcquisition,
    NoveltyAcquisition,
)
from saealib.acquisition.base import (
    AcquisitionFunction,
    AcquisitionResult,
    CompositeAcquisition,
    PointwiseAcquisition,
)
from saealib.acquisition.batch import BatchExpectedImprovement
from saealib.acquisition.ehvi import EHVIAcquisition
from saealib.acquisition.ei import ExpectedImprovement
from saealib.acquisition.lcb import LowerConfidenceBound, gp_ucb_beta_schedule
from saealib.acquisition.mean import CORSDistance, CORSReference, MeanPrediction
from saealib.acquisition.parego import ParEGOAcquisition
from saealib.acquisition.pof import ProbabilityOfFeasibility, ProductOfFeasibility
from saealib.acquisition.smsego import SMSEGOAcquisition
from saealib.acquisition.uncertainty import MaxUncertainty
from saealib.acquisition.winrate import WinRateAcquisition

__all__ = [
    "AcquisitionFunction",
    "AcquisitionResult",
    "BatchExpectedImprovement",
    "CORSDistance",
    "CORSReference",
    "CompositeAcquisition",
    "EHVIAcquisition",
    "ExpectedImprovement",
    "InverseDensityAcquisition",
    "LowerConfidenceBound",
    "MaxUncertainty",
    "MaximinDistanceAcquisition",
    "MeanPrediction",
    "NoveltyAcquisition",
    "ParEGOAcquisition",
    "PointwiseAcquisition",
    "ProbabilityOfFeasibility",
    "ProductOfFeasibility",
    "SMSEGOAcquisition",
    "WinRateAcquisition",
    "gp_ucb_beta_schedule",
]
