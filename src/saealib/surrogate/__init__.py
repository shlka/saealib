from saealib.surrogate.accuracy import (
    RMSE,
    AccuracyEvaluator,
    HeldOutAccuracyEvaluator,
    KFoldAccuracyEvaluator,
    LOOAccuracyEvaluator,
    R2Score,
    SpearmanCorrelation,
    SurrogateAccuracy,
    SurrogateAccuracyMetric,
)
from saealib.surrogate.archive_manager import (
    ArchiveBasedManager,
    DensityManager,
    NichingManager,
    NoveltyManager,
)
from saealib.surrogate.base import Surrogate
from saealib.surrogate.manager import (
    CompositeSurrogateManager,
    GlobalSurrogateManager,
    LocalSurrogateManager,
    SurrogateManager,
    product_combine,
    rank_weighted_combine,
)
from saealib.surrogate.per_objective import PerObjectiveSurrogate
from saealib.surrogate.prediction import SurrogatePrediction
from saealib.surrogate.rbf import RBFSurrogate, gaussian_kernel
from saealib.surrogate.sklearn_surrogate import (
    DTSurrogate,
    GPRSurrogate,
    LGBMSurrogate,
    NNSurrogate,
    SklearnSurrogate,
    SVMSurrogate,
    XGBSurrogate,
)
from saealib.surrogate.switching import (
    AccuracyBasedSurrogateSwitcher,
    GenCtrlSwitcher,
    ManagerSwitcher,
    StrategySwitcher,
)
from saealib.surrogate.torch_surrogate import TorchSurrogate
from saealib.surrogate.training_set import (
    ArchiveObjectiveSet,
    ConstraintObjectiveSet,
    FeasibilityClassificationSet,
    KNNConstraintObjectiveSet,
    KNNObjectiveSet,
    LevelBasedSet,
    PairwiseComparisonSet,
    ReferencePointComparisonSet,
    TopKBipartitionSet,
    TrainingData,
    TrainingSet,
)

__all__ = [
    "RMSE",
    "AccuracyBasedSurrogateSwitcher",
    "AccuracyEvaluator",
    "ArchiveBasedManager",
    "ArchiveObjectiveSet",
    "CompositeSurrogateManager",
    "ConstraintObjectiveSet",
    "DTSurrogate",
    "DensityManager",
    "FeasibilityClassificationSet",
    "GPRSurrogate",
    "GenCtrlSwitcher",
    "GlobalSurrogateManager",
    "HeldOutAccuracyEvaluator",
    "KFoldAccuracyEvaluator",
    "KNNConstraintObjectiveSet",
    "KNNObjectiveSet",
    "LGBMSurrogate",
    "LOOAccuracyEvaluator",
    "LevelBasedSet",
    "LocalSurrogateManager",
    "ManagerSwitcher",
    "NNSurrogate",
    "NichingManager",
    "NoveltyManager",
    "PairwiseComparisonSet",
    "PerObjectiveSurrogate",
    "R2Score",
    "RBFSurrogate",
    "ReferencePointComparisonSet",
    "SVMSurrogate",
    "SklearnSurrogate",
    "SpearmanCorrelation",
    "StrategySwitcher",
    "Surrogate",
    "SurrogateAccuracy",
    "SurrogateAccuracyMetric",
    "SurrogateManager",
    "SurrogatePrediction",
    "TopKBipartitionSet",
    "TorchSurrogate",
    "TrainingData",
    "TrainingSet",
    "XGBSurrogate",
    "gaussian_kernel",
    "product_combine",
    "rank_weighted_combine",
]


def __getattr__(name: str) -> object:
    if name == "GPSurrogate":
        from saealib.surrogate._deprecated import GPSurrogate

        return GPSurrogate
    if name == "RBFsurrogate":
        from saealib.surrogate.rbf import RBFSurrogate

        return RBFSurrogate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
