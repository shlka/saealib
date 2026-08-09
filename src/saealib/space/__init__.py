"""SearchSpace protocol, ServiceRegistry, VectorSpace, and ObjectSpace."""

from saealib.space.object_space import ObjectSpace
from saealib.space.permutation import PermutationSpace
from saealib.space.sequence import SequenceSpace
from saealib.space.services import (
    BoundsService,
    CloneService,
    ComparisonService,
    DenseNumericView,
    DistanceService,
    EquivalenceService,
    FeatureEncoder,
    FingerprintService,
    GenomeCodec,
    SamplingService,
    ValidationService,
)
from saealib.space.space import (
    DerivedSamplingService,
    DerivedValidationService,
    SearchSpace,
    ServiceRegistry,
    ValidationResult,
)
from saealib.space.vector import VectorSpace

__all__ = [
    "BoundsService",
    "CloneService",
    "ComparisonService",
    "DenseNumericView",
    "DerivedSamplingService",
    "DerivedValidationService",
    "DistanceService",
    "EquivalenceService",
    "FeatureEncoder",
    "FingerprintService",
    "GenomeCodec",
    "ObjectSpace",
    "PermutationSpace",
    "SamplingService",
    "SearchSpace",
    "SequenceSpace",
    "ServiceRegistry",
    "ValidationResult",
    "ValidationService",
    "VectorSpace",
]
