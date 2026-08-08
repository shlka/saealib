from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from saealib.core.contracts.data import (
    Contained,
    DataSpec,
    Fixed,
    Product,
    SchemaBinding,
    Var,
    is_set_like,
)
from saealib.core.contracts.vocabulary import (
    Vocabulary,
    VocabularyDescriptor,
    validate_name,
)
from saealib.exceptions import ValidationError

__all__ = [
    "SCHEMA_CONFLICT_REASONS",
    "SCHEMA_VARIABLES",
    "SchemaConstraint",
    "Substitution",
    "UnificationResult",
    "unify_bindings",
    "unify_data_specs",
]


SCHEMA_VARIABLES: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _name, _description in (
    ("representation", "Bound by a RepresentationSpec."),
    ("candidate_count", "Bound by port cardinality and proposal size."),
    ("parent_count", "Bound by crossover arity."),
    ("objective_schema", "Bound by an ObservationSchema."),
    ("constraint_schema", "Bound by an ObservationSchema."),
    ("feature_schema", "Bound by a FeatureEncoder output."),
    ("fidelity", "Bound by a FidelityRef on an observation record."),
    (
        "species",
        "Reserved for cooperative coevolution; bound by nothing yet.",
    ),
    (
        "proposal_group",
        "Reserved for quality-diversity emitters; bound by nothing yet.",
    ),
):
    SCHEMA_VARIABLES.register(
        _name,
        VocabularyDescriptor(name=_name, description=_description),
    )


SCHEMA_CONFLICT_REASONS: Vocabulary[VocabularyDescriptor] = Vocabulary()
for _name, _description in (
    (
        "fixed_mismatch",
        "Two fixed bindings for one variable are not equal.",
    ),
    (
        "carrier_mismatch",
        "The two sides declare incompatible binding operations.",
    ),
    (
        "producer_binding_missing",
        "The consumer constrains a variable the producer does not declare.",
    ),
    (
        "containment_on_producer",
        "A containment requirement was declared on the producing side.",
    ),
    (
        "containment_requires_collection",
        "Containment was checked against a producer value without set semantics.",
    ),
    (
        "containment_unsatisfied",
        "The consumer requires values the producer does not provide.",
    ),
    (
        "containment_undecided",
        "Containment cannot be decided while the producer side is an unbound variable.",
    ),
    (
        "product_arity_mismatch",
        "Two product bindings compose a different arity.",
    ),
    (
        "representation_kind_mismatch",
        (
            "The two representation specs declare different kind names "
            "with no subtype relation."
        ),
    ),
):
    SCHEMA_CONFLICT_REASONS.register(
        _name,
        VocabularyDescriptor(name=_name, description=_description),
    )


def _validate_binding(binding: object) -> SchemaBinding:
    if not isinstance(binding, (Var, Fixed, Contained, Product)):
        raise ValidationError(
            "Schema bindings must be Var, Fixed, Contained, or Product"
        )
    return binding


def _copy_bindings(bindings: Mapping[str, SchemaBinding]) -> dict[str, SchemaBinding]:
    if not isinstance(bindings, Mapping):
        raise ValidationError("Schema bindings must be a mapping")
    copied = dict(bindings)
    for name, binding in copied.items():
        validate_name(name)
        _validate_binding(binding)
    return copied


@dataclass(frozen=True, kw_only=True)
class SchemaConstraint:
    """A finding emitted while unifying one schema variable."""

    variable: str
    reason: str
    detail: str = ""

    def __post_init__(self) -> None:
        """Validate the variable and registered conflict reason."""
        validate_name(self.variable)
        if SCHEMA_CONFLICT_REASONS.get(self.reason) is None:
            raise ValidationError(f"Unknown schema conflict reason: {self.reason!r}")
        if not isinstance(self.detail, str):
            raise ValidationError("Schema constraint detail must be a string")

    @property
    def deferred(self) -> bool:
        """Return whether this finding may be resolved by a later edge."""
        return self.reason == "containment_undecided"

    def __str__(self) -> str:
        """Return a compact diagnostic representation."""
        message = f"{self.variable}: {self.reason}"
        return f"{message} ({self.detail})" if self.detail else message


@dataclass(frozen=True, kw_only=True)
class Substitution:
    """An immutable accumulation of schema-variable assignments."""

    assignments: Mapping[str, SchemaBinding] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and detach the assignment mapping."""
        assignments = _copy_bindings(self.assignments)
        object.__setattr__(self, "assignments", MappingProxyType(assignments))

    def resolve(self, binding: SchemaBinding) -> SchemaBinding:
        """Resolve variable references through this substitution."""
        _validate_binding(binding)

        def visit(current: SchemaBinding, seen: set[str]) -> SchemaBinding:
            if isinstance(current, Var):
                if current.name not in self.assignments:
                    return current
                if current.name in seen:
                    raise ValidationError("Cyclic schema substitution")
                return visit(
                    self.assignments[current.name],
                    seen | {current.name},
                )
            if isinstance(current, Product):
                return Product(
                    elements=tuple(visit(element, seen) for element in current.elements)
                )
            return current

        return visit(binding, set())

    def bind(self, name: str, binding: SchemaBinding) -> Substitution:
        """Return a new substitution with one resolved assignment added."""
        validate_name(name)
        _validate_binding(binding)
        resolved = self.resolve(binding)
        if isinstance(resolved, Var) and resolved.name == name:
            return self
        assignments = dict(self.assignments)
        assignments[name] = resolved
        return Substitution(assignments=assignments)


@dataclass(frozen=True, kw_only=True)
class UnificationResult:
    """The substitution and findings produced by schema unification."""

    substitution: Substitution
    findings: tuple[SchemaConstraint, ...] = ()
    unknown_variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize the unification result."""
        if not isinstance(self.substitution, Substitution):
            raise ValidationError("Unification substitution must be a Substitution")
        findings = tuple(self.findings)
        if any(not isinstance(finding, SchemaConstraint) for finding in findings):
            raise ValidationError(
                "Unification findings must contain SchemaConstraint values"
            )
        unknown_variables = tuple(dict.fromkeys(self.unknown_variables))
        for variable in unknown_variables:
            validate_name(variable)
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "unknown_variables", unknown_variables)

    @property
    def conflicts(self) -> tuple[SchemaConstraint, ...]:
        """Return findings that are not deferred."""
        return tuple(finding for finding in self.findings if not finding.deferred)

    @property
    def deferred(self) -> tuple[SchemaConstraint, ...]:
        """Return findings that may be resolved by a later edge."""
        return tuple(finding for finding in self.findings if finding.deferred)

    @property
    def unified(self) -> bool:
        """Return whether unification found no conflict or unknown variable."""
        return not self.findings and not self.unknown_variables


def _constraint(
    findings: list[SchemaConstraint],
    variable: str,
    reason: str,
    detail: str = "",
) -> None:
    findings.append(SchemaConstraint(variable=variable, reason=reason, detail=detail))


def _unify_pair(
    provided: SchemaBinding,
    required: SchemaBinding,
    variable: str,
    substitution: Substitution,
    findings: list[SchemaConstraint],
) -> Substitution:
    left = substitution.resolve(provided)
    right = substitution.resolve(required)

    if isinstance(left, Var):
        if isinstance(right, Var):
            if left.name == right.name:
                return substitution
            return substitution.bind(left.name, right)
        if isinstance(right, Contained):
            _constraint(
                findings,
                variable,
                "containment_undecided",
                "the producer variable is unbound",
            )
            return substitution
        return substitution.bind(left.name, right)

    if isinstance(right, Var):
        return substitution.bind(right.name, left)

    if isinstance(left, Contained):
        _constraint(
            findings,
            variable,
            "containment_on_producer",
            "the producer declares Contained",
        )
        return substitution

    if isinstance(right, Contained):
        if not isinstance(left, Fixed) or not is_set_like(left.value):
            _constraint(
                findings,
                variable,
                "containment_requires_collection",
                "the producer value has no set semantics",
            )
            return substitution
        try:
            provided_values = frozenset(cast(Iterable[object], left.value))
        except TypeError:
            _constraint(
                findings,
                variable,
                "containment_requires_collection",
                "the producer value cannot be normalized as a collection",
            )
            return substitution
        missing = right.values - provided_values
        if missing:
            _constraint(
                findings,
                variable,
                "containment_unsatisfied",
                f"missing={missing!r}",
            )
        return substitution

    if isinstance(left, Product) and isinstance(right, Product):
        if len(left.elements) != len(right.elements):
            _constraint(
                findings,
                variable,
                "product_arity_mismatch",
                f"provided={len(left.elements)}, required={len(right.elements)}",
            )
            return substitution
        for provided_element, required_element in zip(
            left.elements,
            right.elements,
        ):
            substitution = _unify_pair(
                provided_element,
                required_element,
                variable,
                substitution,
                findings,
            )
        return substitution

    if isinstance(left, Fixed) and isinstance(right, Fixed):
        if left.value != right.value:
            _constraint(
                findings,
                variable,
                "fixed_mismatch",
                f"provided={left.value!r}, required={right.value!r}",
            )
        return substitution

    if isinstance(left, (Fixed, Product)) and isinstance(right, (Fixed, Product)):
        _constraint(
            findings,
            variable,
            "carrier_mismatch",
            f"provided={type(left).__name__}, required={type(right).__name__}",
        )
        return substitution

    raise ValidationError("Unsupported schema binding pair")


def unify_bindings(
    provided: Mapping[str, SchemaBinding],
    required: Mapping[str, SchemaBinding],
    *,
    substitution: Substitution | None = None,
    registry: Vocabulary[VocabularyDescriptor] | None = None,
) -> UnificationResult:
    """Unify producer bindings with consumer-side requirements."""
    provided_bindings = _copy_bindings(provided)
    required_bindings = _copy_bindings(required)
    current = Substitution() if substitution is None else substitution
    if not isinstance(current, Substitution):
        raise ValidationError("Unification substitution must be a Substitution")
    variable_registry = SCHEMA_VARIABLES if registry is None else registry

    unknown_variables: list[str] = []
    for name in (*provided_bindings, *required_bindings):
        if variable_registry.get(name) is None and name not in unknown_variables:
            unknown_variables.append(name)

    findings: list[SchemaConstraint] = []
    for name, required_binding in required_bindings.items():
        if name not in provided_bindings:
            resolved_required = current.resolve(required_binding)
            if not isinstance(resolved_required, Var):
                _constraint(
                    findings,
                    name,
                    "producer_binding_missing",
                    "the producer does not declare this binding",
                )
            continue
        current = _unify_pair(
            provided_bindings[name],
            required_binding,
            name,
            current,
            findings,
        )

    return UnificationResult(
        substitution=current,
        findings=tuple(findings),
        unknown_variables=tuple(unknown_variables),
    )


def unify_data_specs(
    provided: DataSpec,
    required: DataSpec,
    *,
    substitution: Substitution | None = None,
    registry: Vocabulary[VocabularyDescriptor] | None = None,
) -> UnificationResult:
    """Unify the schema bindings declared by two data specifications."""
    if not isinstance(provided, DataSpec) or not isinstance(required, DataSpec):
        raise ValidationError("Unification requires DataSpec values")
    return unify_bindings(
        provided.bindings,
        required.bindings,
        substitution=substitution,
        registry=registry,
    )
