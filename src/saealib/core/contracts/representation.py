"""RepresentationSpec, RepresentationKind, and the REPRESENTATION_KINDS registry.

``RepresentationSpec`` is the value that binds the ``representation`` schema
variable of ``adr-0002`` §6.  A spec is a registered kind name plus its own
named parameters.

``RepresentationKind`` is the registry descriptor for one kind.  The ``unify``
callable defaults to ``None``, meaning per-parameter equality.  Override it
when a parameter should unify by containment rather than equality (e.g. a
sequence length-range that accepts a narrower consumer).

``ParameterSpec`` is a single named parameter declaration.  Its ``value``
field is a ``SchemaBinding`` (``Var`` / ``Fixed`` / ``Contained`` / ``Product``
from ``data.py``).  Using the same carrier types as ``DataSpec.bindings``
means that per-parameter unification reuses the existing ``_unify_pair``
machinery in ``schema.py`` and introduces no second vocabulary for values.

Design note on ``ParameterSpec.value`` vs ``DataSpec.bindings`` carriers
-----------------------------------------------------------------------
``ParameterSpec.value`` is a ``SchemaBinding`` — exactly the same union
(``Var`` / ``Fixed`` / ``Contained`` / ``Product``) as ``DataSpec.bindings``.
This is intentional: we reuse the unification engine in ``schema.py`` without
introducing a parallel carrier vocabulary.  The semantic difference is that
``DataSpec.bindings`` binds *schema variables declared in* ``SCHEMA_VARIABLES``,
whereas ``ParameterSpec.value`` binds *per-kind parameters* whose names are
local to the kind and unknown to ``SCHEMA_VARIABLES``.  Concretely, ``dim`` in
a vector spec is not a schema variable — it is a parameter name understood only
by the ``vector`` kind's unification logic.  We therefore pass a per-kind
parameter vocabulary (built from the kind's ``parameters`` tuple) when calling
``unify_bindings``, so parameter names are not flagged as ``unknown_variables``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from saealib.core.contracts.data import Contained, Fixed, Product, SchemaBinding, Var
from saealib.core.contracts.schema import (
    UnificationResult,
    unify_bindings,
)
from saealib.core.contracts.vocabulary import (
    Vocabulary,
    VocabularyDescriptor,
    validate_identifier,
)
from saealib.exceptions import ConfigurationError, ValidationError

__all__ = [
    "REPRESENTATION_KINDS",
    "ParameterSpec",
    "RepresentationKind",
    "RepresentationSpec",
    "unify_representation_specs",
]


# ---------------------------------------------------------------------------
# ParameterSpec
# ---------------------------------------------------------------------------

_SCHEMA_BINDING_TYPES = (Var, Fixed, Contained, Product)


@dataclass(frozen=True, kw_only=True)
class ParameterSpec:
    """One named parameter of a representation, expressed as a schema binding.

    The ``value`` field uses the same carrier types as ``DataSpec.bindings``
    (``Var`` / ``Fixed`` / ``Contained`` / ``Product``).  This is deliberate:
    per-parameter unification is delegated to the existing ``unify_bindings``
    machinery in ``schema.py``, so no second vocabulary of binding carriers is
    needed.

    **Relationship to ``DataSpec.bindings`` carriers**: ``ParameterSpec.value``
    is the same ``SchemaBinding`` union — not a separate type.  The difference
    is semantic: ``DataSpec.bindings`` binds names registered in
    ``SCHEMA_VARIABLES``, while ``ParameterSpec.value`` binds parameter names
    that are local to one representation kind.  When we call ``unify_bindings``
    for parameter unification we supply a per-kind registry so these local names
    are treated as known (see ``_kind_param_registry``).

    For the core kinds the value in a concrete ``RepresentationSpec`` is always
    a ``Fixed`` wrapping a plain Python scalar — ``Fixed(value=10)`` for a
    dimension, ``Fixed(value=5)`` for a permutation length.  In a
    ``RepresentationKind`` descriptor the value is ``Var(name=...)`` to express
    that the parameter must be bound by the spec.  ``Contained`` and ``Product``
    are available for custom kinds with richer semantics (e.g. alphabet subsets).
    """

    name: str
    value: SchemaBinding

    def __post_init__(self) -> None:
        """Validate the parameter name and value carrier."""
        validate_identifier(self.name)
        if not isinstance(self.value, _SCHEMA_BINDING_TYPES):
            raise ValidationError(
                "ParameterSpec value must be a schema binding "
                "(Var, Fixed, Contained, Product)"
            )


def _make_params_mapping(
    params: tuple[ParameterSpec, ...],
) -> MappingProxyType[str, SchemaBinding]:
    """Convert a parameter tuple into a name → binding mapping."""
    result: dict[str, SchemaBinding] = {}
    for p in params:
        if p.name in result:
            raise ValidationError(
                f"Duplicate parameter name in RepresentationKind: {p.name!r}"
            )
        result[p.name] = p.value
    return MappingProxyType(result)


def _kind_param_registry(
    kind_descriptor: RepresentationKind,
) -> Vocabulary[VocabularyDescriptor]:
    """Build a vocabulary of this kind's parameter names.

    ``unify_bindings`` classifies any name absent from its registry as an
    ``unknown_variable`` and returns ``unified=False``.  Representation
    parameters are local to their kind — not ``SCHEMA_VARIABLES`` — so we
    build a per-kind registry containing exactly the parameter names the kind
    declares.  This prevents ``dim``, ``length``, etc. from being treated as
    unknown variables during parameter unification.
    """
    reg: Vocabulary[VocabularyDescriptor] = Vocabulary(
        name_validator=validate_identifier
    )
    for p in kind_descriptor.parameters:
        reg.register(p.name, VocabularyDescriptor(name=p.name, description=""))
    return reg


# ---------------------------------------------------------------------------
# RepresentationKind
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class RepresentationKind(VocabularyDescriptor):
    """Registry descriptor for one representation kind.

    Attributes
    ----------
    name : str
        The stable kind name (e.g. ``"vector"``).  Registered permanently.
    description : str
        Human-readable description.
    parameters : tuple[ParameterSpec, ...]
        Declared parameters for this kind, in definition order.
    unify : callable or None
        Custom unification logic.  Signature::

            unify(provided: Mapping, required: Mapping,
                  registry: Vocabulary) -> UnificationResult

        where both mappings are ``{param_name: SchemaBinding}`` and
        ``registry`` is the per-kind parameter name vocabulary.  When ``None``
        (the default), per-parameter equality via
        :func:`~saealib.core.contracts.schema.unify_bindings` is used.
        Override only when a parameter must unify by containment instead of
        equality (e.g. the ``sequence`` kind's ``alphabet`` parameter).
    """

    parameters: tuple[ParameterSpec, ...] = ()
    unify: Callable[[Mapping, Mapping, Vocabulary], UnificationResult] | None = None

    def __post_init__(self) -> None:
        """Validate descriptor fields."""
        super().__post_init__()
        try:
            params = tuple(self.parameters)
        except TypeError as exc:
            raise ValidationError(
                "RepresentationKind.parameters must be iterable"
            ) from exc
        for p in params:
            if not isinstance(p, ParameterSpec):
                raise ValidationError(
                    "RepresentationKind.parameters must contain ParameterSpec values"
                )
        object.__setattr__(self, "parameters", params)
        # Validate that parameter names are unique
        _make_params_mapping(params)
        if self.unify is not None and not callable(self.unify):
            raise ValidationError("RepresentationKind.unify must be callable or None")


# ---------------------------------------------------------------------------
# REPRESENTATION_KINDS vocabulary
# ---------------------------------------------------------------------------


REPRESENTATION_KINDS: Vocabulary[RepresentationKind] = Vocabulary(
    name_validator=validate_identifier,
)
"""Registry of representation kind descriptors.

Registered names are permanent (``adr-0002`` §9 item 4).  Do not remove or
rename existing entries.  The initial registrations are ``vector``,
``permutation``, and ``sequence``.
"""


def _sequence_unify(
    provided: Mapping[str, SchemaBinding],
    required: Mapping[str, SchemaBinding],
    registry: Vocabulary[VocabularyDescriptor],
) -> UnificationResult:
    """Containment-based unification for the sequence kind.

    The ``alphabet`` parameter supports containment (``Contained``) so that a
    space offering a large alphabet accepts an operator that only requires a
    subset.  The ``min_length`` and ``max_length`` parameters use equality.
    The same ``unify_bindings`` call handles all three because the caller is
    responsible for putting ``Contained`` in the *required* binding when it
    wants containment semantics; the core engine in ``schema.py`` then applies
    the containment check.
    """
    return unify_bindings(provided, required, registry=registry)


REPRESENTATION_KINDS.register(
    "vector",
    RepresentationKind(
        name="vector",
        description="A fixed-length dense real-valued (or integer) vector.",
        parameters=(ParameterSpec(name="dim", value=Var(name="dim")),),
        unify=None,  # equality on dim — the default path
    ),
)

REPRESENTATION_KINDS.register(
    "permutation",
    RepresentationKind(
        name="permutation",
        description=(
            "A permutation of the integers [0, length).  "
            "Compatible operators must request exactly this kind — "
            "there is no subtype relation with integer vectors."
        ),
        parameters=(ParameterSpec(name="length", value=Var(name="length")),),
        unify=None,  # equality on length
    ),
)

REPRESENTATION_KINDS.register(
    "sequence",
    RepresentationKind(
        name="sequence",
        description=(
            "A variable-length sequence over a finite alphabet.  "
            "The alphabet parameter supports containment: a space with a larger "
            "alphabet satisfies an operator that requires a subset.  "
            "Use Contained(values=frozenset({...})) in the required spec's "
            "alphabet parameter to express a subset requirement."
        ),
        parameters=(
            ParameterSpec(name="alphabet", value=Var(name="alphabet")),
            ParameterSpec(name="min_length", value=Var(name="min_length")),
            ParameterSpec(name="max_length", value=Var(name="max_length")),
        ),
        unify=_sequence_unify,
    ),
)


# ---------------------------------------------------------------------------
# RepresentationSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class RepresentationSpec:
    """A concrete representation: a registered kind name plus its parameters.

    This is the value that binds the ``representation`` schema variable
    declared in :data:`~saealib.core.contracts.schema.SCHEMA_VARIABLES`.

    Parameters
    ----------
    kind : str
        A name registered in :data:`REPRESENTATION_KINDS`.
    parameters : tuple[ParameterSpec, ...]
        Concrete parameter values for the kind.  In a concrete spec the
        parameter values are typically ``Fixed`` bindings wrapping plain
        Python scalars.
    """

    kind: str
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate kind registration and parameter shapes."""
        validate_identifier(self.kind)
        if REPRESENTATION_KINDS.get(self.kind) is None:
            raise ConfigurationError(
                f"Unknown representation kind: {self.kind!r}. "
                f"Register it in REPRESENTATION_KINDS before use."
            )
        try:
            params = tuple(self.parameters)
        except TypeError as exc:
            raise ValidationError(
                "RepresentationSpec.parameters must be iterable"
            ) from exc
        for p in params:
            if not isinstance(p, ParameterSpec):
                raise ValidationError(
                    "RepresentationSpec.parameters must contain ParameterSpec values"
                )
        object.__setattr__(self, "parameters", params)
        _make_params_mapping(params)  # validates uniqueness

    def _as_bindings(self) -> dict[str, SchemaBinding]:
        """Return the parameters as a name → binding mapping."""
        return dict(_make_params_mapping(self.parameters))


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------


def unify_representation_specs(
    provided: RepresentationSpec,
    required: RepresentationSpec,
) -> UnificationResult:
    """Check whether *provided* satisfies *required*.

    Returns a :class:`~saealib.core.contracts.schema.UnificationResult`.
    The result is unified (``result.unified is True``) iff:

    1. Both specs share the same kind name.
    2. The kind's per-parameter unification (or custom ``unify``) finds no
       conflict.

    When the kind names differ the result carries a
    ``representation_kind_mismatch`` finding (which compiler rules
    map to an ``incompatible_representation`` diagnostic) and unification
    fails immediately — there is no subtype lattice between kinds.
    """
    if not isinstance(provided, RepresentationSpec) or not isinstance(
        required, RepresentationSpec
    ):
        raise ValidationError(
            "unify_representation_specs requires RepresentationSpec values"
        )

    from saealib.core.contracts.schema import SchemaConstraint, Substitution

    if provided.kind != required.kind:
        finding = SchemaConstraint(
            variable="representation",
            reason="representation_kind_mismatch",
            detail=(
                f"provided kind={provided.kind!r}, required kind={required.kind!r}"
            ),
        )
        return UnificationResult(
            substitution=Substitution(),
            findings=(finding,),
        )

    kind_descriptor = REPRESENTATION_KINDS.get(provided.kind)
    # kind_descriptor is guaranteed non-None by RepresentationSpec.__post_init__
    assert kind_descriptor is not None

    provided_bindings = provided._as_bindings()
    required_bindings = required._as_bindings()
    param_registry = _kind_param_registry(kind_descriptor)

    if kind_descriptor.unify is not None:
        return kind_descriptor.unify(
            provided_bindings, required_bindings, param_registry
        )

    return unify_bindings(provided_bindings, required_bindings, registry=param_registry)
