from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum

from saealib.core.contracts.vocabulary import Vocabulary, VocabularyDescriptor
from saealib.exceptions import ValidationError

__all__ = [
    "DIAGNOSTIC_CODES",
    "ContractPath",
    "Diagnostic",
    "DiagnosticBag",
    "DiagnosticCodeVocabulary",
    "Severity",
]


class Severity(str, Enum):
    """Severity levels available to compiler diagnostics."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, kw_only=True)
class ContractPath:
    """An ordered component path with an optional role and port."""

    components: tuple[str, ...]
    role: str | None = None
    port: str | None = None

    def __post_init__(self) -> None:
        """Validate path components."""
        components = tuple(self.components)
        if not components or any(
            not isinstance(component, str) or not component for component in components
        ):
            raise ValidationError("A contract path must contain a component id")
        if self.role is not None and (not isinstance(self.role, str) or not self.role):
            raise ValidationError("A contract path role must be a non-empty string")
        if self.port is not None and (not isinstance(self.port, str) or not self.port):
            raise ValidationError("A contract path port must be a non-empty string")
        object.__setattr__(self, "components", components)

    def __str__(self) -> str:
        """Render the path as a dotted name with an optional role."""
        rendered = ".".join(self.components)
        if self.role is not None:
            rendered += f"[{self.role}]"
        if self.port is not None:
            rendered += f".{self.port}"
        return rendered


DiagnosticCodeVocabulary = Vocabulary[VocabularyDescriptor]
DIAGNOSTIC_CODES = DiagnosticCodeVocabulary()
DIAGNOSTIC_CODES.register(
    "unknown_data_spec",
    VocabularyDescriptor(
        name="unknown_data_spec",
        description="A data specification kind has not been registered.",
    ),
)
DIAGNOSTIC_CODES.register(
    "schema_variable_unbound",
    VocabularyDescriptor(
        name="schema_variable_unbound",
        description="A schema variable could not be bound during compilation.",
    ),
)
DIAGNOSTIC_CODES.register(
    "unregistered_diagnostic_code",
    VocabularyDescriptor(
        name="unregistered_diagnostic_code",
        description="A diagnostic was emitted with an unregistered code.",
    ),
)


@dataclass(frozen=True, kw_only=True)
class Diagnostic:
    """A structured compiler finding attached to a contract path."""

    severity: Severity
    code: str
    message: str
    path: ContractPath
    related: tuple[ContractPath, ...] = ()
    resolutions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize diagnostic fields."""
        if not isinstance(self.path, ContractPath):
            raise ValidationError("Diagnostic paths must be ContractPath values")
        related = tuple(self.related)
        if any(not isinstance(path, ContractPath) for path in related):
            raise ValidationError(
                "Related diagnostic paths must be ContractPath values"
            )
        object.__setattr__(self, "related", related)
        object.__setattr__(self, "resolutions", tuple(self.resolutions))

    def __str__(self) -> str:
        """Render the diagnostic with its path."""
        return f"{self.severity.value} [{self.code}] at {self.path}: {self.message}"


class DiagnosticBag:
    """An append-only collection of diagnostics."""

    def __init__(self, diagnostics: Iterable[Diagnostic] = ()) -> None:
        self._diagnostics: list[Diagnostic] = list(diagnostics)

    def append(self, diagnostic: Diagnostic) -> None:
        """Append one diagnostic."""
        self._diagnostics.append(diagnostic)

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        """Append diagnostics from an iterable."""
        self._diagnostics.extend(diagnostics)

    @property
    def has_errors(self) -> bool:
        """Return whether an error diagnostic is present."""
        return any(
            diagnostic.severity is Severity.ERROR for diagnostic in self._diagnostics
        )

    def __iter__(self) -> Iterator[Diagnostic]:
        """Iterate over collected diagnostics."""
        return iter(self._diagnostics)

    def __len__(self) -> int:
        """Return the number of collected diagnostics."""
        return len(self._diagnostics)
