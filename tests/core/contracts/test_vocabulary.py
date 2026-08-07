from dataclasses import dataclass

import pytest

from saealib.core.contracts.vocabulary import (
    Vocabulary,
    VocabularyDescriptor,
    is_valid_name,
)
from saealib.exceptions import ConfigurationError, ValidationError


@dataclass(frozen=True, kw_only=True)
class RichDescriptor:
    name: str
    description: str
    variables: tuple[str, ...] = ()


def test_vocabulary_accepts_core_and_prefixed_names() -> None:
    vocabulary: Vocabulary[VocabularyDescriptor] = Vocabulary()
    core = VocabularyDescriptor(name="core_value", description="Core value")
    extension = VocabularyDescriptor(
        name="extension:value", description="Extension value"
    )

    vocabulary.register("core_value", core)
    vocabulary.register("extension:value", extension)

    assert vocabulary.get("core_value") is core
    assert vocabulary.get("extension:value") is extension
    assert vocabulary.names() == ("core_value", "extension:value")


@pytest.mark.parametrize("name", ["", ":value", "prefix:", "a:b:c", "not valid"])
def test_vocabulary_rejects_malformed_names(name: str) -> None:
    vocabulary: Vocabulary[VocabularyDescriptor] = Vocabulary()

    with pytest.raises(ValidationError):
        vocabulary.register(
            name,
            VocabularyDescriptor(name="valid_name", description="Value"),
        )


def test_vocabulary_rejects_duplicate_registration() -> None:
    vocabulary: Vocabulary[VocabularyDescriptor] = Vocabulary()
    descriptor = VocabularyDescriptor(name="value", description="Value")
    vocabulary.register("value", descriptor)

    with pytest.raises(ConfigurationError):
        vocabulary.register("value", descriptor)


def test_vocabulary_lookup_miss_does_not_raise() -> None:
    vocabulary: Vocabulary[RichDescriptor] = Vocabulary()
    vocabulary.register(
        "known",
        RichDescriptor(name="known", description="Known"),
    )

    assert vocabulary.get("unknown") is None
    assert not vocabulary.contains("unknown")
    assert is_valid_name("unknown")
    assert not is_valid_name("not valid")
    assert not vocabulary.is_deprecated("not valid")
    assert vocabulary.deprecation_reason("not valid") is None


def test_deprecated_name_cannot_be_reregistered() -> None:
    vocabulary: Vocabulary[VocabularyDescriptor] = Vocabulary()
    vocabulary.register(
        "old_value",
        VocabularyDescriptor(name="old_value", description="Old value"),
    )
    vocabulary.deprecate("old_value", "Use new_value instead")

    assert vocabulary.is_deprecated("old_value")
    assert vocabulary.deprecation_reason("old_value") == "Use new_value instead"
    assert vocabulary.get("old_value") is not None
    assert vocabulary.contains("old_value")
    with pytest.raises(ConfigurationError):
        vocabulary.register(
            "old_value",
            VocabularyDescriptor(name="old_value", description="Different value"),
        )
