from __future__ import annotations

from typing import TypeAlias

from saealib.core.contracts.vocabulary import Vocabulary, VocabularyDescriptor

__all__ = ["ROLES", "RoleName"]

RoleName: TypeAlias = str
ROLES: Vocabulary[VocabularyDescriptor] = Vocabulary()

for _name, _description in (
    ("proposer", "Proposes candidates."),
    ("feedback_consumer", "Consumes feedback into a state patch."),
    ("evaluation_planner", "Plans evaluation requests for candidate populations."),
    ("feedback_builder", "Builds feedback batches for candidate populations."),
    ("fitter", "Fits a surrogate model."),
    ("predictor", "Predicts with a fitted surrogate."),
    ("acquisition", "Evaluates acquisition values."),
    ("initializer", "Creates the initial optimization state."),
    ("evaluator", "Evaluates candidate genomes against a problem."),
    ("decomposition", "Scalarizes objective vectors into scores."),
    ("strategy", "Advances an optimization state by one step."),
    ("training_set", "Builds training features and targets."),
    ("population_comparator", "Compares populations."),
    ("pairwise_comparator", "Compares pairs of candidates."),
    ("crossover", "Combines parent genomes into offspring genomes."),
    ("mutation", "Mutates candidate genomes."),
    ("parent_selection", "Selects parent indices for reproduction."),
    ("survivor_selection", "Selects survivor indices from a population."),
    ("duplicate_filter", "Filters offspring rows that duplicate a population."),
):
    ROLES.register(
        _name,
        VocabularyDescriptor(name=_name, description=_description),
    )
