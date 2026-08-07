from saealib.core.contracts.roles import ROLES


def test_roles_are_registered_name_only_descriptors() -> None:
    assert ROLES.names() == (
        "proposer",
        "feedback_consumer",
        "fitter",
        "predictor",
        "acquisition",
        "population_comparator",
        "pairwise_comparator",
        "crossover",
        "mutation",
        "parent_selection",
        "survivor_selection",
    )
    descriptor = ROLES.get("proposer")
    assert descriptor is not None
    assert descriptor.description
