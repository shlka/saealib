"""Regression test for saealib.surrogate.__all__ integrity."""

import saealib.surrogate


def test_all_entries_resolve():
    for name in saealib.surrogate.__all__:
        assert hasattr(saealib.surrogate, name), f"{name} in __all__ but not defined"


def test_star_import():
    namespace: dict = {}
    exec("from saealib.surrogate import *", namespace)
    for name in saealib.surrogate.__all__:
        assert name in namespace
