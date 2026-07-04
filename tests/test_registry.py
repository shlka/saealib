"""Unit tests for saealib.registry: register/get/build resolution."""

import pytest

from saealib.exceptions import ValidationError
from saealib.registry import build, get, register


class Engine:
    def __init__(self, power: int = 100):
        self.power = power


class Car:
    def __init__(self, engine=None, wheels: int = 4):
        self.engine = engine
        self.wheels = wheels


class TestRegisterAndGet:
    def test_register_default_name(self):
        register()(Engine)
        assert get("Engine") is Engine

    def test_register_explicit_name(self):
        register("MyEngine")(Engine)
        assert get("MyEngine") is Engine

    def test_get_unknown_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            get("NoSuchThing")

    def test_get_dotted_path_resolves_without_registration(self):
        from collections import OrderedDict

        resolved = get("collections.OrderedDict")
        assert resolved is OrderedDict

    def test_get_dotted_path_bad_module_raises_validation_error(self):
        with pytest.raises(ValidationError):
            get("no.such.module.Thing")

    def test_get_dotted_path_bad_attr_raises_validation_error(self):
        with pytest.raises(ValidationError):
            get("collections.NoSuchClass")


class TestBuild:
    def setup_method(self):
        register("Engine")(Engine)
        register("Car")(Car)

    def test_build_passthrough_live_instance(self):
        instance = Engine(power=200)
        assert build(instance) is instance

    def test_build_from_bare_string(self):
        obj = build("Engine")
        assert isinstance(obj, Engine)
        assert obj.power == 100

    def test_build_from_spec_with_params(self):
        obj = build({"type": "Engine", "params": {"power": 250}})
        assert isinstance(obj, Engine)
        assert obj.power == 250

    def test_build_recurses_into_nested_spec(self):
        obj = build(
            {
                "type": "Car",
                "params": {"engine": {"type": "Engine", "params": {"power": 300}}},
            }
        )
        assert isinstance(obj, Car)
        assert isinstance(obj.engine, Engine)
        assert obj.engine.power == 300

    def test_build_leaves_non_spec_params_untouched(self):
        obj = build({"type": "Car", "params": {"wheels": 3}})
        assert obj.wheels == 3
        assert obj.engine is None

    def test_build_unknown_type_raises_validation_error(self):
        with pytest.raises(ValidationError):
            build({"type": "NoSuchThing", "params": {}})

    def test_build_unknown_param_raises_validation_error(self):
        with pytest.raises(ValidationError):
            build({"type": "Engine", "params": {"not_a_param": 1}})

    def test_build_missing_required_param_raises_validation_error(self):
        class Needy:
            def __init__(self, required_arg):
                self.required_arg = required_arg

        register("Needy")(Needy)
        with pytest.raises(ValidationError):
            build({"type": "Needy", "params": {}})
