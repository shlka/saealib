"""Unit tests for saealib.registry: register/get/build/to_spec resolution."""

import numpy as np
import pytest

from saealib.exceptions import ValidationError
from saealib.registry import (
    _inject_params,
    _strip_params,
    build,
    get,
    register,
    to_spec,
)


class Engine:
    def __init__(self, power: int = 100):
        self.power = power


class Car:
    def __init__(self, engine=None, wheels: int = 4):
        self.engine = engine
        self.wheels = wheels


class Convoy:
    """Constructor takes *args — exercises the vararg spec path."""

    def __init__(self, *vehicles):
        self.vehicles = vehicles


class OptionalOnly:
    def __init__(self, note: str = "default"):
        self.note = note


class Opaque:
    """Opts out of generic reflection via ``_registry_spec``."""

    def __init__(self, spec=None):
        self._registry_spec = spec


class Powered:
    """Takes a callable param — exercises the callable-ref spec path."""

    def __init__(self, kernel=None):
        self.kernel = kernel


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


class TestBuildVarargs:
    def setup_method(self):
        register("Engine")(Engine)
        register("Convoy")(Convoy)

    def test_list_params_call_positionally(self):
        obj = build(
            {
                "type": "Convoy",
                "params": [
                    {"type": "Engine", "params": {"power": 1}},
                    {"type": "Engine", "params": {"power": 2}},
                ],
            }
        )
        assert isinstance(obj, Convoy)
        assert [v.power for v in obj.vehicles] == [1, 2]

    def test_empty_list_params(self):
        obj = build({"type": "Convoy", "params": []})
        assert obj.vehicles == ()


class TestBuildCallableRef:
    def setup_method(self):
        register("Powered")(Powered)

    def test_build_resolves_callable_ref_param(self):
        import math

        obj = build(
            {"type": "Powered", "params": {"kernel": {"callable": "math.isnan"}}}
        )
        assert obj.kernel is math.isnan

    def test_build_top_level_callable_ref(self):
        import math

        assert build({"callable": "math.isnan"}) is math.isnan


class TestToSpec:
    def setup_method(self):
        register("Engine")(Engine)
        register("Car")(Car)
        register("Convoy")(Convoy)
        register("OptionalOnly")(OptionalOnly)
        register("Opaque")(Opaque)
        register("Powered")(Powered)

    @pytest.mark.parametrize(
        "value", [None, True, 1, 1.5, "s", [1, 2], (1, 2), {"a": 1}]
    )
    def test_primitives_and_containers_pass_through_structurally(self, value):
        assert to_spec(value) == (list(value) if isinstance(value, tuple) else value)

    def test_numpy_array_becomes_list(self):
        assert to_spec(np.array([1.0, 2.0])) == [1.0, 2.0]

    def test_function_becomes_callable_ref(self):
        import math

        assert to_spec(math.isnan) == {"callable": "math.isnan"}

    def test_registered_instance_reflects_constructor_params(self):
        obj = Engine(power=42)
        assert to_spec(obj) == {"type": "Engine", "params": {"power": 42}}

    def test_nested_registered_instance_recurses(self):
        car = Car(engine=Engine(power=7), wheels=3)
        assert to_spec(car) == {
            "type": "Car",
            "params": {
                "engine": {"type": "Engine", "params": {"power": 7}},
                "wheels": 3,
            },
        }

    def test_unregistered_class_falls_back_to_dotted_path_type(self):
        class NotRegistered:
            def __init__(self, note: str = "default"):
                self.note = note

        spec = to_spec(NotRegistered(note="x"))
        assert spec == {
            "type": f"{__name__}.{NotRegistered.__qualname__}",
            "params": {"note": "x"},
        }

    def test_missing_optional_attr_is_omitted(self):
        obj = OptionalOnly()
        del obj.note
        assert to_spec(obj) == {"type": "OptionalOnly", "params": {}}

    def test_missing_required_attr_raises_validation_error(self):
        class Needy:
            def __init__(self, required_arg):
                pass  # does not store self.required_arg

        register("Needy")(Needy)
        with pytest.raises(ValidationError):
            to_spec(Needy(1))

    def test_var_positional_becomes_list_params(self):
        convoy = Convoy(Engine(power=1), Engine(power=2))
        assert to_spec(convoy) == {
            "type": "Convoy",
            "params": [
                {"type": "Engine", "params": {"power": 1}},
                {"type": "Engine", "params": {"power": 2}},
            ],
        }

    def test_var_positional_missing_attr_raises_validation_error(self):
        class BadConvoy:
            def __init__(self, *vehicles):
                pass  # does not store self.vehicles

        register("BadConvoy")(BadConvoy)
        with pytest.raises(ValidationError):
            to_spec(BadConvoy())

    def test_registry_spec_hook_used_instead_of_reflection(self):
        obj = Opaque(spec={"type": "whatever", "params": {"x": 1}})
        assert to_spec(obj) == {"type": "whatever", "params": {"x": 1}}

    def test_registry_spec_hook_none_raises_validation_error(self):
        obj = Opaque(spec=None)
        with pytest.raises(ValidationError):
            to_spec(obj)

    def test_round_trip_build_to_spec_build(self):
        original = Car(engine=Engine(power=9), wheels=6)
        rebuilt = build(to_spec(original))
        assert isinstance(rebuilt, Car)
        assert rebuilt.wheels == 6
        assert rebuilt.engine.power == 9

    def test_round_trip_varargs(self):
        original = Convoy(Engine(power=1), Engine(power=2))
        rebuilt = build(to_spec(original))
        assert isinstance(rebuilt, Convoy)
        assert [v.power for v in rebuilt.vehicles] == [1, 2]

    def test_round_trip_function_param(self):
        import math

        original = Powered(kernel=math.isnan)
        spec = to_spec(original)
        assert spec == {
            "type": "Powered",
            "params": {"kernel": {"callable": "math.isnan"}},
        }
        rebuilt = build(spec)
        assert rebuilt.kernel is math.isnan

    def test_lambda_raises_validation_error(self):
        with pytest.raises(ValidationError):
            to_spec(lambda x: x)

    def test_nested_function_raises_validation_error(self):
        def inner(x):
            return x

        with pytest.raises(ValidationError):
            to_spec(inner)


class TestStripParams:
    def test_strips_key_from_flat_spec(self):
        spec = {"type": "Engine", "params": {"power": 1, "dim": 3}}
        stripped = _strip_params(spec, "dim")
        assert stripped == {"type": "Engine", "params": {"power": 1}}

    def test_strips_recursively_from_nested_dict_params(self):
        spec = {
            "type": "Car",
            "params": {
                "engine": {"type": "Engine", "params": {"power": 1, "direction": [1]}},
                "dim": 3,
            },
        }
        stripped = _strip_params(spec, "dim", "direction")
        assert stripped == {
            "type": "Car",
            "params": {"engine": {"type": "Engine", "params": {"power": 1}}},
        }

    def test_strips_recursively_from_list_params(self):
        spec = {
            "type": "Convoy",
            "params": [{"type": "Engine", "params": {"power": 1, "dim": 3}}],
        }
        stripped = _strip_params(spec, "dim")
        assert stripped == {
            "type": "Convoy",
            "params": [{"type": "Engine", "params": {"power": 1}}],
        }

    def test_is_non_destructive(self):
        spec = {"type": "Engine", "params": {"power": 1, "dim": 3}}
        _strip_params(spec, "dim")
        assert spec == {"type": "Engine", "params": {"power": 1, "dim": 3}}


class TestTopLevelExport:
    def test_register_importable_from_top_level(self):
        import saealib

        assert saealib.register is register
        assert "register" in saealib.__all__

    def test_get_build_to_spec_not_exported_at_top_level(self):
        import saealib

        assert "build" not in saealib.__all__
        assert "get" not in saealib.__all__
        assert "to_spec" not in saealib.__all__
        assert not hasattr(saealib, "build")
        assert not hasattr(saealib, "get")
        assert not hasattr(saealib, "to_spec")

    def test_get_build_to_spec_importable_via_registry_namespace(self):
        import saealib.registry

        assert saealib.registry.build is build
        assert saealib.registry.get is get
        assert saealib.registry.to_spec is to_spec


class TestInjectParams:
    def setup_method(self):
        register("Engine")(Engine)
        register("Car")(Car)

    def test_injects_missing_param(self):
        spec = {"type": "Engine", "params": {}}
        injected = _inject_params(spec, power=250)
        assert injected == {"type": "Engine", "params": {"power": 250}}

    def test_does_not_overwrite_existing_param(self):
        spec = {"type": "Engine", "params": {"power": 42}}
        injected = _inject_params(spec, power=250)
        assert injected == {"type": "Engine", "params": {"power": 42}}

    def test_does_not_inject_param_absent_from_signature(self):
        spec = {"type": "Engine", "params": {}}
        injected = _inject_params(spec, not_a_param=1)
        assert injected == {"type": "Engine", "params": {}}

    def test_injects_into_nested_spec(self):
        spec = {
            "type": "Car",
            "params": {"engine": {"type": "Engine", "params": {}}, "wheels": 4},
        }
        injected = _inject_params(spec, power=99)
        assert injected == {
            "type": "Car",
            "params": {
                "engine": {"type": "Engine", "params": {"power": 99}},
                "wheels": 4,
            },
        }

    def test_is_non_destructive(self):
        spec = {"type": "Engine", "params": {}}
        _inject_params(spec, power=250)
        assert spec == {"type": "Engine", "params": {}}
