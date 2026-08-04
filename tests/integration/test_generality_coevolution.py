import importlib.util
from pathlib import Path


def _load_example():
    path = Path(__file__).parents[2] / "examples" / "generality_coevolution.py"
    spec = importlib.util.spec_from_file_location("generality_coevolution", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coevolution_optimizes_all_coordinate_blocks():
    example = _load_example()
    _context, states, trace = example.main()

    assert all(state.gen > 1 for state in states)
    assert all(state.fe == example.EVALUATION_BUDGET for state in states)
    assert all(len(state.archive) > len(state.population) for state in states)
    assert trace[-1] < trace[0]
