Thank you for your interest in contributing to `saealib`! We welcome contributions from everyone. This document outlines the process for contributing to this project.

### 1. Development Environment Setup

This project uses **[uv](https://github.com/astral-sh/uv)** for dependency management and **[Ruff](https://github.com/astral-sh/ruff)** for linting and formatting.

1. **Install uv** (if you haven't already):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```


2. **Clone the repository**:
    ```bash
    git clone https://github.com/shlka/saealib.git
    cd saealib
    ```


3. **Install dependencies**:
    ```bash
    uv pip install -e .[dev]
    # Or if you are using a virtual environment managed by uv:
    uv sync
    ```



### 2. Coding Style

We adhere to strict coding standards to ensure maintainability.

* **Linter & Formatter**: We use `ruff`. Please ensure your code passes all checks before submitting a PR.
    ```bash
    ruff check .
    ruff format .
    ```


* **Type Hinting**: Type hints are strongly encouraged for all public APIs.
* **Docstrings**: We follow the **NumPy style** docstrings.

### 3. Deprecation Policy

When renaming or removing a public API, add a backward-compatible fallback using the utilities in `src/saealib/_deprecated.py`. Always use `FutureWarning` (visible to users by default).

#### Renamed keyword argument

```python
from saealib._deprecated import deprecated_param

class MyComparator(Comparator):
    @deprecated_param("old_name", "new_name", "0.2.0")
    def __init__(self, new_name=None, ...):
        ...
```

The decorator intercepts `old_name` from `**kwargs`, emits a `FutureWarning`, and forwards the value to `new_name`. Remove `old_name` from the signature entirely.

#### One old argument mapping to multiple new arguments

Use `warn_deprecated` inline when a single old parameter maps to more than one new parameter:

```python
from saealib._deprecated import warn_deprecated

def __init__(self, ..., eps=None, *, eps_cv=1e-6, eps_obj=1e-6):
    if eps is not None:
        warn_deprecated("eps", "eps_cv and eps_obj", "0.2.0")
        eps_cv = eps_obj = eps
```

#### Deprecated class alias

```python
from saealib._deprecated import deprecated_class

@deprecated_class("NewClassName")
class OldClassName(NewClassName):
    """Deprecated alias of :class:`NewClassName`."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
```

#### Testing

Assert the warning category is `FutureWarning`:

```python
def test_old_name_warns():
    with pytest.warns(FutureWarning):
        MyComparator(old_name=1.0)
```

If a test class exercises a deprecated API without testing the warning itself, suppress it at the class level:

```python
@pytest.mark.filterwarnings("ignore::FutureWarning")
class TestLegacyBehavior:
    ...
```

### 4. Testing Policy

We use `pytest` for testing. Since this library involves stochastic evolutionary algorithms, we have a specific testing strategy to distinguish between "expected random variations" and "regressions."

#### Running Tests

```bash
pytest
```

#### Test Categories

1. **Smoke Tests**: Checks if the example code runs without errors.
2. **Unit Tests**: Verifies data structures (`Population`) and deterministic logic (`Repair`, `Constraints`).
3. **Fixed-Seed Integration Tests**: Runs a full optimization with a fixed random seed and asserts the result stays within an acceptable threshold (e.g., `test_integration`). If this fails, check whether your change unintentionally degraded optimization performance.


### 5. Pull Request Guidelines

1. Ensure all tests pass locally.
2. Use the provided Pull Request Template.
