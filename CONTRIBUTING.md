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

### 6. Release Process

#### Versioning

This project follows [Semantic Versioning](https://semver.org/) (`x.y.z`):

* **`x` (major)**: large-scale rewrites or breaking changes to the public API.
* **`y` (minor)**: backward-compatible feature additions (new algorithms, surrogates, strategies, etc.).
* **`z` (patch)**: backward-compatible fixes and small corrections, including documentation-only fixes.

While the major version is `0` (pre-1.0, e.g. `0.1.0`), the public API is not considered stable and may still change between minor versions — this is the standard SemVer meaning of a `0.x` major version, not an exception to it. `0.1.0bN` pre-release identifiers (`b1`, `b2`, ...) mark beta iterations leading up to the `0.1.0` release itself.

#### Tagging and what it triggers

A release is cut by pushing a `vX.Y.Z` (optionally with a `bN` pre-release suffix) tag on `main`. This is the single trigger for:

* Publishing the package to PyPI (`.github/workflows/publish.yml`) — for a tag on `main`, this happens automatically. For a tag cut from a `release/vX.Y.x` branch, see "Backporting to an old minor series" below.
* Building and deploying version-tagged documentation snapshots (`.github/workflows/docs.yml`, `deploy-release` job) — final-release snapshots (`vX.Y.Z/`), the `versions.json` version-switcher list, and the root redirect are all only regenerated here.
* For final releases (`vX.Y.Z`, no pre-release suffix) only: keeping the `release/vX.Y.x` maintenance branch for that minor series in sync with `main` (`create-release-branch` job). The branch is created on the first final release in a minor series, then fast-forwarded on every later final release in the same series — as long as the tag is still an ancestor of `main` (i.e., `main` is still the one shipping patches for that series). Once `main` moves on to a newer minor series, or once `release/vX.Y.x` has taken on maintenance commits of its own (a tag cut directly from the branch rather than from `main`), the job leaves it alone rather than forcing a sync.

Documentation's rolling `dev/` build is deployed separately on every push to `main`, regardless of tags — see "Documentation snapshots" below.

Each final release's minor series (`X.Y`) gets its own `release/vX.Y.x` branch, auto-created and kept in sync with `main` as above for as long as `main` is still shipping that series. `main` itself is not a per-version maintenance branch — once `main` has moved on to a newer minor series, don't try to backdate a patch tag on `main` for an older one. If a mistake is found in an already-released minor series, fix it on the matching `release/vX.Y.x` branch and cut the new patch tag (`vX.Y.Z+1`) from there. Patching an old minor series this way is best-effort, not a maintenance commitment — there's no obligation to backport every fix.

#### Backporting to an old minor series

A tag cut from a `release/vX.Y.x` branch is, by definition, not an ancestor of `main` (`main` has moved on). `publish.yml`'s `check-tag` job detects this and routes the `publish` job through the `pypi-legacy` GitHub Environment instead of `pypi`. Unlike `pypi`, `pypi-legacy` has required reviewers configured (see repo Settings → Environments), so the build still runs automatically but publishing waits for an explicit manual approval. This is intentional: backporting — whether it's a documentation fix or a real code fix for a security/critical issue in an old series — is a deliberate, infrequent action, not something that should auto-publish just because a `v*` tag was pushed.

#### Documentation snapshots

Only **final** releases (tags matching `vX.Y.Z` with no pre-release suffix) get a permanent, frozen documentation snapshot, published at `https://shlka.github.io/saealib/vX.Y.Z/`, built by the `deploy-release` job when the tag is pushed. Pre-release (beta) tags do not create a new permanent snapshot. Instead, a single rolling `dev/` build is deployed by the separate `deploy-dev` job on every push to `main` (`.github/workflows/docs.yml`), so it always reflects the latest commit on `main`, not just the latest tagged one. `dev/` is named to make clear it may be ahead of (or otherwise differ from) the latest stable release, rather than being confused for "the latest recommended version." This keeps the number of permanently published doc versions tied to actual releases instead of growing with every beta iteration or every commit. The site root redirects to the newest final release once one exists, or to `dev/` before the first `1.0`-track release ships — until the first tag is pushed, `versions.json` and the root redirect don't exist yet, so `dev/` must be reached directly.

#### Documentation languages

English is the source language for `docs/` (lowers the barrier for outside contributors); Japanese is maintained as a translation via Sphinx's `gettext`/`sphinx-intl` mechanism, not as separate Japanese `.md` files. Every version snapshot above (`vX.Y.Z/`, `dev/`) is built twice — once as-is (English) and once with `-D language=ja` — producing a parallel `ja/` tree (`ja/vX.Y.Z/`, `ja/dev/`). A script injected on every page (`docs/_static/js/lang-switch.js`) links to the equivalent page in the other language, and the site root auto-redirects based on the browser's `navigator.language`, falling back to English for crawlers/no-JS clients.

Only the narrative sections (`getting_started/`, `tutorials/`, `components/`, `architecture/`, and the root `index.md`) are translated. `api/` is autodoc/autosummary-generated from docstrings and is intentionally left English-only — translating ~150 generated stub pages isn't worth the upkeep, and this matches common practice elsewhere (e.g. NumPy/SciPy translate their guides, not their full API reference).

Missing or not-yet-translated strings fall back to the English source automatically (standard `gettext` behavior) — an incomplete `.po` file never breaks the build or leaves a page blank.

To update translations after editing English content in the sections above:

```bash
# 1. Extract translatable strings from the narrative sections into .pot catalogs
uv run --group docs sphinx-build -M gettext docs docs/_build/gettext
rm -rf docs/_build/gettext/gettext/api   # keep api/ untranslated

# 2. Merge changes into the ja catalogs (new strings added, changed ones marked "fuzzy",
#    removed strings dropped; existing translations are preserved)
uv run --group docs sphinx-intl update -p docs/_build/gettext/gettext -l ja -d docs/locale

# 3. Fill in the empty/fuzzy msgstr entries in docs/locale/ja/LC_MESSAGES/**/*.po

# 4. Verify by building the ja variant locally
uv run --group docs sphinx-build -b html -D language=ja docs docs/_build/html-ja
```

### 7. Public API Export Tiers

`src/saealib/__init__.py` exposes public components in three tiers (see the "Export tiers" comment at the top of that file for the authoritative version):

* **Tier 1** (eager import, listed in `__all__`): entry points likely to be named in the first script or a subclass definition — the 5 root abstractions (`Algorithm`, `OptimizationStrategy`, `Surrogate`, `AcquisitionFunction`, `SurrogateManager`), one or two representative default implementations per concept, and the `Comparator`/`Evaluator`/`Initializer`/`Termination`/`Event` bases with their common defaults.
* **Tier 2** (`_TIER2_MAP`, lazy import via `__getattr__`): every other public component. A name in a subpackage's `__all__` belongs here unless it is namespace-only.
* **namespace-only** (not listed at the top level at all): generic-named bulk sets or domain toolkits, e.g. `saealib.benchmarks` (`sphere`/`zdt*`/`dtlz*`/...), `saealib.registry.get`/`build`/`to_spec`, and `saealib.defaults` (internal). Access these via their subpackage directly.

When you add a new public class or function to a subpackage's `__all__`, add it to `_TIER2_MAP` (or the `NAMESPACE_ONLY` allowlist in `tests/test_exports.py` if it's a namespace-only case) and to `src/saealib/__init__.pyi` in the same PR — `tests/test_exports.py` fails otherwise.
