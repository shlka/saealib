#!/usr/bin/env python3
"""Check that every documentation page exists in all language trees.

The language switcher (``docs/_static/js/lang-switch.js``) rewrites the current
URL path rather than looking a page up, so a page missing from one tree becomes a
404 on the other tree's switcher. Parity therefore admits no exclusions: a page
that is not translated yet still needs a stub carrying a link to the original.
"""

from __future__ import annotations

import sys
from pathlib import Path

DOCS = Path(__file__).parents[1] / "docs"
LANGUAGES = ("en", "ja")


def pages(language: str) -> set[str]:
    """Return every documented page path under one language tree."""
    root = DOCS / language
    return {
        str(path.relative_to(root))
        for path in root.rglob("*.md")
        if "_autosummary" not in path.parts and path.name != "_page_template.md"
    }


def main() -> int:
    """Report pages missing from any language tree."""
    found = {language: pages(language) for language in LANGUAGES}
    shared = set.intersection(*found.values())

    failed = False
    for language, present in found.items():
        for other in LANGUAGES:
            if other == language:
                continue
            missing = sorted(found[other] - present)
            if missing:
                failed = True
                print(f"Missing {language} documentation ({len(missing)}):")
                for name in missing:
                    print(f"  docs/{language}/{name}")

    if not failed:
        print(f"Language parity OK: {len(shared)} pages in {', '.join(LANGUAGES)}.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
