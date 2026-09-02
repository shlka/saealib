#!/usr/bin/env python3
"""Regenerate the built-in Stage contract table in every language tree."""

from pathlib import Path

from saealib._stage_docs import HEADERS, update_stage_docs

PAGE = "concepts/observation_and_state/stage.md"

if __name__ == "__main__":
    docs = Path(__file__).parents[1] / "docs"
    for language, headers in HEADERS.items():
        update_stage_docs(docs / language / PAGE, headers)
