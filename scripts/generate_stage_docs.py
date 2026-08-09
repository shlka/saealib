#!/usr/bin/env python3
"""Regenerate the built-in Stage contract table."""

from pathlib import Path

from saealib._stage_docs import update_stage_docs

if __name__ == "__main__":
    update_stage_docs(Path(__file__).parents[1] / "docs/components/stage.md")
