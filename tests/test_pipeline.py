"""Tests for Pipeline, Stage.to_pseudocode(), and Pipeline.__getitem__()."""

import numpy as np
import pytest

from saealib import (
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
)
from saealib.execution.evaluator import SerialEvaluator
from saealib.pipeline import Pipeline
from saealib.stages import (
    ArchiveUpdateStage,
    AskStage,
    CountGenerationStage,
    SurrogateOnlyLoopStage,
    SurrogateScoreStage,
    TellStage,
    TopKSelectionStage,
    TrueEvaluationStage,
)
from saealib.surrogate.prediction import SurrogatePrediction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockSurrogateManager:
    def fit(self, archive, ctx=None):
        pass

    def score_candidates(self, candidates_x, archive, ctx=None, *, refit=True):
        n = len(candidates_x)
        scores = np.linspace(1.0, 0.0, n)
        predictions = [
            SurrogatePrediction(
                value=np.array([[1.0]]), std=None, label=None, metadata={}
            )
            for _ in range(n)
        ]
        return scores, predictions


class _MockProvider:
    def __init__(self):
        self.algorithm = GA(
            crossover=CrossoverBLXAlpha(crossover_rate=0.9, alpha=0.4),
            mutation=MutationUniform(mutation_rate=0.1),
            parent_selection=SequentialSelection(),
            survivor_selection=TruncationSelection(),
        )
        self.surrogate_manager = _MockSurrogateManager()
        self.evaluator = SerialEvaluator()

    def dispatch(self, event):
        pass


def _make_provider():
    return _MockProvider()


def _make_ps_pipeline():
    p = _make_provider()
    return Pipeline(
        [
            CountGenerationStage(),
            AskStage(p.algorithm),
            SurrogateScoreStage(p.surrogate_manager),
            TopKSelectionStage(k=5),
            TrueEvaluationStage(p.evaluator),
            ArchiveUpdateStage(),
            TellStage(p.algorithm),
        ],
        name="ps",
        label="Pre-selection strategy",
    )


def _make_gb_pipeline(gen_ctrl: int = 3):
    p = _make_provider()
    return Pipeline(
        [
            SurrogateOnlyLoopStage(p.algorithm, p.surrogate_manager, gen_ctrl),
            CountGenerationStage(),
            AskStage(p.algorithm),
            TrueEvaluationStage(p.evaluator),
            ArchiveUpdateStage(),
            TellStage(p.algorithm),
        ],
        name="gb",
        label="Generation-based strategy",
    )


# ---------------------------------------------------------------------------
# Pipeline validation
# ---------------------------------------------------------------------------


class TestPipelineValidation:
    def test_non_stage_raises(self):
        with pytest.raises(TypeError, match="not a Stage instance"):
            Pipeline([CountGenerationStage(), "not_a_stage"])  # type: ignore[list-item]

    def test_empty_pipeline_allowed(self):
        p = Pipeline([])
        assert p.stages == []


# ---------------------------------------------------------------------------
# Pipeline.__getitem__
# ---------------------------------------------------------------------------


class TestPipelineGetItem:
    def test_lookup_existing_stage(self):
        p = _make_ps_pipeline()
        stage = p["count_generation"]
        assert isinstance(stage, CountGenerationStage)

    def test_lookup_missing_stage_raises(self):
        p = _make_ps_pipeline()
        with pytest.raises(KeyError):
            p["nonexistent"]

    def test_lookup_all_named_stages(self):
        p = _make_ps_pipeline()
        expected_names = {
            "count_generation",
            "ask",
            "surrogate_score",
            "top_k_selection",
            "true_evaluation",
            "archive_update",
            "tell",
        }
        for name in expected_names:
            assert p[name] is not None


# ---------------------------------------------------------------------------
# Stage.to_pseudocode — leaf stages
# ---------------------------------------------------------------------------


class TestLeafStagePseudocode:
    def test_count_generation_notation(self):
        out = CountGenerationStage().to_pseudocode()
        assert r"\State" in out
        assert "gen" in out

    def test_expand_false_returns_single_line(self):
        stage = CountGenerationStage()
        out = stage.to_pseudocode(expand=False)
        assert "\n" not in out

    def test_expand_true_leaf_returns_single_line(self):
        # Leaf stages have no sub-stages; expand has no effect.
        stage = CountGenerationStage()
        assert stage.to_pseudocode(expand=True) == stage.to_pseudocode(expand=False)

    def test_indent_adds_leading_spaces(self):
        stage = CountGenerationStage()
        out2 = stage.to_pseudocode(indent=2)
        assert out2.startswith("    ")


# ---------------------------------------------------------------------------
# Pipeline.to_pseudocode
# ---------------------------------------------------------------------------


class TestPipelinePseudocode:
    def test_expand_false_one_line(self):
        p = _make_ps_pipeline()
        out = p.to_pseudocode(expand=False)
        assert "\n" not in out
        assert "Pre-selection strategy" in out

    def test_expand_true_has_comment_header(self):
        p = _make_ps_pipeline()
        out = p.to_pseudocode(expand=True)
        assert r"\Comment{Pre-selection strategy}" in out

    def test_expand_true_contains_all_stage_notations(self):
        p = _make_ps_pipeline()
        out = p.to_pseudocode(expand=True)
        assert r"gen \leftarrow gen + 1" in out
        # AskStage with GA expands to \Comment{Generate offspring} + sub-stages
        assert r"Generate offspring" in out
        assert r"\mathrm{select}" in out
        assert r"\mathrm{crossover}" in out
        assert r"\mathrm{mutate}" in out
        assert r"\text{score}" in out
        assert r"\text{top-}" in out
        assert r"\text{eval}" in out

    def test_expand_true_line_count_more_than_stages_when_ask_expands(self):
        p = _make_ps_pipeline()
        lines = p.to_pseudocode(expand=True).splitlines()
        # AskStage expands into sub-stages, so line count > 1 + len(p.stages)
        assert len(lines) > 1 + len(p.stages)

    def test_no_name_falls_back_to_notation(self):
        stage = CountGenerationStage()
        p = Pipeline([stage])
        out = p.to_pseudocode(expand=False)
        assert "Pipeline" in out

    def test_nested_pipeline_expands(self):
        inner = Pipeline([CountGenerationStage()], label="inner")
        outer = Pipeline([inner], label="outer")
        out = outer.to_pseudocode(expand=True)
        assert r"\Comment{inner}" in out
        assert r"gen \leftarrow gen + 1" in out


# ---------------------------------------------------------------------------
# SurrogateOnlyLoopStage.to_pseudocode
# ---------------------------------------------------------------------------


class TestSurrogateOnlyLoopPseudocode:
    def _make(self, gen_ctrl: int = 3):
        p = _make_provider()
        return SurrogateOnlyLoopStage(p.algorithm, p.surrogate_manager, gen_ctrl)

    def test_expand_false_is_single_line(self):
        out = self._make().to_pseudocode(expand=False)
        assert "\n" not in out

    def test_expand_true_emits_for_endfor(self):
        out = self._make().to_pseudocode(expand=True)
        assert r"\For{" in out
        assert r"\EndFor" in out

    def test_expand_true_contains_inner_stages(self):
        out = self._make().to_pseudocode(expand=True)
        assert r"gen \leftarrow gen + 1" in out
        # AskStage with GA expands: \Comment{Generate offspring} + sub-stages
        assert r"Generate offspring" in out
        assert r"\mathrm{select}" in out
        assert r"\text{score}" in out
        assert r"\text{tell}" in out

    def test_gen_ctrl_zero_has_no_for_block(self):
        out = self._make(gen_ctrl=0).to_pseudocode(expand=True)
        assert r"\For{" not in out

    def test_stages_attribute_exposed(self):
        sol = self._make()
        assert sol.stages is not None
        assert len(sol.stages) == 4  # CountGeneration, Ask, Score, Tell

    def test_stages_empty_when_gen_ctrl_zero(self):
        sol = self._make(gen_ctrl=0)
        assert sol.stages == []

    def test_in_gb_pipeline_expand(self):
        pipeline = _make_gb_pipeline(gen_ctrl=2)
        out = pipeline.to_pseudocode(expand=True)
        assert r"\For{" in out
        assert r"\EndFor" in out
        # Outer stages after the loop
        assert r"\mathcal{Q}_{eval} \leftarrow \text{eval}(\mathcal{Q})" in out

    def test_indent_propagates_into_for_block(self):
        sol = self._make()
        out = sol.to_pseudocode(expand=True, indent=1)
        first_line = out.splitlines()[0]
        assert first_line.startswith("  ")
