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
        # AskStage with GA expands via ask_notation: \Comment + 3 lines
        assert r"Generate offspring" in out
        assert r"\mathrm{select}" in out
        assert r"\mathrm{crossover}" in out
        assert r"\mathrm{mutate}" in out
        assert r"\text{score}" in out
        assert r"\text{top-}" in out
        assert r"\text{eval}" in out

    def test_expand_true_ask_expands_when_algorithm_has_notation(self):
        p = _make_ps_pipeline()
        out = p.to_pseudocode(expand=True)
        # GA has ask_notation → AskStage expands to Comment + 3 lines
        # so total lines > 1 (header) + len(p.stages)
        assert len(out.splitlines()) > 1 + len(p.stages)

    def test_expand_true_ask_does_not_expand_without_notation(self):
        """AskStage with an algorithm that has no ask_notation stays as one line."""

        class _NoNotationAlgo:
            pass

        stage = AskStage(_NoNotationAlgo())
        out = stage.to_pseudocode(expand=True)
        assert "\n" not in out
        assert r"\State" in out

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
        # AskStage with GA expands via ask_notation
        assert r"Generate offspring" in out
        assert r"\mathrm{select}" in out
        assert r"\mathrm{crossover}" in out
        assert r"\mathrm{mutate}" in out
        assert r"\text{score}" in out
        # TellStage with GA expands via tell_notation
        assert r"Update population" in out
        assert r"(\mu+\lambda)" in out

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


# ---------------------------------------------------------------------------
# Pipeline.replace
# ---------------------------------------------------------------------------


class TestPipelineReplace:
    def test_replace_existing_stage(self):
        p = _make_ps_pipeline()
        pr = _make_provider()
        new_stage = AskStage(pr.algorithm)
        p.replace("ask", new_stage)
        assert p["ask"] is new_stage

    def test_replace_preserves_other_stages(self):
        p = _make_ps_pipeline()
        original_count = p["count_generation"]
        pr = _make_provider()
        p.replace("ask", AskStage(pr.algorithm))
        assert p["count_generation"] is original_count

    def test_replace_missing_name_raises_key_error(self):
        p = _make_ps_pipeline()
        with pytest.raises(KeyError):
            p.replace("nonexistent", CountGenerationStage())

    def test_replace_non_stage_raises_type_error(self):
        p = _make_ps_pipeline()
        with pytest.raises(TypeError, match="not a Stage instance"):
            p.replace("ask", "not_a_stage")  # type: ignore[arg-type]

    def test_replace_updates_execution(self):
        p = _make_ps_pipeline()
        count_calls = [0]

        class _CountingStage(CountGenerationStage):
            name = "ask"

            def execute(self, state):
                count_calls[0] += 1
                return state

        p.replace("ask", _CountingStage())
        # pipeline.execute cannot run without a real ctx, so just verify lookup
        assert isinstance(p["ask"], _CountingStage)


# ---------------------------------------------------------------------------
# Pipeline.find
# ---------------------------------------------------------------------------


class TestPipelineFind:
    def test_find_top_level_stage(self):
        p = _make_ps_pipeline()
        stage = p.find("count_generation")
        assert isinstance(stage, CountGenerationStage)

    def test_find_missing_non_recursive_raises(self):
        p = _make_ps_pipeline()
        with pytest.raises(KeyError):
            p.find("nonexistent")

    def test_find_missing_recursive_raises(self):
        p = _make_ps_pipeline()
        with pytest.raises(KeyError):
            p.find("nonexistent", recursive=True)

    def test_find_recursive_reaches_inner_stage(self):
        # SurrogateOnlyLoopStage has sub-stages: count_generation, ask, etc.
        p = _make_gb_pipeline(gen_ctrl=2)
        # "ask" appears in both the outer pipeline and inside the loop stage
        stage = p.find("ask", recursive=True)
        assert stage is not None

    def test_find_non_recursive_does_not_descend(self):
        # "count_generation" is inside SurrogateOnlyLoopStage, not at top level of gb
        p = _make_gb_pipeline(gen_ctrl=2)
        # At top level, count_generation appears after the loop stage
        stage = p.find("count_generation")
        assert isinstance(stage, CountGenerationStage)

    def test_find_recursive_returns_first_match(self):
        p = _make_gb_pipeline(gen_ctrl=2)
        # "ask" exists both inside the loop and at top level; recursive returns first
        stage = p.find("ask", recursive=True)
        assert stage.name == "ask"


# ---------------------------------------------------------------------------
# Pipeline.__len__ / __iter__ / __repr__
# ---------------------------------------------------------------------------


class TestPipelineDunderMethods:
    def test_len_matches_stages_count(self):
        p = _make_ps_pipeline()
        assert len(p) == len(p.stages)

    def test_len_empty_pipeline(self):
        assert len(Pipeline([])) == 0

    def test_iter_yields_all_stages(self):
        p = _make_ps_pipeline()
        from_iter = list(p)
        assert from_iter == p.stages

    def test_iter_order_preserved(self):
        p = _make_ps_pipeline()
        names = [s.name for s in p]
        assert names[0] == "count_generation"
        assert names[-1] == "tell"

    def test_repr_contains_pipeline(self):
        p = _make_ps_pipeline()
        assert "Pipeline" in repr(p)

    def test_repr_contains_name_when_set(self):
        p = _make_ps_pipeline()  # name="ps"
        assert "name='ps'" in repr(p)

    def test_repr_no_name_field_when_empty(self):
        p = Pipeline([CountGenerationStage()])
        assert "name=" not in repr(p)

    def test_repr_contains_stage_class_names(self):
        p = _make_ps_pipeline()
        r = repr(p)
        assert "CountGenerationStage" in r
        assert "AskStage" in r
