"""Tests for analyzer.py — JSONL loading, metrics, co-occurrence, category breakdown.

Covers task #10 from the ranked task list:
  #10: Update analyzer.py — heatmaps, quality charts, category breakdown

TDD approach: test deterministic data transforms; skip visual output validation.

Citations:
  - _instructions.md L362-374  (failure pattern analysis & heatmap)
  - _instructions.md L416-428  (visualization requirements)
  - _instructions.md L394-404  (failure analysis report deliverables)
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    FailureModeResult,
    JudgeResult,
    QualityDimensionResult,
    RepairQA,
)
from tests.conftest import VALID_REPAIR_QA_DATA


# ============================================================================
# Helpers
# ============================================================================

def _make_judge_result(
    trace_id: str = "qa_001",
    fm_overrides: dict | None = None,
    qd_overrides: dict | None = None,
) -> JudgeResult:
    """Build a JudgeResult with optional per-field overrides."""
    fm_defaults = dict(
        incomplete_answer=0, safety_violations=0, unrealistic_tools=0,
        overcomplicated_solution=0, missing_context=0, poor_quality_tips=0,
    )
    qd_defaults = dict(
        answer_coherence=1, step_actionability=1, tool_realism=1,
        safety_specificity=1, tip_usefulness=1, problem_answer_alignment=1,
        appropriate_scope=1, category_accuracy=1,
    )
    if fm_overrides:
        fm_defaults.update(fm_overrides)
    if qd_overrides:
        qd_defaults.update(qd_overrides)

    return JudgeResult(
        trace_id=trace_id,
        failure_modes=FailureModeResult(**fm_defaults),
        quality_scores=QualityDimensionResult(**qd_defaults),
    )


def _write_judge_jsonl(path: Path, results: list[JudgeResult]) -> None:
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.model_dump(mode="json")) + "\n")


def _write_validated_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ============================================================================
# Loading judge results from JSONL
# ============================================================================

class TestLoadJudgeResults:
    """Verify loading JudgeResult objects from JSONL."""

    def test_loads_valid_jsonl(self, tmp_path):
        from analyzer import load_judge_results

        results = [_make_judge_result(f"qa_{i:03d}") for i in range(5)]
        fpath = tmp_path / "results.jsonl"
        _write_judge_jsonl(fpath, results)

        loaded = load_judge_results(str(fpath))
        assert len(loaded) == 5
        assert all(isinstance(r, JudgeResult) for r in loaded)

    def test_raises_on_missing_file(self, tmp_path):
        from analyzer import load_judge_results

        with pytest.raises(FileNotFoundError):
            load_judge_results(str(tmp_path / "nope.jsonl"))


# ============================================================================
# DataFrame construction (join results + categories)
# ============================================================================

class TestBuildAnalysisDataFrame:
    """Verify DataFrame built from judge results + categories."""

    def test_has_expected_columns(self, tmp_path):
        from analyzer import build_analysis_dataframe

        results = [
            _make_judge_result("qa_001"),
            _make_judge_result("qa_002", fm_overrides={"safety_violations": 1}),
        ]
        categories = ["plumbing_repair", "electrical_repair"]

        df = build_analysis_dataframe(results, categories)

        # Must have trace_id, category, all 6 FM columns, all 8 QD columns
        assert "trace_id" in df.columns
        assert "category" in df.columns
        assert "incomplete_answer" in df.columns
        assert "answer_coherence" in df.columns
        assert len(df) == 2

    def test_failure_values_correct(self, tmp_path):
        from analyzer import build_analysis_dataframe

        results = [
            _make_judge_result("qa_001", fm_overrides={"missing_context": 1}),
        ]
        df = build_analysis_dataframe(results, ["plumbing_repair"])

        assert df.iloc[0]["missing_context"] == 1
        assert df.iloc[0]["incomplete_answer"] == 0


# ============================================================================
# Failure co-occurrence matrix
# ============================================================================

class TestFailureCooccurrence:
    """Verify failure mode co-occurrence (correlation) computation."""

    def test_cooccurrence_shape(self):
        from analyzer import compute_failure_cooccurrence, build_analysis_dataframe

        results = [
            _make_judge_result("qa_001", fm_overrides={"incomplete_answer": 1, "missing_context": 1}),
            _make_judge_result("qa_002", fm_overrides={"safety_violations": 1}),
            _make_judge_result("qa_003"),
        ]
        df = build_analysis_dataframe(results, ["plumbing_repair"] * 3)
        cooc = compute_failure_cooccurrence(df)

        assert cooc.shape == (6, 6)
        # Diagonal should be total counts of that failure mode
        assert cooc.loc["incomplete_answer", "incomplete_answer"] == 1

    def test_cooccurrence_symmetric(self):
        from analyzer import compute_failure_cooccurrence, build_analysis_dataframe

        results = [
            _make_judge_result("qa_001", fm_overrides={"incomplete_answer": 1, "missing_context": 1}),
            _make_judge_result("qa_002"),
        ]
        df = build_analysis_dataframe(results, ["plumbing_repair"] * 2)
        cooc = compute_failure_cooccurrence(df)

        assert cooc.loc["incomplete_answer", "missing_context"] == cooc.loc["missing_context", "incomplete_answer"]


# ============================================================================
# Category-level failure rates
# ============================================================================

class TestCategoryFailureRates:
    """Verify per-category failure rate aggregation."""

    def test_rates_per_category(self):
        from analyzer import compute_category_failure_rates, build_analysis_dataframe

        results = [
            _make_judge_result("qa_001", fm_overrides={"incomplete_answer": 1}),
            _make_judge_result("qa_002"),
            _make_judge_result("qa_003", fm_overrides={"safety_violations": 1}),
            _make_judge_result("qa_004"),
        ]
        categories = ["plumbing_repair", "plumbing_repair", "electrical_repair", "electrical_repair"]
        df = build_analysis_dataframe(results, categories)
        rates = compute_category_failure_rates(df)

        assert "plumbing_repair" in rates
        assert "electrical_repair" in rates
        # plumbing: 1/2 = 50%, electrical: 1/2 = 50%
        assert rates["plumbing_repair"]["overall_failure_rate"] == 50.0
        assert rates["electrical_repair"]["overall_failure_rate"] == 50.0


# ============================================================================
# Quality dimension summary
# ============================================================================

class TestQualityDimensionSummary:
    """Verify quality dimension pass rate aggregation."""

    def test_all_pass(self):
        from analyzer import compute_quality_summary, build_analysis_dataframe

        results = [_make_judge_result(f"qa_{i:03d}") for i in range(4)]
        df = build_analysis_dataframe(results, ["plumbing_repair"] * 4)
        summary = compute_quality_summary(df)

        assert summary["answer_coherence"]["pass_rate"] == 100.0
        assert summary["overall_quality_pass_rate"] == 100.0

    def test_mixed_results(self):
        from analyzer import compute_quality_summary, build_analysis_dataframe

        results = [
            _make_judge_result("qa_001"),
            _make_judge_result("qa_002", qd_overrides={"safety_specificity": 0}),
        ]
        df = build_analysis_dataframe(results, ["plumbing_repair"] * 2)
        summary = compute_quality_summary(df)

        assert summary["safety_specificity"]["pass_rate"] == 50.0
        # Only 1 of 2 items passes all 8 → 50%
        assert summary["overall_quality_pass_rate"] == 50.0


# ============================================================================
# Most problematic items
# ============================================================================

class TestMostProblematicItems:
    """Identify items with 3+ failure flags (_instructions.md L426)."""

    def test_finds_multi_failure_items(self):
        from analyzer import find_most_problematic, build_analysis_dataframe

        results = [
            _make_judge_result("qa_001", fm_overrides={
                "incomplete_answer": 1, "missing_context": 1, "poor_quality_tips": 1,
            }),
            _make_judge_result("qa_002"),
            _make_judge_result("qa_003", fm_overrides={"safety_violations": 1}),
        ]
        df = build_analysis_dataframe(results, ["plumbing_repair"] * 3)
        problematic = find_most_problematic(df, min_failures=3)

        assert len(problematic) == 1
        assert problematic[0]["trace_id"] == "qa_001"
        assert problematic[0]["failure_count"] == 3


# ============================================================================
# Analysis report (JSON)
# ============================================================================

class TestSaveAnalysisReport:
    """Verify analysis report saved as JSON."""

    def test_saves_json(self, tmp_path):
        from analyzer import save_analysis_report

        report = {"total_samples": 5, "overall_failure_rate": 40.0}
        outfile = tmp_path / "report.json"
        save_analysis_report(report, str(outfile))

        assert outfile.exists()
        data = json.loads(outfile.read_text())
        assert data["total_samples"] == 5
