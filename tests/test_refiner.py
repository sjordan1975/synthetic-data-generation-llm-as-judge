"""Tests for refiner.py — deterministic scaffolding for prompt correction workflow.

Covers task #11 from the ranked task list:
  #11: Rewrite refiner.py — targeted prompt correction + before/after comparison

TDD approach: test comparison logic, delta computation, report generation.
Prompt correction strategy is deferred until pipeline produces real failure data.

Citations:
  - _instructions.md L376-380  (prompt correction strategy)
  - _instructions.md L406-408  (corrected prompts + before/after comparison)
  - _instructions.md L424      (per-mode failure trend before vs after)
  - _instructions.md L615-621  (>80% failure reduction, ≥80% quality pass)
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import FailureModeResult, JudgeResult, QualityDimensionResult
from labeler import FAILURE_MODE_FIELDS, QUALITY_DIM_FIELDS


# ============================================================================
# Helpers
# ============================================================================

def _make_judge_result(
    trace_id: str = "qa_001",
    fm_overrides: dict | None = None,
    qd_overrides: dict | None = None,
) -> JudgeResult:
    fm = dict(
        incomplete_answer=0, safety_violations=0, unrealistic_tools=0,
        overcomplicated_solution=0, missing_context=0, poor_quality_tips=0,
    )
    qd = dict(
        answer_coherence=1, step_actionability=1, tool_realism=1,
        safety_specificity=1, tip_usefulness=1, problem_answer_alignment=1,
        appropriate_scope=1, category_accuracy=1,
    )
    if fm_overrides:
        fm.update(fm_overrides)
    if qd_overrides:
        qd.update(qd_overrides)
    return JudgeResult(
        trace_id=trace_id,
        failure_modes=FailureModeResult(**fm),
        quality_scores=QualityDimensionResult(**qd),
    )


# ============================================================================
# Load analysis report
# ============================================================================

class TestLoadAnalysisReport:

    def test_loads_valid_report(self, tmp_path):
        from refiner import load_analysis_report

        report = {"total_samples": 50, "overall_failure_rate": 30.0}
        fpath = tmp_path / "report.json"
        fpath.write_text(json.dumps(report))
        loaded = load_analysis_report(str(fpath))
        assert loaded["total_samples"] == 50

    def test_raises_on_missing(self, tmp_path):
        from refiner import load_analysis_report

        with pytest.raises(FileNotFoundError):
            load_analysis_report(str(tmp_path / "nope.json"))


# ============================================================================
# Before/after comparison
# ============================================================================

class TestComputeComparison:
    """Verify before/after delta computation across FM + QD."""

    def test_improvement_calculated(self):
        from refiner import compute_before_after

        # Baseline: 50% failure, 50% quality pass
        baseline = [
            _make_judge_result("b_001", fm_overrides={"incomplete_answer": 1}),
            _make_judge_result("b_002"),
        ]
        # Corrected: 0% failure, 100% quality pass
        corrected = [
            _make_judge_result("c_001"),
            _make_judge_result("c_002"),
        ]
        comparison = compute_before_after(baseline, corrected)

        assert comparison["baseline"]["overall_failure_rate"] == 50.0
        assert comparison["corrected"]["overall_failure_rate"] == 0.0
        assert comparison["failure_reduction_pct"] == 100.0
        assert comparison["reduction_target_met"] is True  # >80%

    def test_no_improvement(self):
        from refiner import compute_before_after

        same = [
            _make_judge_result("x_001", fm_overrides={"safety_violations": 1}),
            _make_judge_result("x_002"),
        ]
        comparison = compute_before_after(same, same)

        assert comparison["failure_reduction_pct"] == 0.0
        assert comparison["reduction_target_met"] is False

    def test_per_mode_deltas(self):
        from refiner import compute_before_after

        baseline = [
            _make_judge_result("b_001", fm_overrides={"incomplete_answer": 1, "missing_context": 1}),
            _make_judge_result("b_002", fm_overrides={"incomplete_answer": 1}),
        ]
        corrected = [
            _make_judge_result("c_001"),
            _make_judge_result("c_002", fm_overrides={"incomplete_answer": 1}),
        ]
        comparison = compute_before_after(baseline, corrected)

        fm_deltas = comparison["failure_mode_deltas"]
        # incomplete_answer: 100% → 50% = -50pp
        assert fm_deltas["incomplete_answer"]["baseline_rate"] == 100.0
        assert fm_deltas["incomplete_answer"]["corrected_rate"] == 50.0
        # missing_context: 50% → 0% = -50pp
        assert fm_deltas["missing_context"]["corrected_rate"] == 0.0

    def test_per_dimension_deltas(self):
        from refiner import compute_before_after

        baseline = [
            _make_judge_result("b_001", qd_overrides={"safety_specificity": 0}),
            _make_judge_result("b_002"),
        ]
        corrected = [
            _make_judge_result("c_001"),
            _make_judge_result("c_002"),
        ]
        comparison = compute_before_after(baseline, corrected)

        qd_deltas = comparison["quality_dimension_deltas"]
        assert qd_deltas["safety_specificity"]["baseline_rate"] == 50.0
        assert qd_deltas["safety_specificity"]["corrected_rate"] == 100.0

    def test_quality_pass_rate(self):
        from refiner import compute_before_after

        baseline = [
            _make_judge_result("b_001", qd_overrides={"tip_usefulness": 0}),
            _make_judge_result("b_002", qd_overrides={"tip_usefulness": 0}),
        ]
        corrected = [
            _make_judge_result("c_001"),
            _make_judge_result("c_002", qd_overrides={"tip_usefulness": 0}),
        ]
        comparison = compute_before_after(baseline, corrected)

        assert comparison["baseline"]["overall_quality_pass_rate"] == 0.0
        assert comparison["corrected"]["overall_quality_pass_rate"] == 50.0
        assert comparison["quality_target_met"] is False  # need ≥80%


# ============================================================================
# Identify weakest areas
# ============================================================================

class TestIdentifyWeakest:
    """Find the failure modes and quality dims most in need of correction."""

    def test_finds_top_failure_modes(self):
        from refiner import identify_weakest_areas

        results = [
            _make_judge_result("q_001", fm_overrides={"incomplete_answer": 1, "poor_quality_tips": 1}),
            _make_judge_result("q_002", fm_overrides={"incomplete_answer": 1}),
            _make_judge_result("q_003"),
        ]
        weak = identify_weakest_areas(results)

        # incomplete_answer at 66.7%, poor_quality_tips at 33.3%
        assert weak["failure_modes"][0]["mode"] == "incomplete_answer"
        assert weak["failure_modes"][0]["rate"] > 60

    def test_finds_weakest_quality_dims(self):
        from refiner import identify_weakest_areas

        results = [
            _make_judge_result("q_001", qd_overrides={"tip_usefulness": 0, "safety_specificity": 0}),
            _make_judge_result("q_002", qd_overrides={"tip_usefulness": 0}),
            _make_judge_result("q_003"),
        ]
        weak = identify_weakest_areas(results)

        # tip_usefulness at 33.3%, safety_specificity at 66.7%
        assert weak["quality_dimensions"][0]["dimension"] == "tip_usefulness"


# ============================================================================
# Save comparison report
# ============================================================================

class TestSaveComparisonReport:

    def test_saves_json(self, tmp_path):
        from refiner import save_comparison_report

        report = {"baseline": {}, "corrected": {}, "reduction_target_met": True}
        outfile = tmp_path / "comparison.json"
        save_comparison_report(report, str(outfile))

        assert outfile.exists()
        data = json.loads(outfile.read_text())
        assert data["reduction_target_met"] is True
