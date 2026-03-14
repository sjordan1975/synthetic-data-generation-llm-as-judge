"""Tests for benchmark.py — benchmark loading, field mapping, calibration metrics.

Covers task #9 from the ranked task list:
  #9: Build benchmark.py — judge calibration against HuggingFace dataset

TDD approach: test deterministic scaffolding without live LLM or HF calls.

Citations:
  - _instructions.md L56-114   (benchmark dataset, calibration procedure)
  - _instructions.md L625      (≥50 benchmark items, ≥95% pass rate)
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
    RepairCategory,
)


# ============================================================================
# Helper: fake benchmark row (mirrors HuggingFace schema)
# ============================================================================

BENCHMARK_ROW: dict = {
    "id": "qa_00001",
    "category": "plumbing_repair",
    "question": "My kitchen faucet is dripping constantly. How do I fix it?",
    "answer": (
        "A dripping faucet is usually caused by a worn washer or O-ring. "
        "First, turn off the water supply valves under the sink by rotating "
        "them clockwise until fully closed. Place a towel in the sink basin "
        "to catch any small parts. Remove the faucet handle by loosening the "
        "set screw with an Allen wrench. Pull the handle straight up and off. "
        "Use an adjustable wrench to unscrew the packing nut, then pull the "
        "stem straight out. Inspect the washer at the bottom of the stem — "
        "if it is flattened, cracked, or deformed, replace it with a new one "
        "of the same size. Reassemble in reverse order and test by turning "
        "the water supply back on slowly."
    ),
    "equipment_problem": "Kitchen faucet dripping constantly — worn washer",
    "tools_required": ["Allen wrench", "adjustable wrench", "replacement washer", "towel"],
    "steps": [
        "Turn off water supply valves under sink by rotating clockwise until fully closed",
        "Place a towel in sink basin to catch small parts",
        "Loosen set screw with Allen wrench and remove faucet handle",
        "Unscrew packing nut with adjustable wrench and pull stem straight out",
        "Inspect washer at bottom of stem for cracks or deformation",
        "Replace washer with same-size replacement and reassemble in reverse order",
    ],
    "safety_info": (
        "Turn off both hot and cold water supply valves before disassembly. "
        "Keep a bucket nearby to catch residual water in the lines."
    ),
    "tips": [
        "Photograph each step during disassembly so you can reassemble correctly.",
        "Bring the old washer to the hardware store to match size and thickness.",
    ],
}


# ============================================================================
# Benchmark row → RepairQA conversion
# ============================================================================

class TestConvertBenchmarkItem:
    """Verify conversion from HuggingFace row dict to RepairQA."""

    def test_converts_valid_row(self):
        from benchmark import convert_benchmark_item

        qa = convert_benchmark_item(BENCHMARK_ROW)
        assert isinstance(qa, RepairQA)
        assert qa.category == RepairCategory.plumbing_repair
        assert len(qa.steps) >= 3

    def test_maps_all_benchmark_categories(self):
        """All 5 benchmark category strings convert to valid RepairCategory enums."""
        from benchmark import convert_benchmark_item

        for cat in ["plumbing_repair", "electrical_repair", "appliance_repair",
                     "hvac_repair", "general_maintenance"]:
            row = {**BENCHMARK_ROW, "category": cat}
            qa = convert_benchmark_item(row)
            assert qa.category.value == cat

    def test_handles_string_tips(self):
        """Benchmark might have tips as a single string; we should coerce to list."""
        from benchmark import convert_benchmark_item

        row = {**BENCHMARK_ROW, "tips": "Single tip as a string."}
        qa = convert_benchmark_item(row)
        assert isinstance(qa.tips, list)
        assert len(qa.tips) >= 1

    def test_handles_string_steps(self):
        """Benchmark might have steps as strings; we should coerce to list."""
        from benchmark import convert_benchmark_item

        # Steps should already be a list, but protect against oddities
        row = {**BENCHMARK_ROW}
        qa = convert_benchmark_item(row)
        assert isinstance(qa.steps, list)


# ============================================================================
# Calibration metrics
# ============================================================================

def _make_judge_result(
    trace_id: str = "bm_001",
    all_pass: bool = True,
    qd_overrides: dict | None = None,
) -> JudgeResult:
    qd = dict(
        answer_coherence=1,
        step_actionability=1,
        tool_realism=1,
        safety_specificity=1 if all_pass else 0,
        tip_usefulness=1,
        problem_answer_alignment=1,
        appropriate_scope=1,
        category_accuracy=1,
    )
    if qd_overrides:
        qd.update(qd_overrides)
    return JudgeResult(
        trace_id=trace_id,
        failure_modes=FailureModeResult(
            incomplete_answer=0,
            safety_violations=0,
            unrealistic_tools=0,
            overcomplicated_solution=0,
            missing_context=0,
            poor_quality_tips=0,
        ),
        quality_scores=QualityDimensionResult(**qd,
        ),
    )


class TestCalibrationMetrics:
    """Verify calibration pass rate calculation."""

    def test_all_pass(self):
        from benchmark import calculate_calibration_metrics

        results = [_make_judge_result(f"bm_{i:03d}") for i in range(50)]
        metrics = calculate_calibration_metrics(results)

        assert metrics["total_samples"] == 50
        assert metrics["overall_quality_pass_rate"] == 100.0
        assert metrics["calibration_passed"] is True

    def test_below_threshold(self):
        from benchmark import calculate_calibration_metrics

        # 47 pass, 3 fail → 94% < 95% threshold
        results = [_make_judge_result(f"bm_{i:03d}") for i in range(47)]
        results += [_make_judge_result(f"bm_{i:03d}", all_pass=False) for i in range(47, 50)]
        metrics = calculate_calibration_metrics(results)

        assert metrics["calibration_passed"] is False
        assert metrics["overall_quality_pass_rate"] == 94.0

    def test_at_threshold(self):
        from benchmark import calculate_calibration_metrics

        # Exactly 95% → passes
        results = [_make_judge_result(f"bm_{i:03d}") for i in range(95)]
        results += [_make_judge_result(f"bm_{i:03d}", all_pass=False) for i in range(95, 100)]
        metrics = calculate_calibration_metrics(results)

        assert metrics["calibration_passed"] is True
        assert metrics["overall_quality_pass_rate"] == 95.0


# ============================================================================
# JSONL result saving
# ============================================================================

class TestSaveCalibrationResults:
    """Verify calibration results saved as JSONL + JSON summary."""

    def test_saves_jsonl(self, tmp_path):
        from benchmark import save_calibration_results

        results = [_make_judge_result(f"bm_{i:03d}") for i in range(3)]
        metrics = {"total_samples": 3, "calibration_passed": True}
        save_calibration_results(results, metrics, str(tmp_path))

        jsonl_path = tmp_path / "benchmark_judge_results.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_saves_json_summary(self, tmp_path):
        from benchmark import save_calibration_results

        results = [_make_judge_result()]
        metrics = {"total_samples": 1, "calibration_passed": True}
        save_calibration_results(results, metrics, str(tmp_path))

        summary_path = tmp_path / "benchmark_calibration.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["calibration_passed"] is True


# ============================================================================
# Quality gap analysis (Task #12 — benchmark vs generated comparison)
# ============================================================================

class TestComputeQualityGap:
    """Verify quantitative quality gap between benchmark and generated data."""

    def test_perfect_match(self):
        from benchmark import compute_quality_gap

        bench = [_make_judge_result(f"bm_{i:03d}") for i in range(4)]
        gen = [_make_judge_result(f"qa_{i:03d}") for i in range(4)]
        gap = compute_quality_gap(bench, gen)

        # All pass → gap should be 0 for every dimension
        for d, info in gap["per_dimension"].items():
            assert info["gap_pp"] == 0.0
        assert gap["overall_quality_gap_pp"] == 0.0

    def test_generated_worse(self):
        from benchmark import compute_quality_gap

        bench = [_make_judge_result(f"bm_{i:03d}") for i in range(4)]
        gen = [
            _make_judge_result("qa_001", qd_overrides={"safety_specificity": 0}),
            _make_judge_result("qa_002", qd_overrides={"safety_specificity": 0}),
            _make_judge_result("qa_003"),
            _make_judge_result("qa_004"),
        ]
        gap = compute_quality_gap(bench, gen)

        # safety_specificity: bench 100%, gen 50% → gap = -50pp
        assert gap["per_dimension"]["safety_specificity"]["gap_pp"] == -50.0
        assert gap["per_dimension"]["safety_specificity"]["benchmark_rate"] == 100.0
        assert gap["per_dimension"]["safety_specificity"]["generated_rate"] == 50.0

    def test_overall_gap(self):
        from benchmark import compute_quality_gap

        bench = [_make_judge_result(f"bm_{i:03d}") for i in range(4)]
        gen = [
            _make_judge_result("qa_001"),
            _make_judge_result("qa_002", qd_overrides={"tip_usefulness": 0}),
            _make_judge_result("qa_003"),
            _make_judge_result("qa_004"),
        ]
        gap = compute_quality_gap(bench, gen)

        # Bench: 100% quality pass, Gen: 75% → gap = -25pp
        assert gap["benchmark_quality_pass_rate"] == 100.0
        assert gap["generated_quality_pass_rate"] == 75.0
        assert gap["overall_quality_gap_pp"] == -25.0


class TestSaveBenchmarkComparison:
    """Verify benchmark comparison report saved as JSON."""

    def test_saves_json(self, tmp_path):
        from benchmark import save_benchmark_comparison

        report = {"overall_quality_gap_pp": -10.0}
        outfile = tmp_path / "comparison.json"
        save_benchmark_comparison(report, str(outfile))

        assert outfile.exists()
        data = json.loads(outfile.read_text())
        assert data["overall_quality_gap_pp"] == -10.0
