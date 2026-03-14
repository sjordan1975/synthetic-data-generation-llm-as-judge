"""Tests for labeler.py — judge prompt construction, JSONL I/O, metrics.

Covers task #8 from the ranked task list:
  #8: Rewrite labeler.py — LLM-as-Judge with 6 failure modes + 8 quality dims

TDD approach: test deterministic scaffolding without live LLM calls.

Citations:
  - _instructions.md L315-358  (judge output: 6 failure modes + 8 quality dims)
  - _instructions.md L165-174  (8 quality dimension definitions)
  - _instructions.md L319-326  (6 failure mode definitions)
  - _instructions.md L570-574  (JSONL storage)
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
# Helper: build JSONL test files
# ============================================================================

def _write_validated_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ============================================================================
# Prompt construction
# ============================================================================

class TestBuildJudgePrompt:
    """Verify the judge prompt includes all required evaluation criteria."""

    def test_prompt_contains_all_failure_modes(self, valid_repair_qa):
        from labeler import build_judge_prompt

        prompt = build_judge_prompt(valid_repair_qa, trace_id="qa_001")

        for mode in [
            "incomplete_answer",
            "safety_violations",
            "unrealistic_tools",
            "overcomplicated_solution",
            "missing_context",
            "poor_quality_tips",
        ]:
            assert mode in prompt, f"Failure mode '{mode}' missing from prompt"

    def test_prompt_contains_all_quality_dimensions(self, valid_repair_qa):
        from labeler import build_judge_prompt

        prompt = build_judge_prompt(valid_repair_qa, trace_id="qa_001")

        for dim in [
            "answer_coherence",
            "step_actionability",
            "tool_realism",
            "safety_specificity",
            "tip_usefulness",
            "problem_answer_alignment",
            "appropriate_scope",
            "category_accuracy",
        ]:
            assert dim in prompt, f"Quality dimension '{dim}' missing from prompt"

    def test_prompt_includes_sample_content(self, valid_repair_qa):
        from labeler import build_judge_prompt

        prompt = build_judge_prompt(valid_repair_qa, trace_id="qa_001")

        assert valid_repair_qa.question in prompt
        assert valid_repair_qa.equipment_problem in prompt
        assert valid_repair_qa.category.value in prompt

    def test_prompt_renders_tips_as_list(self, valid_repair_qa):
        """tips is now List[str], prompt should render each tip."""
        from labeler import build_judge_prompt

        prompt = build_judge_prompt(valid_repair_qa, trace_id="qa_001")

        for tip in valid_repair_qa.tips:
            assert tip in prompt

    def test_prompt_includes_trace_id(self, valid_repair_qa):
        from labeler import build_judge_prompt

        prompt = build_judge_prompt(valid_repair_qa, trace_id="qa_042")
        assert "qa_042" in prompt


# ============================================================================
# JSONL data loading
# ============================================================================

class TestLoadValidatedData:
    """Verify load_validated_data reads JSONL and returns RepairQA objects."""

    def test_loads_valid_jsonl(self, tmp_path):
        from labeler import load_validated_data

        fpath = tmp_path / "validated.jsonl"
        _write_validated_jsonl(fpath, [VALID_REPAIR_QA_DATA] * 3)

        samples = load_validated_data(str(fpath))
        assert len(samples) == 3
        assert all(isinstance(s, RepairQA) for s in samples)

    def test_raises_on_missing_file(self, tmp_path):
        from labeler import load_validated_data

        with pytest.raises(FileNotFoundError):
            load_validated_data(str(tmp_path / "nope.jsonl"))


# ============================================================================
# Result saving (JSONL)
# ============================================================================

def _make_judge_result(trace_id: str = "qa_001", has_failure: bool = False) -> JudgeResult:
    """Build a JudgeResult for testing."""
    return JudgeResult(
        trace_id=trace_id,
        failure_modes=FailureModeResult(
            incomplete_answer=1 if has_failure else 0,
            safety_violations=0,
            unrealistic_tools=0,
            overcomplicated_solution=0,
            missing_context=0,
            poor_quality_tips=0,
        ),
        quality_scores=QualityDimensionResult(
            answer_coherence=1,
            step_actionability=1,
            tool_realism=1,
            safety_specificity=1,
            tip_usefulness=1,
            problem_answer_alignment=1,
            appropriate_scope=1,
            category_accuracy=1,
        ),
    )


class TestSaveJudgeResults:
    """Verify save_judge_results writes JSONL."""

    def test_writes_jsonl(self, tmp_path):
        from labeler import save_judge_results

        results = [_make_judge_result("qa_001"), _make_judge_result("qa_002", has_failure=True)]
        outfile = tmp_path / "judge_results.jsonl"
        save_judge_results(results, str(outfile))

        lines = outfile.read_text().strip().splitlines()
        assert len(lines) == 2

        r1 = json.loads(lines[0])
        assert r1["trace_id"] == "qa_001"
        assert r1["failure_modes"]["incomplete_answer"] == 0

        r2 = json.loads(lines[1])
        assert r2["trace_id"] == "qa_002"
        assert r2["failure_modes"]["incomplete_answer"] == 1

    def test_creates_parent_dirs(self, tmp_path):
        from labeler import save_judge_results

        outfile = tmp_path / "nested" / "results.jsonl"
        save_judge_results([_make_judge_result()], str(outfile))
        assert outfile.exists()


# ============================================================================
# Baseline metrics calculation
# ============================================================================

class TestCalculateBaselineMetrics:
    """Verify metrics aggregation from judge results."""

    def test_all_passing(self):
        from labeler import calculate_baseline_metrics

        results = [_make_judge_result(f"qa_{i:03d}") for i in range(5)]
        metrics = calculate_baseline_metrics(results)

        assert metrics["total_samples"] == 5
        assert metrics["overall_failure_rate"] == 0.0
        assert metrics["overall_quality_pass_rate"] == 100.0

    def test_mixed_results(self):
        from labeler import calculate_baseline_metrics

        results = [
            _make_judge_result("qa_001"),                       # all pass
            _make_judge_result("qa_002", has_failure=True),     # one failure
            _make_judge_result("qa_003"),                       # all pass
        ]
        metrics = calculate_baseline_metrics(results)

        assert metrics["total_samples"] == 3
        assert metrics["samples_with_failures"] == 1
        # 1/3 ≈ 33.3%
        assert 33.0 <= metrics["overall_failure_rate"] <= 34.0

    def test_per_mode_rates(self):
        from labeler import calculate_baseline_metrics

        results = [
            _make_judge_result("qa_001", has_failure=True),     # incomplete_answer=1
            _make_judge_result("qa_002", has_failure=True),
            _make_judge_result("qa_003"),
        ]
        metrics = calculate_baseline_metrics(results)

        assert metrics["failure_modes"]["incomplete_answer"]["count"] == 2
        assert metrics["failure_modes"]["safety_violations"]["count"] == 0
