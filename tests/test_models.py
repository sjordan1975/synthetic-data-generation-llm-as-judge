"""Tests for Pydantic models — RepairQA, GenerationMeta, JudgeResult.

Covers tasks #1–4 from the ranked task list:
  #1: Fix RepairQA tips→List[str], steps min→3
  #2: Add category field to RepairQA
  #3: Add GenerationMeta model
  #4: Add judge output models (FailureModeResult, QualityDimensionResult, JudgeResult)

Citations:
  - _instructions.md L238-248  (7 required fields, tips as list of strings)
  - _instructions.md L271-281  (validation rules: steps≥3, tools≥1, tips≥1)
  - _instructions.md L283-291  (generation metadata)
  - _instructions.md L315-358  (judge output: 6 failure modes + 8 quality dims)
"""

import pytest
from pydantic import ValidationError

from models import RepairQA, GenerationMeta, FailureModeResult, QualityDimensionResult, JudgeResult


# ============================================================================
# Task #1: RepairQA field constraints
# ============================================================================

class TestRepairQAFieldConstraints:
    """Verify RepairQA enforces spec constraints (L271-281)."""

    def test_valid_repair_qa(self, valid_repair_qa):
        """Happy path: valid data produces a valid model."""
        assert valid_repair_qa.question.startswith("How do I")
        assert len(valid_repair_qa.steps) >= 3
        assert len(valid_repair_qa.tools_required) >= 1
        assert len(valid_repair_qa.tips) >= 1

    # --- tips must be List[str] (L248) ---

    def test_tips_is_list_of_strings(self, valid_repair_qa):
        """tips field must be a list, not a plain string."""
        assert isinstance(valid_repair_qa.tips, list)
        assert all(isinstance(t, str) for t in valid_repair_qa.tips)

    def test_tips_rejects_plain_string(self, valid_repair_qa_data):
        """A plain string for tips must be rejected."""
        valid_repair_qa_data["tips"] = "Just a plain string tip"
        with pytest.raises(ValidationError):
            RepairQA(**valid_repair_qa_data)

    def test_tips_rejects_empty_list(self, valid_repair_qa_data):
        """tips must contain at least 1 item (L279)."""
        valid_repair_qa_data["tips"] = []
        with pytest.raises(ValidationError):
            RepairQA(**valid_repair_qa_data)

    # --- steps must have ≥3 items (L275) ---

    def test_steps_rejects_fewer_than_three(self, valid_repair_qa_data):
        """steps must contain at least 3 items."""
        valid_repair_qa_data["steps"] = ["Step one", "Step two"]
        with pytest.raises(ValidationError):
            RepairQA(**valid_repair_qa_data)

    def test_steps_accepts_exactly_three(self, valid_repair_qa_data):
        """Boundary: exactly 3 steps should pass."""
        valid_repair_qa_data["steps"] = ["Step one", "Step two", "Step three"]
        qa = RepairQA(**valid_repair_qa_data)
        assert len(qa.steps) == 3

    # --- tools_required must have ≥1 item (L277) ---

    def test_tools_rejects_empty_list(self, valid_repair_qa_data):
        """tools_required must contain at least 1 item."""
        valid_repair_qa_data["tools_required"] = []
        with pytest.raises(ValidationError):
            RepairQA(**valid_repair_qa_data)

    # --- question and answer must be non-empty (L273) ---

    def test_question_rejects_empty_string(self, valid_repair_qa_data):
        """question must be non-empty."""
        valid_repair_qa_data["question"] = ""
        with pytest.raises(ValidationError):
            RepairQA(**valid_repair_qa_data)

    def test_answer_rejects_empty_string(self, valid_repair_qa_data):
        """answer must be non-empty."""
        valid_repair_qa_data["answer"] = ""
        with pytest.raises(ValidationError):
            RepairQA(**valid_repair_qa_data)

    # --- safety_info must be present and non-empty (L281) ---

    def test_safety_info_rejects_empty_string(self, valid_repair_qa_data):
        """safety_info must not be empty."""
        valid_repair_qa_data["safety_info"] = ""
        with pytest.raises(ValidationError):
            RepairQA(**valid_repair_qa_data)


# ============================================================================
# Task #2: category field
# ============================================================================

VALID_CATEGORIES = [
    "appliance_repair",
    "plumbing_repair",
    "electrical_repair",
    "hvac_repair",
    "general_maintenance",
]


class TestRepairQACategory:
    """Verify RepairQA has a category field matching the 5 repair domains (L299-309)."""

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_accepts_valid_categories(self, valid_repair_qa_data, category):
        valid_repair_qa_data["category"] = category
        qa = RepairQA(**valid_repair_qa_data)
        assert qa.category == category

    def test_rejects_unknown_category(self, valid_repair_qa_data):
        valid_repair_qa_data["category"] = "automotive_repair"
        with pytest.raises(ValidationError):
            RepairQA(**valid_repair_qa_data)

    def test_category_is_required(self, valid_repair_qa_data):
        del valid_repair_qa_data["category"]
        with pytest.raises(ValidationError):
            RepairQA(**valid_repair_qa_data)


# ============================================================================
# Task #3: GenerationMeta model
# ============================================================================

class TestGenerationMeta:
    """Verify generation metadata model tracks per-item provenance (L283-291)."""

    def test_valid_generation_meta(self):
        meta = GenerationMeta(
            prompt_template="plumbing_repair",
            model="gpt-4o-mini",
        )
        assert meta.prompt_template == "plumbing_repair"
        assert meta.model == "gpt-4o-mini"
        assert meta.timestamp is not None  # auto-populated
        assert meta.validation_passed is None  # optional, not yet known

    def test_validation_passed_can_be_set(self):
        meta = GenerationMeta(
            prompt_template="electrical_repair",
            model="gpt-4o-mini",
            validation_passed=True,
        )
        assert meta.validation_passed is True

    def test_prompt_template_required(self):
        with pytest.raises(ValidationError):
            GenerationMeta(model="gpt-4o-mini")

    def test_model_required(self):
        with pytest.raises(ValidationError):
            GenerationMeta(prompt_template="plumbing_repair")


# ============================================================================
# Task #4: Judge output models
# ============================================================================

FAILURE_MODES = [
    "incomplete_answer",
    "safety_violations",
    "unrealistic_tools",
    "overcomplicated_solution",
    "missing_context",
    "poor_quality_tips",
]

QUALITY_DIMENSIONS = [
    "answer_coherence",
    "step_actionability",
    "tool_realism",
    "safety_specificity",
    "tip_usefulness",
    "problem_answer_alignment",
    "appropriate_scope",
    "category_accuracy",
]


def build_valid_failure_result(**overrides) -> dict:
    """Build valid FailureModeResult payload."""
    data = {mode: 0 for mode in FAILURE_MODES}
    data.update(overrides)
    return data


def build_valid_quality_result(**overrides) -> dict:
    """Build valid QualityDimensionResult payload."""
    data = {dim: 1 for dim in QUALITY_DIMENSIONS}
    data.update(overrides)
    return data


class TestFailureModeResult:
    """Verify FailureModeResult enforces 6 binary failure modes (L315-328)."""

    def test_all_pass(self):
        result = FailureModeResult(**build_valid_failure_result())
        assert result.overall_failure is False

    def test_any_fail_triggers_overall(self):
        result = FailureModeResult(**build_valid_failure_result(safety_violations=1))
        assert result.overall_failure is True

    @pytest.mark.parametrize("mode", FAILURE_MODES)
    def test_rejects_value_outside_0_1(self, mode):
        data = build_valid_failure_result(**{mode: 2})
        with pytest.raises(ValidationError):
            FailureModeResult(**data)

    @pytest.mark.parametrize("mode", FAILURE_MODES)
    def test_each_mode_individually_triggers_overall(self, mode):
        data = build_valid_failure_result(**{mode: 1})
        result = FailureModeResult(**data)
        assert result.overall_failure is True


class TestQualityDimensionResult:
    """Verify QualityDimensionResult enforces 8 binary quality scores (L161-174)."""

    def test_all_pass(self):
        result = QualityDimensionResult(**build_valid_quality_result())
        assert result.quality_pass is True

    def test_any_fail_triggers_quality_fail(self):
        result = QualityDimensionResult(**build_valid_quality_result(safety_specificity=0))
        assert result.quality_pass is False

    @pytest.mark.parametrize("dim", QUALITY_DIMENSIONS)
    def test_each_dimension_individually_triggers_fail(self, dim):
        data = build_valid_quality_result(**{dim: 0})
        result = QualityDimensionResult(**data)
        assert result.quality_pass is False

    @pytest.mark.parametrize("dim", QUALITY_DIMENSIONS)
    def test_rejects_value_outside_0_1(self, dim):
        data = build_valid_quality_result(**{dim: 2})
        with pytest.raises(ValidationError):
            QualityDimensionResult(**data)


class TestJudgeResult:
    """Verify JudgeResult combines failure modes + quality dims (L330-358)."""

    def test_valid_judge_result(self):
        result = JudgeResult(
            trace_id="qa_003",
            failure_modes=FailureModeResult(**build_valid_failure_result()),
            quality_scores=QualityDimensionResult(**build_valid_quality_result()),
        )
        assert result.trace_id == "qa_003"
        assert result.failure_modes.overall_failure is False
        assert result.quality_scores.quality_pass is True

    def test_trace_id_required(self):
        with pytest.raises(ValidationError):
            JudgeResult(
                failure_modes=FailureModeResult(**build_valid_failure_result()),
                quality_scores=QualityDimensionResult(**build_valid_quality_result()),
            )

    def test_mixed_pass_fail(self):
        """Item can pass all failure modes but fail quality dimensions."""
        result = JudgeResult(
            trace_id="qa_007",
            failure_modes=FailureModeResult(**build_valid_failure_result()),
            quality_scores=QualityDimensionResult(
                **build_valid_quality_result(tip_usefulness=0)
            ),
        )
        assert result.failure_modes.overall_failure is False
        assert result.quality_scores.quality_pass is False
