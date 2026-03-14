"""Phase 3: LLM-as-Judge Labeling for Synthetic Repair Q&A

This module uses an LLM-as-Judge to evaluate the quality of synthetic repair
Q&A pairs. It evaluates each item across two independent axes:

1. **6 Failure Modes** (_instructions.md L319-326): Binary flags for common
   defects (incomplete answer, safety violations, unrealistic tools,
   overcomplicated solution, missing context, poor quality tips).

2. **8 Quality Dimensions** (_instructions.md L165-174): Binary pass/fail
   scores for semantic quality (answer coherence, step actionability, tool
   realism, safety specificity, tip usefulness, problem-answer alignment,
   appropriate scope, category accuracy).

About LLM-as-Judge for Quality Assessment:
We use an LLM to evaluate content quality based on explicit, documented
criteria with positive and negative examples. The LLM acts as a consistent
judge, producing structured Pydantic output via Instructor. Temperature is
set low (0.2) for deterministic evaluation — distinct from the higher
temperature (0.8) used during generation.

The judge output is a JudgeResult per item containing both failure modes
and quality scores. An item is "failed overall" if any failure flag is set.
An item achieves "quality pass" only if all 8 dimensions pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from models import (
    FailureModeResult,
    JudgeResult,
    QualityDimensionResult,
    RepairQA,
)
from prompt_loader import load_judge_template

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_MODEL: str = "gpt-4o-mini"
JUDGE_TEMPERATURE: float = 0.2  # Low temperature for deterministic evaluation
DEFAULT_JUDGE_PROMPT_VERSION: str = "v1"


# ---------------------------------------------------------------------------
# Client initialisation (deferred so tests can import without API keys)
# ---------------------------------------------------------------------------

_client = None  # Lazy singleton


def _get_client():
    """Return an Instructor-wrapped OpenAI client, initialising on first call."""
    global _client
    if _client is not None:
        return _client

    import instructor
    from openai import OpenAI

    for candidate in [Path(".env.local"), Path("../.env.local")]:
        if candidate.exists():
            load_dotenv(dotenv_path=str(candidate))
            break

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please create a .env.local file with your API key."
        )

    _client = instructor.from_openai(
        OpenAI(api_key=api_key, base_url=base_url)
    )
    return _client


# ---------------------------------------------------------------------------
# Judge prompt construction
# ---------------------------------------------------------------------------

def build_judge_prompt(sample: RepairQA, trace_id: str) -> str:
    """
    Build the full evaluation prompt for the LLM-as-Judge.

    The prompt includes:
    - The complete RepairQA item to evaluate
    - Explicit criteria with positive/negative examples for all 6 failure
      modes (_instructions.md L319-326) and all 8 quality dimensions
      (_instructions.md L165-174)
    - Instructions for binary scoring (0/1)
    - Context about the target audience (homeowners)

    Args:
        sample: The RepairQA item to evaluate.
        trace_id: Unique identifier for this evaluation.

    Returns:
        The formatted prompt string.
    """
    tips_rendered = chr(10).join(f"  - {tip}" for tip in sample.tips)
    steps_rendered = chr(10).join(
        f"  {i+1}. {step}" for i, step in enumerate(sample.steps)
    )

    template = load_judge_template(version=DEFAULT_JUDGE_PROMPT_VERSION)

    return template.format(
        trace_id=trace_id,
        category=sample.category.value,
        question=sample.question,
        answer=sample.answer,
        equipment_problem=sample.equipment_problem,
        tools_required=", ".join(sample.tools_required),
        steps=steps_rendered,
        safety_info=sample.safety_info,
        tips=tips_rendered,
    )


def judge_sample(
    sample: RepairQA,
    trace_id: str,
    model: str = DEFAULT_JUDGE_MODEL,
) -> JudgeResult:
    """
    Use the LLM-as-Judge to evaluate a single RepairQA item.

    Returns a JudgeResult containing both failure mode flags and quality
    dimension scores. Uses Instructor to enforce Pydantic structure.

    Args:
        sample: The RepairQA item to evaluate.
        trace_id: Unique identifier for this evaluation.
        model: LLM model identifier for the judge.

    Returns:
        JudgeResult with failure_modes and quality_scores.

    About LLM-as-Judge:
    - Uses Instructor to enforce JudgeResult schema
    - Temperature set low (0.2) for consistent, deterministic evaluation
    - Returns validated Pydantic object with binary scores
    """
    prompt = build_judge_prompt(sample, trace_id)
    client = _get_client()

    try:
        result = client.chat.completions.create(
            model=model,
            response_model=JudgeResult,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert evaluator for home DIY repair content. "
                        "Provide consistent, objective quality assessments using "
                        "the exact scoring criteria provided."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=JUDGE_TEMPERATURE,
            max_tokens=500,
        )

        # Ensure the trace_id matches what we assigned
        if result.trace_id != trace_id:
            result = result.model_copy(update={"trace_id": trace_id})

        return result

    except Exception as e:
        print(f"  ✗ {trace_id}: Judge evaluation failed - {e}")
        raise


def load_validated_data(input_file: str = "data/validated_baseline.jsonl") -> list[RepairQA]:
    """
    Load validated data from Phase 2 (JSONL format).

    Args:
        input_file: Path to validated data JSONL file.

    Returns:
        List of validated RepairQA objects.

    Raises:
        FileNotFoundError: If input file doesn't exist.
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Validated data file not found: {input_file}\n"
            f"Please run validator.py first to validate the data."
        )

    samples: list[RepairQA] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(RepairQA(**json.loads(line)))

    print(f"✓ Loaded {len(samples)} validated samples from {input_file}")
    return samples


# ---------------------------------------------------------------------------
# JSONL result saving
# ---------------------------------------------------------------------------

def save_judge_results(
    results: list[JudgeResult],
    output_file: str = "data/judge_results_baseline.jsonl",
) -> None:
    """
    Save judge results to JSONL (_instructions.md L570).

    Each line is one JudgeResult serialised to JSON.

    Args:
        results: List of JudgeResult objects.
        output_file: Output JSONL path.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(f"\n✓ Judge results saved to {output_file} ({len(results)} items)")


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------

FAILURE_MODE_FIELDS: list[str] = [
    "incomplete_answer",
    "safety_violations",
    "unrealistic_tools",
    "overcomplicated_solution",
    "missing_context",
    "poor_quality_tips",
]

QUALITY_DIM_FIELDS: list[str] = [
    "answer_coherence",
    "step_actionability",
    "tool_realism",
    "safety_specificity",
    "tip_usefulness",
    "problem_answer_alignment",
    "appropriate_scope",
    "category_accuracy",
]


def calculate_baseline_metrics(results: list[JudgeResult]) -> dict:
    """
    Aggregate judge results into baseline metrics.

    Args:
        results: List of JudgeResult objects from the judge.

    Returns:
        Dict with total_samples, per-mode failure counts/rates,
        overall_failure_rate, per-dimension pass counts/rates,
        and overall_quality_pass_rate.
    """
    total = len(results)
    if total == 0:
        return {"total_samples": 0}

    metrics: dict = {"total_samples": total}

    # --- Failure modes ---
    fm_counts: dict[str, int] = {m: 0 for m in FAILURE_MODE_FIELDS}
    samples_with_failure = 0

    for r in results:
        if r.failure_modes.overall_failure:
            samples_with_failure += 1
        for m in FAILURE_MODE_FIELDS:
            fm_counts[m] += getattr(r.failure_modes, m)

    metrics["failure_modes"] = {
        m: {"count": c, "rate": round(c / total * 100, 1)}
        for m, c in fm_counts.items()
    }
    metrics["samples_with_failures"] = samples_with_failure
    metrics["overall_failure_rate"] = round(samples_with_failure / total * 100, 1)

    # --- Quality dimensions ---
    qd_pass_counts: dict[str, int] = {d: 0 for d in QUALITY_DIM_FIELDS}
    quality_pass_count = 0

    for r in results:
        if r.quality_scores.quality_pass:
            quality_pass_count += 1
        for d in QUALITY_DIM_FIELDS:
            qd_pass_counts[d] += getattr(r.quality_scores, d)

    metrics["quality_dimensions"] = {
        d: {"pass_count": c, "pass_rate": round(c / total * 100, 1)}
        for d, c in qd_pass_counts.items()
    }
    metrics["overall_quality_pass_rate"] = round(quality_pass_count / total * 100, 1)

    return metrics


def print_baseline_report(metrics: dict) -> None:
    """
    Print baseline metrics report to console.

    Args:
        metrics: Metrics dictionary from calculate_baseline_metrics.
    """
    print("\n" + "=" * 70)
    print("BASELINE JUDGE REPORT")
    print("=" * 70)
    total = metrics["total_samples"]
    print(f"\nTotal Samples Evaluated: {total}")
    print(f"Samples with Any Failure: {metrics['samples_with_failures']} "
          f"({metrics['overall_failure_rate']}%)")
    print(f"Overall Quality Pass Rate: {metrics['overall_quality_pass_rate']}%")

    print("\nFailure Mode Breakdown:")
    print("-" * 70)
    for mode, data in metrics["failure_modes"].items():
        print(f"  {mode:30s}: {data['count']:3d} failures ({data['rate']:5.1f}%)")

    print("\nQuality Dimension Pass Rates:")
    print("-" * 70)
    for dim, data in metrics["quality_dimensions"].items():
        print(f"  {dim:30s}: {data['pass_count']:3d}/{total} ({data['pass_rate']:5.1f}%)")

    print("=" * 70)


def main():
    """
    Main execution function for Phase 3: LLM-as-Judge Labeling.

    Steps:
    1. Load validated data from Phase 2 (JSONL)
    2. Evaluate each sample with the LLM judge (6 failure modes + 8 quality dims)
    3. Calculate baseline metrics
    4. Save judge results as JSONL
    5. Print baseline report
    """
    print("=" * 70)
    print("PHASE 3: LLM-AS-JUDGE LABELING")
    print("6 Failure Modes + 8 Quality Dimensions")
    print("=" * 70)
    print()

    try:
        # Load validated data
        samples = load_validated_data("data/validated_baseline.jsonl")

        # Evaluate each sample
        print(f"\nEvaluating {len(samples)} samples...")
        results: list[JudgeResult] = []

        for i, sample in enumerate(samples, start=1):
            trace_id = f"qa_{i:03d}"
            print(f"  {trace_id}: Evaluating...")
            result = judge_sample(sample, trace_id)
            results.append(result)
            fm_flag = "✗" if result.failure_modes.overall_failure else "✓"
            qd_flag = "✓" if result.quality_scores.quality_pass else "✗"
            print(f"  {trace_id}: failures={fm_flag}  quality={qd_flag}")

        # Calculate metrics
        metrics = calculate_baseline_metrics(results)

        # Save results
        save_judge_results(results, "data/judge_results_baseline.jsonl")

        # Save metrics as JSON summary
        metrics_path = Path("outputs/baseline_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Metrics saved to {metrics_path}")

        # Print report
        print_baseline_report(metrics)

        print("\n" + "=" * 70)
        print("PHASE 3 COMPLETE")
        print("=" * 70)
        print("\nNext Steps:")
        print("1. Review data/judge_results_baseline.jsonl")
        print("2. Proceed to Phase 4: Analysis & Heatmap")
        print("   Run: python analyzer.py")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run Phase 2 first:")
        print("  python validator.py")

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
