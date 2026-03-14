"""Phase 5: Prompt Refinement and Before/After Comparison

This module implements the data-driven prompt correction loop
(_instructions.md L376-380):

1. Load baseline judge results + analysis report
2. Identify the weakest failure modes and quality dimensions
3. Apply targeted prompt corrections (data-driven, not guesses)
4. Re-generate and re-judge with corrected prompts
5. Compute before/after comparison (per-mode + per-dimension deltas)
6. Save comparison report + before/after visualisation

Success criteria (_instructions.md L615-621):
- Post-correction failure rate ≤ 80% of baseline (>80% reduction)
- Post-correction quality pass rate ≥ 80%

Note: The prompt correction strategy will be filled in after the
pipeline produces real failure data. The comparison and reporting
scaffolding is fully deterministic and tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from models import JudgeResult
from labeler import FAILURE_MODE_FIELDS, QUALITY_DIM_FIELDS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_REDUCTION_TARGET: float = 80.0   # >80% reduction in failure rate
QUALITY_PASS_TARGET: float = 80.0        # ≥80% overall quality pass rate


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_analysis_report(
    input_file: str = "outputs/analysis_report.json",
) -> dict:
    """
    Load Phase 4 analysis report.

    Args:
        input_file: Path to analysis report JSON.

    Returns:
        Analysis report dictionary.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Analysis report not found: {input_file}\n"
            f"Please run analyzer.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    print(f"\u2713 Loaded analysis report from {input_file}")
    return report


def load_judge_results(input_file: str) -> list[JudgeResult]:
    """
    Load judge results from JSONL.

    Args:
        input_file: Path to JSONL file.

    Returns:
        List of JudgeResult objects.
    """
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Judge results not found: {input_file}")
    results: list[JudgeResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(JudgeResult(**json.loads(line)))
    return results


# ---------------------------------------------------------------------------
# Before / After comparison
# ---------------------------------------------------------------------------

def _compute_rates(results: list[JudgeResult]) -> dict:
    """Compute failure + quality rates from a list of JudgeResults."""
    n = len(results)
    if n == 0:
        return {}

    fm_rates: dict[str, float] = {}
    for m in FAILURE_MODE_FIELDS:
        count = sum(getattr(r.failure_modes, m) for r in results)
        fm_rates[m] = round(count / n * 100, 1)

    qd_rates: dict[str, float] = {}
    for d in QUALITY_DIM_FIELDS:
        count = sum(getattr(r.quality_scores, d) for r in results)
        qd_rates[d] = round(count / n * 100, 1)

    overall_failure = sum(
        1 for r in results if r.failure_modes.overall_failure
    )
    overall_quality_pass = sum(
        1 for r in results if r.quality_scores.quality_pass
    )

    return {
        "total_samples": n,
        "overall_failure_rate": round(overall_failure / n * 100, 1),
        "overall_quality_pass_rate": round(overall_quality_pass / n * 100, 1),
        "failure_mode_rates": fm_rates,
        "quality_dim_rates": qd_rates,
    }


def compute_before_after(
    baseline: list[JudgeResult],
    corrected: list[JudgeResult],
) -> dict:
    """
    Compute before/after comparison across failure modes + quality dims.

    Args:
        baseline: JudgeResults from baseline run.
        corrected: JudgeResults from corrected run.

    Returns:
        Dict with baseline stats, corrected stats, per-mode/dim deltas,
        and whether reduction/quality targets are met.
    """
    b = _compute_rates(baseline)
    c = _compute_rates(corrected)

    # Per failure-mode deltas
    fm_deltas: dict = {}
    for m in FAILURE_MODE_FIELDS:
        br = b["failure_mode_rates"][m]
        cr = c["failure_mode_rates"][m]
        fm_deltas[m] = {
            "baseline_rate": br,
            "corrected_rate": cr,
            "delta_pp": round(cr - br, 1),
        }

    # Per quality-dimension deltas
    qd_deltas: dict = {}
    for d in QUALITY_DIM_FIELDS:
        br = b["quality_dim_rates"][d]
        cr = c["quality_dim_rates"][d]
        qd_deltas[d] = {
            "baseline_rate": br,
            "corrected_rate": cr,
            "delta_pp": round(cr - br, 1),
        }

    # Overall targets
    b_fail = b["overall_failure_rate"]
    c_fail = c["overall_failure_rate"]
    if b_fail > 0:
        reduction_pct = round((b_fail - c_fail) / b_fail * 100, 1)
    else:
        reduction_pct = 0.0 if c_fail == 0 else -100.0

    return {
        "baseline": b,
        "corrected": c,
        "failure_mode_deltas": fm_deltas,
        "quality_dimension_deltas": qd_deltas,
        "failure_reduction_pct": reduction_pct,
        "reduction_target_met": reduction_pct >= FAILURE_REDUCTION_TARGET,
        "quality_target_met": c["overall_quality_pass_rate"] >= QUALITY_PASS_TARGET,
    }


# ---------------------------------------------------------------------------
# Weakness identification
# ---------------------------------------------------------------------------

def identify_weakest_areas(results: list[JudgeResult]) -> dict:
    """
    Identify failure modes and quality dimensions most in need of correction.

    Returns lists sorted worst-first so the refiner knows where to focus.

    Args:
        results: JudgeResult list (typically baseline).

    Returns:
        Dict with sorted failure_modes and quality_dimensions lists.
    """
    rates = _compute_rates(results)

    fm_list = [
        {"mode": m, "rate": rates["failure_mode_rates"][m]}
        for m in FAILURE_MODE_FIELDS
        if rates["failure_mode_rates"][m] > 0
    ]
    fm_list.sort(key=lambda x: x["rate"], reverse=True)

    qd_list = [
        {"dimension": d, "pass_rate": rates["quality_dim_rates"][d]}
        for d in QUALITY_DIM_FIELDS
    ]
    qd_list.sort(key=lambda x: x["pass_rate"])

    return {
        "failure_modes": fm_list,
        "quality_dimensions": qd_list,
        "overall_failure_rate": rates["overall_failure_rate"],
        "overall_quality_pass_rate": rates["overall_quality_pass_rate"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_comparison_report(
    report: dict,
    output_file: str = "outputs/refinement_comparison.json",
) -> None:
    """
    Save before/after comparison report as JSON.

    Args:
        report: Comparison dict from compute_before_after.
        output_file: Output path.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\u2713 Comparison report saved to {output_file}")


def create_before_after_chart(
    comparison: dict,
    output_file: str = "outputs/before_after_failure_rates.png",
) -> None:
    """
    Grouped bar chart: per-mode failure rate before vs after (_instructions.md L424).

    Args:
        comparison: Dict from compute_before_after.
        output_file: Output path.
    """
    import numpy as np

    modes = list(comparison["failure_mode_deltas"].keys())
    baseline_vals = [comparison["failure_mode_deltas"][m]["baseline_rate"] for m in modes]
    corrected_vals = [comparison["failure_mode_deltas"][m]["corrected_rate"] for m in modes]

    x = np.arange(len(modes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="salmon")
    ax.bar(x + width / 2, corrected_vals, width, label="Corrected", color="mediumseagreen")
    ax.set_ylabel("Failure Rate (%)")
    ax.set_title("Before vs After: Per-Mode Failure Rates", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(baseline_vals), max(corrected_vals), 10) * 1.2)
    plt.tight_layout()

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\u2713 Before/after chart saved to {output_file}")


def create_quality_before_after_chart(
    comparison: dict,
    output_file: str = "outputs/before_after_quality_scores.png",
) -> None:
    """
    Grouped bar chart: per-dimension quality pass rate before vs after (_instructions.md L428).
    """
    import numpy as np

    dims = list(comparison["quality_dimension_deltas"].keys())
    baseline_vals = [comparison["quality_dimension_deltas"][d]["baseline_rate"] for d in dims]
    corrected_vals = [comparison["quality_dimension_deltas"][d]["corrected_rate"] for d in dims]

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, baseline_vals, width, label="Baseline", color="lightskyblue")
    ax.bar(x + width / 2, corrected_vals, width, label="Corrected", color="mediumseagreen")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Before vs After: Quality Dimension Pass Rates", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\u2713 Quality before/after chart saved to {output_file}")


def print_comparison_summary(comparison: dict) -> None:
    """
    Print before/after comparison to console.
    """
    print("\n" + "=" * 70)
    print("REFINEMENT COMPARISON")
    print("=" * 70)

    b = comparison["baseline"]
    c = comparison["corrected"]
    print(f"\nBaseline:  {b['total_samples']} items, "
          f"{b['overall_failure_rate']}% failure, "
          f"{b['overall_quality_pass_rate']}% quality pass")
    print(f"Corrected: {c['total_samples']} items, "
          f"{c['overall_failure_rate']}% failure, "
          f"{c['overall_quality_pass_rate']}% quality pass")

    red = comparison['failure_reduction_pct']
    print(f"\nFailure Reduction: {red}%  (target: \u2265{FAILURE_REDUCTION_TARGET}%)")
    print(f"  {'\u2713 MET' if comparison['reduction_target_met'] else '\u2717 NOT MET'}")
    print(f"Quality Pass Rate: {c['overall_quality_pass_rate']}%  (target: \u2265{QUALITY_PASS_TARGET}%)")
    print(f"  {'\u2713 MET' if comparison['quality_target_met'] else '\u2717 NOT MET'}")

    print("\nPer-Mode Failure Rate Deltas:")
    print("-" * 70)
    for m, d in comparison["failure_mode_deltas"].items():
        arrow = "\u2193" if d["delta_pp"] < 0 else ("\u2191" if d["delta_pp"] > 0 else "\u2192")
        print(f"  {m:30s}: {d['baseline_rate']:5.1f}% \u2192 {d['corrected_rate']:5.1f}%  ({arrow} {abs(d['delta_pp']):.1f}pp)")

    print("\nPer-Dimension Quality Pass Rate Deltas:")
    print("-" * 70)
    for d_name, d in comparison["quality_dimension_deltas"].items():
        arrow = "\u2191" if d["delta_pp"] > 0 else ("\u2193" if d["delta_pp"] < 0 else "\u2192")
        print(f"  {d_name:30s}: {d['baseline_rate']:5.1f}% \u2192 {d['corrected_rate']:5.1f}%  ({arrow} {abs(d['delta_pp']):.1f}pp)")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main (placeholder — will be completed after pipeline run)
# ---------------------------------------------------------------------------

def main():
    """
    Main execution for Phase 5: Prompt Refinement.

    Steps:
    1. Load baseline analysis report + judge results
    2. Identify weakest areas
    3. Apply corrected prompts (TODO: data-driven, after pipeline run)
    4. Re-generate + re-judge with corrected prompts
    5. Compute before/after comparison
    6. Generate charts + report

    NOTE: Steps 3–4 will be implemented after the pipeline produces
    real failure data. The comparison and reporting scaffolding below
    is ready to consume those results.
    """
    print("=" * 70)
    print("PHASE 5: PROMPT REFINEMENT & COMPARISON")
    print("=" * 70)
    print()

    try:
        # 1. Load baseline
        report = load_analysis_report("outputs/analysis_report.json")
        baseline_results = load_judge_results("data/judge_results_baseline.jsonl")

        # 2. Identify weaknesses
        weak = identify_weakest_areas(baseline_results)
        print(f"\nBaseline failure rate: {weak['overall_failure_rate']}%")
        print(f"Baseline quality pass: {weak['overall_quality_pass_rate']}%")
        if weak["failure_modes"]:
            print("\nTop failure modes:")
            for fm in weak["failure_modes"][:3]:
                print(f"  {fm['mode']}: {fm['rate']}%")

        # 3-4. TODO: Apply corrections, re-generate, re-judge
        #      This will produce data/judge_results_corrected.jsonl
        corrected_path = "data/judge_results_corrected.jsonl"
        try:
            corrected_results = load_judge_results(corrected_path)
        except FileNotFoundError:
            print(f"\n\u26a0  Corrected results not found ({corrected_path}).")
            print("   Run the correction pipeline first, then re-run this module.")
            return

        # 5. Compare
        comparison = compute_before_after(baseline_results, corrected_results)

        # 6. Charts + report
        create_before_after_chart(comparison)
        create_quality_before_after_chart(comparison)
        save_comparison_report(comparison)
        print_comparison_summary(comparison)

        print("\n" + "=" * 70)
        print("PHASE 5 COMPLETE")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n\u2717 Error: {e}")
        print("\nPlease run earlier phases first.")

    except Exception as e:
        print(f"\n\u2717 Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
