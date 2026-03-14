"""Phase 7 (early): Judge Calibration Against Benchmark Dataset

This module validates that the LLM-as-Judge works correctly by running it
against known-good data from the HuggingFace benchmark dataset
(dipenbhuva/home-diy-repair-qa).

About Judge Calibration (_instructions.md L80-113):
Before trusting the judge to evaluate generated data, we must verify it
produces accurate scores on data we *know* is high quality. If the judge
fails benchmark items, the judge prompt needs fixing — not the data.

Procedure:
1. Load 50 random items from the benchmark dataset
2. Convert each to a RepairQA object (field mapping + category enum coercion)
3. Run the judge on each item (reuses labeler.build_judge_prompt + judge_sample)
4. Calculate calibration metrics — require ≥95% overall quality pass rate
5. Save results as JSONL + JSON summary to outputs/

Success criterion: ≥95% of benchmark items must pass all 8 quality dimensions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from models import JudgeResult, RepairCategory, RepairQA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_DATASET_ID: str = "dipenbhuva/home-diy-repair-qa"
DEFAULT_SAMPLE_SIZE: int = 50
CALIBRATION_THRESHOLD: float = 95.0  # ≥95% overall quality pass rate


# ---------------------------------------------------------------------------
# Benchmark loading and conversion
# ---------------------------------------------------------------------------

def load_benchmark_sample(
    dataset_id: str = BENCHMARK_DATASET_ID,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42,
) -> list[dict]:
    """
    Load a random sample from the HuggingFace benchmark dataset.

    About HuggingFace Datasets:
    The ``datasets`` library by HuggingFace provides a simple API to download
    and access curated datasets. We load the ``train`` split and randomly
    sample ``sample_size`` items for judge calibration.

    Args:
        dataset_id: HuggingFace dataset identifier.
        sample_size: Number of items to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        List of dictionaries (one per benchmark item).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(sample_size, len(ds)))
    return [ds[i] for i in indices]


def convert_benchmark_item(row: dict) -> RepairQA:
    """
    Convert a single HuggingFace benchmark row to a RepairQA object.

    Handles minor type mismatches (e.g. tips or steps arriving as a single
    string instead of a list) so that benchmark items pass Pydantic validation.

    Args:
        row: Dictionary from the HuggingFace dataset.

    Returns:
        A validated RepairQA instance.
    """
    # Enum values match benchmark dataset category names directly
    category = RepairCategory(row["category"])

    # Coerce tips/steps to lists if they arrive as strings
    tips = row["tips"]
    if isinstance(tips, str):
        tips = [tips]

    steps = row["steps"]
    if isinstance(steps, str):
        steps = [steps]

    return RepairQA(
        question=row["question"],
        answer=row["answer"],
        equipment_problem=row["equipment_problem"],
        tools_required=row["tools_required"],
        steps=steps,
        safety_info=row["safety_info"],
        tips=tips,
        category=category,
    )


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def calculate_calibration_metrics(results: list[JudgeResult]) -> dict:
    """
    Calculate calibration metrics from judge results on benchmark data.

    The key metric is the overall quality pass rate — the percentage of
    benchmark items where all 8 quality dimensions passed. This must be
    ≥95% for the judge to be considered calibrated (_instructions.md L107).

    Args:
        results: List of JudgeResult objects from judging benchmark items.

    Returns:
        Dict with total_samples, per-dimension pass rates,
        overall_quality_pass_rate, and calibration_passed flag.
    """
    total = len(results)
    if total == 0:
        return {"total_samples": 0, "calibration_passed": False}

    from labeler import QUALITY_DIM_FIELDS, FAILURE_MODE_FIELDS

    # Per-dimension pass counts
    qd_counts: dict[str, int] = {d: 0 for d in QUALITY_DIM_FIELDS}
    fm_counts: dict[str, int] = {m: 0 for m in FAILURE_MODE_FIELDS}
    quality_pass_count = 0
    failure_count = 0

    for r in results:
        if r.quality_scores.quality_pass:
            quality_pass_count += 1
        if r.failure_modes.overall_failure:
            failure_count += 1
        for d in QUALITY_DIM_FIELDS:
            qd_counts[d] += getattr(r.quality_scores, d)
        for m in FAILURE_MODE_FIELDS:
            fm_counts[m] += getattr(r.failure_modes, m)

    overall_qp_rate = round(quality_pass_count / total * 100, 1)

    return {
        "total_samples": total,
        "quality_dimensions": {
            d: {"pass_count": c, "pass_rate": round(c / total * 100, 1)}
            for d, c in qd_counts.items()
        },
        "failure_modes": {
            m: {"count": c, "rate": round(c / total * 100, 1)}
            for m, c in fm_counts.items()
        },
        "samples_with_failures": failure_count,
        "overall_failure_rate": round(failure_count / total * 100, 1),
        "quality_pass_count": quality_pass_count,
        "overall_quality_pass_rate": overall_qp_rate,
        "calibration_threshold": CALIBRATION_THRESHOLD,
        "calibration_passed": overall_qp_rate >= CALIBRATION_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_calibration_results(
    results: list[JudgeResult],
    metrics: dict,
    output_dir: str = "outputs",
) -> None:
    """
    Save calibration results: JSONL of per-item judge results + JSON summary.

    Args:
        results: List of JudgeResult objects.
        metrics: Calibration metrics dict.
        output_dir: Directory to write output files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Per-item results as JSONL
    jsonl_path = out / "benchmark_judge_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
    print(f"✓ Benchmark judge results saved to {jsonl_path}")

    # Summary as JSON
    summary_path = out / "benchmark_calibration.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ Calibration summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_calibration_report(metrics: dict) -> None:
    """Print a formatted calibration report to console."""
    print("\n" + "=" * 70)
    print("JUDGE CALIBRATION REPORT (Benchmark)")
    print("=" * 70)

    total = metrics["total_samples"]
    qp = metrics["overall_quality_pass_rate"]
    passed = metrics["calibration_passed"]
    threshold = metrics["calibration_threshold"]

    print(f"\nBenchmark Items Evaluated: {total}")
    print(f"Overall Quality Pass Rate: {qp}%  (threshold: ≥{threshold}%)")
    status = "✓ CALIBRATION PASSED" if passed else "✗ CALIBRATION FAILED"
    print(f"Status: {status}")

    print("\nQuality Dimension Pass Rates:")
    print("-" * 70)
    for dim, data in metrics["quality_dimensions"].items():
        print(f"  {dim:30s}: {data['pass_count']:3d}/{total} ({data['pass_rate']:5.1f}%)")

    if metrics["samples_with_failures"] > 0:
        print(f"\nFailure Modes Detected on Benchmark (unexpected):")
        print("-" * 70)
        for mode, data in metrics["failure_modes"].items():
            if data["count"] > 0:
                print(f"  {mode:30s}: {data['count']:3d} ({data['rate']:5.1f}%)")

    print("=" * 70)

    if not passed:
        print("\n⚠  The judge is failing benchmark items. This means the judge")
        print("   prompt needs adjustment — the benchmark data is the standard.")
        print("   Review the per-dimension rates above to identify which criteria")
        print("   are too strict, then revise build_judge_prompt() in labeler.py.")


# ---------------------------------------------------------------------------
# Quality gap analysis (Task #12 — _instructions.md L412, L430)
# ---------------------------------------------------------------------------

def compute_quality_gap(
    benchmark_results: list[JudgeResult],
    generated_results: list[JudgeResult],
) -> dict:
    """
    Compute per-dimension quality gap between benchmark and generated data.

    A negative gap means the generated data scores lower than the benchmark.

    Args:
        benchmark_results: JudgeResults from judging benchmark items.
        generated_results: JudgeResults from judging generated (corrected) items.

    Returns:
        Dict with per-dimension rates, gaps, and overall quality gap.
    """
    from labeler import QUALITY_DIM_FIELDS

    def _dim_rates(results: list[JudgeResult]) -> dict[str, float]:
        n = len(results)
        return {
            d: round(sum(getattr(r.quality_scores, d) for r in results) / n * 100, 1)
            for d in QUALITY_DIM_FIELDS
        }

    def _quality_pass_rate(results: list[JudgeResult]) -> float:
        n = len(results)
        return round(sum(1 for r in results if r.quality_scores.quality_pass) / n * 100, 1)

    b_rates = _dim_rates(benchmark_results)
    g_rates = _dim_rates(generated_results)
    b_qp = _quality_pass_rate(benchmark_results)
    g_qp = _quality_pass_rate(generated_results)

    per_dim: dict = {}
    for d in QUALITY_DIM_FIELDS:
        per_dim[d] = {
            "benchmark_rate": b_rates[d],
            "generated_rate": g_rates[d],
            "gap_pp": round(g_rates[d] - b_rates[d], 1),
        }

    return {
        "benchmark_samples": len(benchmark_results),
        "generated_samples": len(generated_results),
        "benchmark_quality_pass_rate": b_qp,
        "generated_quality_pass_rate": g_qp,
        "overall_quality_gap_pp": round(g_qp - b_qp, 1),
        "per_dimension": per_dim,
    }


def save_benchmark_comparison(
    report: dict,
    output_file: str = "outputs/benchmark_comparison.json",
) -> None:
    """
    Save benchmark comparison report as JSON.

    Args:
        report: Quality gap dict from compute_quality_gap.
        output_file: Output path.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ Benchmark comparison saved to {output_file}")


def create_benchmark_vs_generated_chart(
    gap: dict,
    output_file: str = "outputs/benchmark_vs_generated.png",
) -> None:
    """
    Side-by-side bar chart of quality dimension pass rates:
    benchmark vs generated (_instructions.md L430).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from labeler import QUALITY_DIM_FIELDS

    dims = list(QUALITY_DIM_FIELDS)
    bench_vals = [gap["per_dimension"][d]["benchmark_rate"] for d in dims]
    gen_vals = [gap["per_dimension"][d]["generated_rate"] for d in dims]

    x = np.arange(len(dims))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, bench_vals, width, label="Benchmark", color="royalblue")
    ax.bar(x + width / 2, gen_vals, width, label="Generated", color="mediumseagreen")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Benchmark vs Generated: Quality Dimension Pass Rates",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Benchmark vs generated chart saved to {output_file}")


def print_quality_gap_report(gap: dict) -> None:
    """Print quality gap report to console."""
    print("\n" + "=" * 70)
    print("QUALITY GAP ANALYSIS: Benchmark vs Generated")
    print("=" * 70)

    print(f"\nBenchmark: {gap['benchmark_samples']} items, "
          f"{gap['benchmark_quality_pass_rate']}% quality pass")
    print(f"Generated: {gap['generated_samples']} items, "
          f"{gap['generated_quality_pass_rate']}% quality pass")
    print(f"Overall Gap: {gap['overall_quality_gap_pp']:+.1f}pp")

    print("\nPer-Dimension Gaps:")
    print("-" * 70)
    for d, info in gap["per_dimension"].items():
        arrow = "↑" if info["gap_pp"] > 0 else ("↓" if info["gap_pp"] < 0 else "→")
        print(f"  {d:30s}: bench {info['benchmark_rate']:5.1f}%  "
              f"gen {info['generated_rate']:5.1f}%  "
              f"({arrow} {abs(info['gap_pp']):.1f}pp)")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Run judge calibration against the benchmark dataset.

    Steps:
    1. Load 50 random benchmark items from HuggingFace
    2. Convert each to RepairQA
    3. Run the LLM judge on each
    4. Calculate calibration metrics
    5. Save results + print report
    """
    print("=" * 70)
    print("JUDGE CALIBRATION: Benchmark Validation")
    print("=" * 70)
    print()

    from labeler import judge_sample

    # 1. Load benchmark sample
    print(f"Loading {DEFAULT_SAMPLE_SIZE} items from {BENCHMARK_DATASET_ID}...")
    raw_items = load_benchmark_sample()
    print(f"✓ Loaded {len(raw_items)} benchmark items\n")

    # 2. Convert to RepairQA
    samples: list[RepairQA] = []
    for row in raw_items:
        try:
            samples.append(convert_benchmark_item(row))
        except Exception as e:
            print(f"  ⚠ Skipping benchmark item {row.get('id', '?')}: {e}")

    print(f"✓ Converted {len(samples)} items to RepairQA\n")

    # 3. Run judge on each
    print(f"Evaluating {len(samples)} benchmark items with judge...")
    results: list[JudgeResult] = []

    for i, sample in enumerate(samples, start=1):
        trace_id = f"bm_{i:03d}"
        print(f"  {trace_id}: Evaluating...")
        result = judge_sample(sample, trace_id)
        results.append(result)
        qd_flag = "✓" if result.quality_scores.quality_pass else "✗"
        print(f"  {trace_id}: quality={qd_flag}")

    # 4. Calculate metrics
    metrics = calculate_calibration_metrics(results)

    # 5. Save + report
    save_calibration_results(results, metrics)
    print_calibration_report(metrics)


if __name__ == "__main__":
    main()
