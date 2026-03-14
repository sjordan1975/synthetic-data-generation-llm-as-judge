"""Phase 4: Pattern Analysis and Visualization

This module analyses the judge results from Phase 3 to identify failure
patterns and quality gaps, then produces the visualisations required by
_instructions.md L362-428:

1. **Failure Mode Heatmap** — per-item binary failure flags (L372)
2. **Failure Co-occurrence Matrix** — which modes appear together (L368-370)
3. **Failure Rates by Repair Category** — which template is weakest (L422)
4. **Quality Dimension Pass Rates** — bar chart across 8 dims (L428)
5. **Quality Dimension Heatmap** — per-item quality scores (L402)
6. **Most Problematic Items** — items with 3+ failure flags (L426)
7. **Analysis Report** — JSON with all metrics + recommendations (L394-404)

Inputs:
  - ``data/judge_results_baseline.jsonl`` (JudgeResult per item)
  - ``data/validated_baseline.jsonl`` (RepairQA per item, for category info)

Outputs saved to ``outputs/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from models import JudgeResult, RepairQA
from labeler import FAILURE_MODE_FIELDS, QUALITY_DIM_FIELDS


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_judge_results(
    input_file: str = "data/judge_results_baseline.jsonl",
) -> list[JudgeResult]:
    """
    Load judge results from JSONL.

    Args:
        input_file: Path to judge results JSONL.

    Returns:
        List of JudgeResult objects.

    Raises:
        FileNotFoundError: If input file doesn't exist.
    """
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Judge results not found: {input_file}\n"
            f"Please run labeler.py first."
        )

    results: list[JudgeResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(JudgeResult(**json.loads(line)))

    print(f"\u2713 Loaded {len(results)} judge results from {input_file}")
    return results


def load_categories(
    input_file: str = "data/validated_baseline.jsonl",
) -> list[str]:
    """
    Load category strings from validated JSONL (one per item, same order).

    Args:
        input_file: Path to validated data JSONL.

    Returns:
        List of category value strings (e.g. 'plumbing_repair').
    """
    path = Path(input_file)
    if not path.exists():
        return []  # graceful fallback — categories will be "unknown"

    categories: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            categories.append(data.get("category", "unknown"))
    return categories


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

def build_analysis_dataframe(
    results: list[JudgeResult],
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a flat DataFrame from judge results + categories.

    Columns: trace_id, category, 6 FM columns, overall_failure,
    8 QD columns, quality_pass.

    Args:
        results: List of JudgeResult objects.
        categories: Optional list of category strings (same length as results).

    Returns:
        DataFrame ready for analysis.
    """
    if categories is None or len(categories) != len(results):
        categories = ["unknown"] * len(results)

    rows: list[dict] = []
    for r, cat in zip(results, categories):
        row: dict = {"trace_id": r.trace_id, "category": cat}
        # Failure modes
        for m in FAILURE_MODE_FIELDS:
            row[m] = getattr(r.failure_modes, m)
        row["overall_failure"] = int(r.failure_modes.overall_failure)
        # Quality dimensions
        for d in QUALITY_DIM_FIELDS:
            row[d] = getattr(r.quality_scores, d)
        row["quality_pass"] = int(r.quality_scores.quality_pass)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_failure_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute failure mode co-occurrence matrix (_instructions.md L368-370).

    Cell (i, j) = number of items where failure mode i AND j both triggered.
    Diagonal = total count of that mode.

    Args:
        df: Analysis DataFrame with FM columns.

    Returns:
        6×6 DataFrame of co-occurrence counts.
    """
    fm = df[FAILURE_MODE_FIELDS].values  # (n_items, 6) numpy array
    cooc = fm.T @ fm  # dot product gives co-occurrence counts
    return pd.DataFrame(cooc, index=FAILURE_MODE_FIELDS, columns=FAILURE_MODE_FIELDS)


def compute_category_failure_rates(df: pd.DataFrame) -> dict:
    """
    Compute per-category failure rates (_instructions.md L422).

    Args:
        df: Analysis DataFrame with 'category' and FM columns.

    Returns:
        Dict keyed by category with overall_failure_rate and per-mode rates.
    """
    rates: dict = {}
    for cat, group in df.groupby("category"):
        n = len(group)
        cat_data: dict = {"count": n}
        # Per-mode rates
        for m in FAILURE_MODE_FIELDS:
            cat_data[m] = round(group[m].sum() / n * 100, 1)
        # Overall
        cat_data["overall_failure_rate"] = round(
            group["overall_failure"].sum() / n * 100, 1
        )
        rates[str(cat)] = cat_data
    return rates


def compute_quality_summary(df: pd.DataFrame) -> dict:
    """
    Compute quality dimension pass rates across all items.

    Args:
        df: Analysis DataFrame with QD columns.

    Returns:
        Dict with per-dimension pass rates and overall quality pass rate.
    """
    n = len(df)
    summary: dict = {}
    for d in QUALITY_DIM_FIELDS:
        pass_count = int(df[d].sum())
        summary[d] = {
            "pass_count": pass_count,
            "pass_rate": round(pass_count / n * 100, 1),
        }
    quality_pass_count = int(df["quality_pass"].sum())
    summary["overall_quality_pass_rate"] = round(quality_pass_count / n * 100, 1)
    return summary


def find_most_problematic(
    df: pd.DataFrame, min_failures: int = 3
) -> list[dict]:
    """
    Identify items with >= min_failures failure flags (_instructions.md L426).

    Args:
        df: Analysis DataFrame.
        min_failures: Minimum failure count threshold.

    Returns:
        List of dicts with trace_id, failure_count, and failed modes.
    """
    df = df.copy()
    df["failure_count"] = df[FAILURE_MODE_FIELDS].sum(axis=1)
    bad = df[df["failure_count"] >= min_failures].sort_values(
        "failure_count", ascending=False
    )
    items: list[dict] = []
    for _, row in bad.iterrows():
        failed_modes = [m for m in FAILURE_MODE_FIELDS if row[m] == 1]
        items.append({
            "trace_id": row["trace_id"],
            "failure_count": int(row["failure_count"]),
            "failed_modes": failed_modes,
        })
    return items


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def create_failure_heatmap(
    df: pd.DataFrame, output_file: str = "outputs/failure_heatmap.png"
) -> None:
    """
    Create a per-item failure mode heatmap (_instructions.md L372).

    Rows = items (trace_id), Columns = 6 failure modes.
    Red = fail (1), Green = pass (0).
    """
    heatmap_data = df.set_index("trace_id")[FAILURE_MODE_FIELDS]

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))
    sns.heatmap(
        heatmap_data, cmap="RdYlGn_r", annot=True, fmt="d",
        cbar_kws={"label": "Failure (1) / Pass (0)"},
        linewidths=0.5, linecolor="gray", vmin=0, vmax=1, ax=ax,
    )
    ax.set_title("Failure Mode Heatmap", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Failure Modes")
    ax.set_ylabel("trace_id")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    _save_fig(fig, output_file)


def create_cooccurrence_heatmap(
    cooc: pd.DataFrame, output_file: str = "outputs/failure_cooccurrence.png"
) -> None:
    """
    Create failure co-occurrence heatmap (_instructions.md L368).

    Symmetric 6×6 matrix showing how often failure modes appear together.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cooc, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.5, linecolor="gray", ax=ax,
    )
    ax.set_title("Failure Mode Co-occurrence", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    _save_fig(fig, output_file)


def create_category_failure_chart(
    cat_rates: dict, output_file: str = "outputs/category_failure_rates.png"
) -> None:
    """
    Bar chart of overall failure rate per repair category (_instructions.md L422).
    """
    cats = list(cat_rates.keys())
    rates = [cat_rates[c]["overall_failure_rate"] for c in cats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(cats, rates, color="steelblue")
    ax.set_xlabel("Overall Failure Rate (%)")
    ax.set_title("Failure Rate by Repair Category", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    for i, v in enumerate(rates):
        ax.text(v + 1, i, f"{v:.1f}%", va="center")
    plt.tight_layout()

    _save_fig(fig, output_file)


def create_quality_bar_chart(
    qd_summary: dict, output_file: str = "outputs/quality_dimension_scores.png"
) -> None:
    """
    Bar chart of quality dimension pass rates (_instructions.md L428).
    """
    dims = [d for d in QUALITY_DIM_FIELDS]
    rates = [qd_summary[d]["pass_rate"] for d in dims]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(dims, rates, color="mediumseagreen")
    ax.set_xlabel("Pass Rate (%)")
    ax.set_title("Quality Dimension Pass Rates", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    for i, v in enumerate(rates):
        ax.text(v + 1, i, f"{v:.1f}%", va="center")
    plt.tight_layout()

    _save_fig(fig, output_file)


def create_quality_heatmap(
    df: pd.DataFrame, output_file: str = "outputs/quality_dimension_heatmap.png"
) -> None:
    """
    Per-item quality dimension heatmap (_instructions.md L402).

    Rows = items, Columns = 8 quality dimensions.
    Green = pass (1), Red = fail (0).
    """
    heatmap_data = df.set_index("trace_id")[QUALITY_DIM_FIELDS]

    fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.35)))
    sns.heatmap(
        heatmap_data, cmap="RdYlGn", annot=True, fmt="d",
        cbar_kws={"label": "Pass (1) / Fail (0)"},
        linewidths=0.5, linecolor="gray", vmin=0, vmax=1, ax=ax,
    )
    ax.set_title("Quality Dimension Heatmap", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Quality Dimensions")
    ax.set_ylabel("trace_id")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    _save_fig(fig, output_file)


def _save_fig(fig, output_file: str) -> None:
    """Save a matplotlib figure and close it."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\u2713 Saved {output_file}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def save_analysis_report(
    report: dict, output_file: str = "outputs/analysis_report.json"
) -> None:
    """
    Save analysis report as JSON.

    Args:
        report: Report dictionary.
        output_file: Output path.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\u2713 Analysis report saved to {output_file}")


def print_analysis_summary(
    cat_rates: dict,
    qd_summary: dict,
    problematic: list[dict],
    total: int,
    failure_rate: float,
) -> None:
    """
    Print analysis summary to console.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nTotal Samples: {total}")
    print(f"Overall Failure Rate: {failure_rate}%")
    print(f"Overall Quality Pass Rate: {qd_summary['overall_quality_pass_rate']}%")

    print("\nFailure Rate by Category:")
    print("-" * 70)
    for cat, data in cat_rates.items():
        print(f"  {cat:25s}: {data['overall_failure_rate']:5.1f}% ({data['count']} items)")

    print("\nQuality Dimension Pass Rates:")
    print("-" * 70)
    for d in QUALITY_DIM_FIELDS:
        info = qd_summary[d]
        print(f"  {d:30s}: {info['pass_rate']:5.1f}%")

    if problematic:
        print(f"\nMost Problematic Items (\u22653 failures):")
        print("-" * 70)
        for item in problematic:
            modes = ", ".join(item["failed_modes"])
            print(f"  {item['trace_id']}: {item['failure_count']} failures [{modes}]")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Main execution function for Phase 4: Pattern Analysis & Visualisation.

    Steps:
    1. Load judge results (JSONL) + categories
    2. Build analysis DataFrame
    3. Compute metrics (co-occurrence, category rates, quality summary)
    4. Generate all visualisations (6 charts)
    5. Save analysis report (JSON)
    6. Print summary
    """
    print("=" * 70)
    print("PHASE 4: PATTERN ANALYSIS & VISUALIZATION")
    print("=" * 70)
    print()

    try:
        # 1. Load data
        results = load_judge_results("data/judge_results_baseline.jsonl")
        categories = load_categories("data/validated_baseline.jsonl")

        # 2. Build DataFrame
        df = build_analysis_dataframe(results, categories)

        # 3. Compute metrics
        cooc = compute_failure_cooccurrence(df)
        cat_rates = compute_category_failure_rates(df)
        qd_summary = compute_quality_summary(df)
        problematic = find_most_problematic(df, min_failures=3)
        total = len(df)
        failure_rate = round(df["overall_failure"].sum() / total * 100, 1)

        # 4. Visualisations
        print("\nGenerating visualisations...")
        create_failure_heatmap(df)
        create_cooccurrence_heatmap(cooc)
        create_category_failure_chart(cat_rates)
        create_quality_bar_chart(qd_summary)
        create_quality_heatmap(df)

        # 5. Report
        report = {
            "phase": "Phase 4 - Analysis",
            "total_samples": total,
            "overall_failure_rate": failure_rate,
            "category_failure_rates": cat_rates,
            "quality_summary": qd_summary,
            "cooccurrence_matrix": cooc.to_dict(),
            "most_problematic_items": problematic,
        }
        save_analysis_report(report)

        # 6. Console summary
        print_analysis_summary(cat_rates, qd_summary, problematic, total, failure_rate)

        print("\n" + "=" * 70)
        print("PHASE 4 COMPLETE")
        print("=" * 70)
        print("\nOutputs:")
        print("  outputs/failure_heatmap.png")
        print("  outputs/failure_cooccurrence.png")
        print("  outputs/category_failure_rates.png")
        print("  outputs/quality_dimension_scores.png")
        print("  outputs/quality_dimension_heatmap.png")
        print("  outputs/analysis_report.json")
        print("\nNext: Phase 5 — Prompt Refinement (python refiner.py)")

    except FileNotFoundError as e:
        print(f"\n\u2717 Error: {e}")
        print("\nPlease run Phase 3 first: python labeler.py")

    except Exception as e:
        print(f"\n\u2717 Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
