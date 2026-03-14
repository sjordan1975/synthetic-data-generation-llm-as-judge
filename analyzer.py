"""
Phase 4: Pattern Analysis and Heatmap Visualization

This module analyzes the labeled failure mode data from Phase 3 to identify
patterns and visualize failure distributions. It creates a heatmap showing
which samples failed which quality criteria, enabling data-driven prompt
refinement in Phase 5.

About Analysis:
Pattern analysis helps identify systematic issues in synthetic data generation.
By visualizing failure modes across samples, we can spot:
- Which failure modes are most common
- Whether certain repair categories have more failures
- Correlations between different failure types
- Specific samples that need attention

This analysis guides Phase 5 prompt refinement to reduce failure rates.
"""

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def load_labeled_data(input_file: str = "data/labeled_data.csv") -> pd.DataFrame:
    """
    Load labeled data from Phase 3.
    
    Args:
        input_file: Path to labeled data CSV file
    
    Returns:
        pd.DataFrame: DataFrame with failure mode labels
    
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Labeled data file not found: {input_file}\n"
            f"Please run labeler.py first to label the data."
        )
    
    df = pd.read_csv(input_path)
    
    print(f"✓ Loaded {len(df)} labeled samples from {input_file}")
    return df


def calculate_failure_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive failure mode metrics.
    
    Args:
        df: DataFrame with failure mode labels
    
    Returns:
        Dict: Detailed metrics including rates, counts, and patterns
    
    Metrics Calculated:
    - Per-mode failure rates and counts
    - Overall failure rate (any failure)
    - Sample-level failure counts
    - Most problematic samples
    """
    failure_modes = [
        'incomplete_answer',
        'safety_violations',
        'unrealistic_tools',
        'overcomplicated_solution',
        'missing_context',
        'poor_quality_tips'
    ]
    
    total_samples = len(df)
    
    metrics = {
        'total_samples': total_samples,
        'failure_modes': {},
        'sample_analysis': {}
    }
    
    # Per-mode metrics
    for mode in failure_modes:
        failure_count = df[mode].sum()
        failure_rate = (failure_count / total_samples) * 100
        
        metrics['failure_modes'][mode] = {
            'count': int(failure_count),
            'rate': round(failure_rate, 1),
            'failed_samples': df[df[mode] == 1]['trace_id'].tolist()
        }
    
    # Overall metrics
    df['total_failures'] = df[failure_modes].sum(axis=1)
    df['has_any_failure'] = df['total_failures'] > 0
    
    overall_failures = df['has_any_failure'].sum()
    overall_rate = (overall_failures / total_samples) * 100
    
    metrics['overall'] = {
        'samples_with_failures': int(overall_failures),
        'failure_rate': round(overall_rate, 1),
        'samples_passing_all': int(total_samples - overall_failures)
    }
    
    # Most problematic samples
    problematic = df.nlargest(5, 'total_failures')[['trace_id', 'question', 'total_failures']]
    metrics['most_problematic_samples'] = [
        {
            'trace_id': int(row['trace_id']),
            'question': row['question'][:60] + '...',
            'failure_count': int(row['total_failures'])
        }
        for _, row in problematic.iterrows()
    ]
    
    return metrics


def create_failure_heatmap(df: pd.DataFrame, output_file: str = "outputs/failure_heatmap.png") -> None:
    """
    Create a heatmap visualization of failure modes across samples.
    
    Args:
        df: DataFrame with failure mode labels
        output_file: Output file path for heatmap image
    
    About Heatmaps:
    Heatmaps are ideal for visualizing binary data across multiple dimensions.
    - Rows: Individual samples (1-20)
    - Columns: Failure modes (6 types)
    - Colors: Red (1=failure) / Green (0=pass)
    
    This makes it easy to spot:
    - Samples with multiple failures (red rows)
    - Common failure modes (red columns)
    - Patterns and correlations
    """
    failure_modes = [
        'incomplete_answer',
        'safety_violations',
        'unrealistic_tools',
        'overcomplicated_solution',
        'missing_context',
        'poor_quality_tips'
    ]
    
    # Extract failure mode columns for heatmap
    heatmap_data = df[failure_modes].copy()
    
    # Create figure and axis
    # About figure size: Width accommodates column labels, height scales with samples
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    # cmap: Color map - 'RdYlGn_r' = Red (bad) to Green (good), reversed
    # annot: Show values (0 or 1) in cells
    # fmt: Format as integers
    # cbar_kws: Colorbar customization
    # linewidths: Cell borders for clarity
    sns.heatmap(
        heatmap_data,
        cmap='RdYlGn_r',  # Red for failures, green for passes
        annot=True,
        fmt='d',
        cbar_kws={'label': 'Failure (1) / Pass (0)'},
        linewidths=0.5,
        linecolor='gray',
        vmin=0,
        vmax=1,
        yticklabels=df['trace_id'].tolist()  # Use trace_id as row labels
    )
    
    # Customize plot
    plt.title('Failure Mode Heatmap - Synthetic Repair Q&A Quality Assessment', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Failure Modes', fontsize=12, fontweight='bold')
    plt.ylabel('Sample ID (trace_id)', fontsize=12, fontweight='bold')
    
    # Rotate column labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Heatmap saved to {output_file}")


def generate_analysis_report(metrics: Dict, output_file: str = "outputs/analysis_report.json") -> None:
    """
    Generate a detailed analysis report in JSON format.
    
    Args:
        metrics: Metrics dictionary from calculate_failure_metrics
        output_file: Output file path for JSON report
    
    Report Contents:
    - Overall failure statistics
    - Per-mode breakdown with failed sample IDs
    - Most problematic samples
    - Recommendations for Phase 5 refinement
    """
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add recommendations based on metrics
    recommendations = []
    
    for mode, data in metrics['failure_modes'].items():
        if data['rate'] > 20:  # More than 20% failure rate
            recommendations.append({
                'failure_mode': mode,
                'issue': f"High failure rate: {data['rate']}%",
                'action': f"Review and strengthen {mode.replace('_', ' ')} criteria in prompts"
            })
    
    if not recommendations:
        recommendations.append({
            'status': 'excellent',
            'message': 'All failure modes below 20% threshold',
            'action': 'Prompts are generating high-quality data'
        })
    
    report = {
        'phase': 'Phase 4 - Analysis',
        'metrics': metrics,
        'recommendations': recommendations
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Analysis report saved to {output_file}")


def print_analysis_summary(metrics: Dict) -> None:
    """
    Print analysis summary to console.
    
    Args:
        metrics: Metrics dictionary from calculate_failure_metrics
    """
    print("\n" + "=" * 70)
    print("FAILURE MODE ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal Samples Analyzed: {metrics['total_samples']}")
    print(f"Samples with Any Failure: {metrics['overall']['samples_with_failures']} ({metrics['overall']['failure_rate']}%)")
    print(f"Samples Passing All Checks: {metrics['overall']['samples_passing_all']}")
    
    print("\nFailure Mode Breakdown:")
    print("-" * 70)
    
    for mode, data in metrics['failure_modes'].items():
        mode_display = mode.replace('_', ' ').title()
        print(f"  {mode_display:30s}: {data['count']:2d} failures ({data['rate']:5.1f}%)")
    
    print("\nMost Problematic Samples:")
    print("-" * 70)
    
    if metrics['most_problematic_samples']:
        for sample in metrics['most_problematic_samples']:
            if sample['failure_count'] > 0:
                print(f"  Sample {sample['trace_id']:2d}: {sample['failure_count']} failures - {sample['question']}")
    else:
        print("  No problematic samples identified")
    
    print("=" * 70)


def main():
    """
    Main execution function for Phase 4: Analysis & Heatmap
    
    Steps:
    1. Load labeled data from Phase 3
    2. Calculate comprehensive failure metrics
    3. Create heatmap visualization
    4. Generate detailed analysis report
    5. Print summary to console
    """
    print("=" * 70)
    print("PHASE 4: PATTERN ANALYSIS & VISUALIZATION")
    print("Failure Mode Heatmap and Metrics")
    print("=" * 70)
    print()
    
    try:
        # Load labeled data
        df = load_labeled_data("data/labeled_data.csv")
        
        # Calculate metrics
        print("\nCalculating failure mode metrics...")
        metrics = calculate_failure_metrics(df)
        
        # Create heatmap
        print("Generating failure mode heatmap...")
        create_failure_heatmap(df, "outputs/failure_heatmap.png")
        
        # Generate report
        print("Creating analysis report...")
        generate_analysis_report(metrics, "outputs/analysis_report.json")
        
        # Print summary
        print_analysis_summary(metrics)
        
        print("\n" + "=" * 70)
        print("PHASE 4 COMPLETE")
        print("=" * 70)
        print("\nNext Steps:")
        print("1. Review outputs/failure_heatmap.png")
        print("2. Review outputs/analysis_report.json")
        print("3. Identify patterns for prompt refinement")
        print("4. Proceed to Phase 5: Prompt Refinement")
    
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run Phase 3 first:")
        print("  python labeler.py")
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
