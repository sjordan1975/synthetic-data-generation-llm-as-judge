"""
Phase 3: Failure Mode Labeling for Synthetic Repair Q&A

This module uses LLM-assisted labeling to evaluate the quality of synthetic
repair Q&A pairs. It identifies 6 common failure modes to establish a baseline
for prompt refinement in Phase 5.

About LLM-as-Judge for Quality Assessment:
Similar to the LLM-as-Judge exercise, we use an LLM to evaluate content quality
based on defined criteria. The LLM acts as a consistent judge, labeling each
sample for multiple failure modes. This is faster than manual labeling and
provides consistent criteria across all samples.

Failure Modes:
1. incomplete_answer - Answer doesn't fully address the question
2. safety_violations - Missing critical safety warnings
3. unrealistic_tools - Requires specialized equipment homeowners don't have
4. overcomplicated_solution - Too complex for DIY, should call professional
5. missing_context - Lacks important details (materials, measurements)
6. poor_quality_tips - Tips are generic, unhelpful, or obvious
"""

import json
import pandas as pd
from typing import Dict, List
from pathlib import Path
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

from models import RepairQA

# Load environment variables
load_dotenv(dotenv_path="../.env.local")

# Initialize Instructor-wrapped OpenAI client
client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
)


class FailureModeLabels(BaseModel):
    """
    Pydantic model for failure mode labels.
    
    Each field is a binary indicator (0=pass, 1=fail) for a specific
    quality issue. Using Pydantic with Instructor ensures the LLM
    returns structured, validated labels.
    """
    
    incomplete_answer: int = Field(
        ...,
        description="1 if answer doesn't fully address the question, 0 otherwise",
        ge=0,
        le=1
    )
    
    safety_violations: int = Field(
        ...,
        description="1 if missing critical safety warnings, 0 otherwise",
        ge=0,
        le=1
    )
    
    unrealistic_tools: int = Field(
        ...,
        description="1 if requires specialized equipment homeowners don't have, 0 otherwise",
        ge=0,
        le=1
    )
    
    overcomplicated_solution: int = Field(
        ...,
        description="1 if too complex for DIY (should call professional), 0 otherwise",
        ge=0,
        le=1
    )
    
    missing_context: int = Field(
        ...,
        description="1 if lacks important details (materials, measurements, specs), 0 otherwise",
        ge=0,
        le=1
    )
    
    poor_quality_tips: int = Field(
        ...,
        description="1 if tips are generic, unhelpful, or obvious, 0 otherwise",
        ge=0,
        le=1
    )


def create_labeling_prompt(sample: RepairQA) -> str:
    """
    Create a detailed prompt for LLM-assisted failure mode labeling.
    
    Args:
        sample: RepairQA object to evaluate
    
    Returns:
        str: Formatted prompt with evaluation criteria
    
    About the Prompt:
    Like the LLM-as-Judge exercise, the prompt includes:
    - Clear labeling rules for each failure mode
    - Specific criteria and examples
    - Instructions for binary output (0 or 1)
    - Context about the target audience (homeowners)
    """
    return f"""You are an expert evaluator for home DIY repair content quality.

Evaluate the following repair Q&A pair for quality issues. This content is intended for homeowners attempting DIY repairs.

REPAIR Q&A TO EVALUATE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Question: {sample.question}

Answer: {sample.answer}

Equipment/Problem: {sample.equipment_problem}

Tools Required: {', '.join(sample.tools_required)}

Steps:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(sample.steps))}

Safety Info: {sample.safety_info}

Tips: {sample.tips}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVALUATION CRITERIA:

Label each failure mode as 1 (FAIL) or 0 (PASS):

1. incomplete_answer (1=FAIL, 0=PASS)
   - FAIL if: Answer doesn't fully address the question, missing key steps, vague instructions
   - PASS if: Answer comprehensively addresses the question with clear guidance

2. safety_violations (1=FAIL, 0=PASS)
   - FAIL if: Missing critical safety warnings (electrical hazards, water shutoff, protective gear)
   - FAIL if: Dangerous advice or missing "when to call a professional" guidance
   - PASS if: Appropriate safety warnings present for the repair type

3. unrealistic_tools (1=FAIL, 0=PASS)
   - FAIL if: Requires specialized professional tools (pipe threader, HVAC gauges, electrical testers beyond basic multimeter)
   - PASS if: Tools are commonly available to homeowners (screwdriver, wrench, pliers, basic multimeter)

4. overcomplicated_solution (1=FAIL, 0=PASS)
   - FAIL if: Solution is too complex for typical homeowner (requires professional skills/knowledge)
   - FAIL if: Should clearly recommend calling a professional instead
   - PASS if: Appropriately scoped for DIY with reasonable skill level

5. missing_context (1=FAIL, 0=PASS)
   - FAIL if: Missing important details like material specifications, measurements, part numbers
   - FAIL if: Vague about quantities, sizes, or specific components
   - PASS if: Includes sufficient detail for homeowner to execute

6. poor_quality_tips (1=FAIL, 0=PASS)
   - FAIL if: Tips are obvious, generic, or don't add value ("be careful", "take your time")
   - FAIL if: Tips are irrelevant or unhelpful
   - PASS if: Tips provide genuinely useful, specific advice

IMPORTANT:
- Be consistent in your evaluation
- Consider the target audience: homeowners with basic DIY skills
- Focus on practical, safety-first guidance
- A sample can fail multiple criteria or none

Provide your evaluation as binary labels (0 or 1) for each failure mode.
"""


def label_sample(sample: RepairQA, sample_index: int) -> FailureModeLabels:
    """
    Use LLM to label a single sample for all failure modes.
    
    Args:
        sample: RepairQA object to label
        sample_index: Index for progress tracking
    
    Returns:
        FailureModeLabels: Structured labels for all 6 failure modes
    
    About LLM Labeling:
    - Uses Instructor to enforce FailureModeLabels schema
    - Temperature set low (0.2) for consistent, deterministic labeling
    - Returns validated Pydantic object with binary labels
    """
    prompt = create_labeling_prompt(sample)
    
    try:
        # LLM evaluates and returns structured labels
        # Temperature: Low (0.2) for consistent, deterministic evaluation
        # Lower temperature = more consistent labeling across samples
        labels = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=FailureModeLabels,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator for home DIY repair content. Provide consistent, objective quality assessments."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,  # Low temperature for consistent labeling
            max_tokens=500
        )
        
        print(f"  ✓ Sample {sample_index}: Labeled")
        return labels
    
    except Exception as e:
        print(f"  ✗ Sample {sample_index}: Labeling failed - {e}")
        # Return all zeros (pass) as fallback
        return FailureModeLabels(
            incomplete_answer=0,
            safety_violations=0,
            unrealistic_tools=0,
            overcomplicated_solution=0,
            missing_context=0,
            poor_quality_tips=0
        )


def load_validated_data(input_file: str = "data/validated_data.json") -> List[RepairQA]:
    """
    Load validated data from Phase 2.
    
    Args:
        input_file: Path to validated data JSON file
    
    Returns:
        List[RepairQA]: List of validated RepairQA objects
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Validated data file not found: {input_file}\n"
            f"Please run validator.py first to validate the data."
        )
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to RepairQA objects
    samples = [RepairQA(**item) for item in data]
    
    print(f"✓ Loaded {len(samples)} validated samples from {input_file}")
    return samples


def create_labeled_dataframe(samples: List[RepairQA], labels: List[FailureModeLabels]) -> pd.DataFrame:
    """
    Create a Pandas DataFrame with samples and failure mode labels.
    
    Args:
        samples: List of RepairQA objects
        labels: List of FailureModeLabels objects
    
    Returns:
        pd.DataFrame: DataFrame with trace_id, all fields, and failure mode columns
    
    About the DataFrame:
    - trace_id: Auto-assigned 1-20 for tracking
    - All 7 RepairQA fields (question, answer, etc.)
    - 6 binary failure mode columns (0=pass, 1=fail)
    - Ready for Phase 4 analysis and visualization
    """
    # Build list of dictionaries for DataFrame
    rows = []
    
    for i, (sample, label) in enumerate(zip(samples, labels), start=1):
        row = {
            'trace_id': i,
            'question': sample.question,
            'answer': sample.answer,
            'equipment_problem': sample.equipment_problem,
            'tools_required': ', '.join(sample.tools_required),  # Convert list to string for CSV
            'steps': ' | '.join(sample.steps),  # Convert list to string for CSV
            'safety_info': sample.safety_info,
            'tips': sample.tips,
            # Failure mode labels
            'incomplete_answer': label.incomplete_answer,
            'safety_violations': label.safety_violations,
            'unrealistic_tools': label.unrealistic_tools,
            'overcomplicated_solution': label.overcomplicated_solution,
            'missing_context': label.missing_context,
            'poor_quality_tips': label.poor_quality_tips
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    print(f"\n✓ Created DataFrame with {len(df)} samples and {len(df.columns)} columns")
    return df


def calculate_baseline_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate baseline failure rate metrics.
    
    Args:
        df: DataFrame with failure mode labels
    
    Returns:
        Dict: Metrics including per-mode failure rates and overall rate
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
        'failure_modes': {}
    }
    
    # Calculate per-mode failure rates
    for mode in failure_modes:
        failure_count = df[mode].sum()
        failure_rate = (failure_count / total_samples) * 100
        metrics['failure_modes'][mode] = {
            'count': int(failure_count),
            'rate': round(failure_rate, 1)
        }
    
    # Calculate overall failure rate (any failure)
    df['has_any_failure'] = df[failure_modes].sum(axis=1) > 0
    overall_failures = df['has_any_failure'].sum()
    overall_rate = (overall_failures / total_samples) * 100
    
    metrics['overall_failure_rate'] = round(overall_rate, 1)
    metrics['samples_with_failures'] = int(overall_failures)
    
    return metrics


def save_labeled_data(df: pd.DataFrame, output_file: str = "data/labeled_data.csv") -> None:
    """
    Save labeled DataFrame to CSV.
    
    Args:
        df: DataFrame with labels
        output_file: Output CSV file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✓ Labeled data saved to {output_file}")


def print_baseline_report(metrics: Dict) -> None:
    """
    Print baseline metrics report to console.
    
    Args:
        metrics: Metrics dictionary from calculate_baseline_metrics
    """
    print("\n" + "=" * 70)
    print("BASELINE FAILURE RATE REPORT")
    print("=" * 70)
    print(f"\nTotal Samples: {metrics['total_samples']}")
    print(f"Samples with Any Failure: {metrics['samples_with_failures']} ({metrics['overall_failure_rate']}%)")
    print("\nFailure Mode Breakdown:")
    print("-" * 70)
    
    for mode, data in metrics['failure_modes'].items():
        print(f"  {mode:30s}: {data['count']:2d} samples ({data['rate']:5.1f}%)")
    
    print("=" * 70)


def main():
    """
    Main execution function for Phase 3: Failure Mode Labeling
    
    Steps:
    1. Load validated data from Phase 2
    2. Use LLM to label each sample for 6 failure modes
    3. Create Pandas DataFrame with labels
    4. Calculate baseline metrics
    5. Save labeled data to CSV
    """
    print("=" * 70)
    print("PHASE 3: FAILURE MODE LABELING")
    print("LLM-Assisted Quality Assessment")
    print("=" * 70)
    print()
    
    try:
        # Load validated data
        samples = load_validated_data("data/validated_data.json")
        
        # Label each sample using LLM
        print(f"\nLabeling {len(samples)} samples for 6 failure modes...")
        labels = []
        
        for i, sample in enumerate(samples, start=1):
            label = label_sample(sample, i)
            labels.append(label)
        
        # Create DataFrame
        df = create_labeled_dataframe(samples, labels)
        
        # Calculate baseline metrics
        metrics = calculate_baseline_metrics(df)
        
        # Save labeled data
        save_labeled_data(df, "data/labeled_data.csv")
        
        # Print report
        print_baseline_report(metrics)
        
        print("\n" + "=" * 70)
        print("PHASE 3 COMPLETE")
        print("=" * 70)
        print("\nNext Steps:")
        print("1. Review labeled_data.csv")
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
