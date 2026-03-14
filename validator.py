"""
Phase 2: Data Validation for Synthetic Repair Q&A

This module validates the synthetic data generated in Phase 1 against the
RepairQA Pydantic model to ensure structural correctness before proceeding
to failure analysis.

About Validation:
Validation ensures that all generated samples have the correct structure,
data types, and meet minimum quality requirements (field lengths, list sizes).
This catches any issues early before investing time in manual labeling.

Process:
1. Load baseline.jsonl from Phase 1 (JSONL format, one record per line)
2. Validate each sample against RepairQA model
3. Separate valid and invalid samples
4. Generate validation report with statistics
5. Save validated samples as JSONL for Phase 3
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from models import RepairQA


def load_synthetic_data(input_file: str = "data/baseline.jsonl") -> list[dict]:
    """
    Load synthetic data from a JSONL file (one JSON object per line).

    The optional "metadata" key is stripped before returning so that
    downstream validation sees only RepairQA fields.

    Args:
        input_file: Path to synthetic data JSONL file.

    Returns:
        List of dictionaries (RepairQA fields only, metadata removed).

    Raises:
        FileNotFoundError: If input file doesn't exist.
        json.JSONDecodeError: If any line is not valid JSON.
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Synthetic data file not found: {input_file}\n"
            f"Please run data_generator.py first to generate the data."
        )

    data: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Strip generation metadata — not part of the RepairQA schema
            record.pop("metadata", None)
            data.append(record)

    print(f"✓ Loaded {len(data)} samples from {input_file}")
    return data


def validate_samples(raw_data: list[dict]) -> tuple[list[RepairQA], list[dict]]:
    """
    Validate raw data against RepairQA Pydantic model.
    
    Args:
        raw_data: List of dictionaries to validate
    
    Returns:
        Tuple containing:
        - List of valid RepairQA objects
        - List of validation errors (dicts with 'index', 'data', 'error')
    
    About Pydantic Validation:
    Pydantic automatically validates:
    - All required fields are present
    - Field types match (str, List[str], etc.)
    - Minimum length/size constraints are met
    - Custom validators pass (if defined)
    
    If validation fails, Pydantic raises ValidationError with detailed
    information about what went wrong.
    """
    valid_samples = []
    validation_errors = []
    
    print(f"\nValidating {len(raw_data)} samples against RepairQA schema...")
    
    for i, sample_data in enumerate(raw_data):
        try:
            # Pydantic validation happens here
            # If successful, returns a validated RepairQA object
            # If fails, raises ValidationError
            validated_sample = RepairQA(**sample_data)
            valid_samples.append(validated_sample)
            print(f"  ✓ Sample {i+1}: Valid")
        
        except ValidationError as e:
            # Capture validation error details
            error_info = {
                'index': i + 1,
                'question': sample_data.get('question', 'N/A')[:60],
                'error': str(e),
                'error_count': len(e.errors())
            }
            validation_errors.append(error_info)
            print(f"  ✗ Sample {i+1}: Invalid - {len(e.errors())} error(s)")
    
    print(f"\n✓ Validation complete:")
    print(f"  Valid samples: {len(valid_samples)}")
    print(f"  Invalid samples: {len(validation_errors)}")
    
    return valid_samples, validation_errors


def save_validated_data(
    valid_samples: list[RepairQA],
    output_file: str = "data/validated_baseline.jsonl",
) -> None:
    """
    Save validated samples to a JSONL file.

    Args:
        valid_samples: List of validated RepairQA objects.
        output_file: Output file path (JSONL).

    About Saving:
    Only structurally valid samples are saved. This ensures Phase 3
    (failure labeling) works with clean, well-formed data.

    About JSONL:
    JSON Lines format stores one JSON object per line. This is consistent
    with the generation output format and is streaming-friendly.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(f"\n✓ Saved {len(valid_samples)} validated samples to {output_file}")


def generate_validation_report(
    total_samples: int,
    valid_samples: list[RepairQA],
    validation_errors: list[dict],
    output_file: str = "outputs/validation_report.txt",
) -> None:
    """
    Generate a detailed validation report.
    
    Args:
        total_samples: Total number of samples processed
        valid_samples: List of valid samples
        validation_errors: List of validation error details
        output_file: Output file path for report
    
    Report Contents:
    - Summary statistics (valid/invalid counts, success rate)
    - Details of any validation errors
    - Recommendations for next steps
    """
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics
    valid_count = len(valid_samples)
    invalid_count = len(validation_errors)
    success_rate = (valid_count / total_samples * 100) if total_samples > 0 else 0
    
    # Build report
    report_lines = [
        "=" * 70,
        "PHASE 2: VALIDATION REPORT",
        "Synthetic Data Structural Validation",
        "=" * 70,
        "",
        "SUMMARY",
        "-" * 70,
        f"Total Samples Processed: {total_samples}",
        f"Valid Samples: {valid_count}",
        f"Invalid Samples: {invalid_count}",
        f"Success Rate: {success_rate:.1f}%",
        "",
    ]
    
    if validation_errors:
        report_lines.extend([
            "VALIDATION ERRORS",
            "-" * 70,
            ""
        ])
        
        for error in validation_errors:
            report_lines.extend([
                f"Sample {error['index']}:",
                f"  Question: {error['question']}...",
                f"  Error Count: {error['error_count']}",
                f"  Details: {error['error'][:200]}...",
                ""
            ])
    else:
        report_lines.extend([
            "✓ ALL SAMPLES VALID",
            "-" * 70,
            "All samples passed Pydantic validation.",
            "Ready to proceed to Phase 3: Failure Labeling.",
            ""
        ])
    
    report_lines.extend([
        "=" * 70,
        "NEXT STEPS",
        "-" * 70,
    ])
    
    if valid_count > 0:
        report_lines.extend([
            f"✓ {valid_count} validated samples saved to data/validated_data.json",
            "✓ Ready for Phase 3: Failure Mode Labeling",
            "",
            "Run: python labeler.py",
        ])
    else:
        report_lines.extend([
            "✗ No valid samples found!",
            "✗ Review validation errors above",
            "✗ Fix data generation issues in data_generator.py",
            "✗ Re-run: python data_generator.py",
        ])
    
    report_lines.append("=" * 70)
    
    # Write report
    report_text = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n✓ Validation report saved to {output_file}")
    
    # Also print to console
    print("\n" + report_text)


def main():
    """
    Main execution function for Phase 2: Validation
    
    Steps:
    1. Load synthetic data from Phase 1
    2. Validate against RepairQA schema
    3. Save valid samples
    4. Generate validation report
    """
    print("=" * 70)
    print("PHASE 2: DATA VALIDATION")
    print("Structural Quality Check")
    print("=" * 70)
    print()
    
    try:
        # Load synthetic data (JSONL)
        raw_data = load_synthetic_data("data/baseline.jsonl")

        # Validate samples
        valid_samples, validation_errors = validate_samples(raw_data)

        # Save validated data (JSONL)
        if valid_samples:
            save_validated_data(valid_samples, "data/validated_baseline.jsonl")
        
        # Generate report
        generate_validation_report(
            total_samples=len(raw_data),
            valid_samples=valid_samples,
            validation_errors=validation_errors,
            output_file="outputs/validation_report.txt"
        )
        
        print("\n" + "=" * 70)
        print("PHASE 2 COMPLETE")
        print("=" * 70)
        
        if valid_samples:
            print(f"\n✓ {len(valid_samples)} samples validated and ready for Phase 3")
        else:
            print("\n✗ No valid samples - please review errors and regenerate data")
    
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run Phase 1 first:")
        print("  python data_generator.py")
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
