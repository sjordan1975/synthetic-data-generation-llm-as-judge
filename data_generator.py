"""
Phase 1: Synthetic Data Generation for Home DIY Repair Q&A

This module generates synthetic home repair Q&A pairs using 5 different
prompt templates (one per repair category). It uses the Instructor library
to ensure LLM outputs match the RepairQA Pydantic model structure.

About Instructor:
Instructor is a Python library that wraps OpenAI's API to enforce structured
outputs. It uses Pydantic models to guarantee the LLM returns valid JSON with
all required fields and correct types. This eliminates the need for manual
parsing and validation of LLM responses.

Generation Strategy:
- 5 prompt templates covering different repair domains
- Random template selection for diversity
- ≥50 total samples generated per run (_instructions.md L206)
- All outputs validated against RepairQA schema
- Each item tracks its category and generation metadata (_instructions.md L283-291)
- Output in JSONL format (_instructions.md L570)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from models import GenerationMeta, RepairQA, RepairCategory
from prompt_loader import load_generation_prompt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_NUM_SAMPLES: int = 50  # _instructions.md L206: ≥50 per run
DEFAULT_MODEL: str = "gpt-4o-mini"
GENERATOR_TEMPERATURE: float = 0.8  # Creative diversity for generation


# ---------------------------------------------------------------------------
# Client initialisation (deferred so tests can import without API keys)
# ---------------------------------------------------------------------------

_client = None  # Lazy singleton


def _get_client():
    """Return an Instructor-wrapped OpenAI client, initialising on first call.

    About dotenv:
    python-dotenv loads environment variables from a .env file into os.environ.
    This keeps sensitive data (like API keys) out of source code and version control.
    The .env.local file should be added to .gitignore to prevent accidental commits.
    """
    global _client
    if _client is not None:
        return _client

    import instructor
    from openai import OpenAI

    # Try project-local .env.local first, then parent directory
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


# =============================================================================
# PROMPT TEMPLATES - loaded from prompts/generation/<version>/
# =============================================================================

# Default prompt version. Override via PROMPT_VERSION env var or function args.
DEFAULT_PROMPT_VERSION: str = "v1"


def get_prompt_templates(version: str = DEFAULT_PROMPT_VERSION) -> dict[str, str]:
    """
    Load all 5 generation prompts for the given version.

    Returns:
        Dict mapping category name to prompt text.
    """
    return {
        cat.value: load_generation_prompt(cat.value, version=version)
        for cat in RepairCategory
    }


# Backwards-compatible module-level reference (lazy-loaded)
PROMPT_TEMPLATES: dict[str, str] = get_prompt_templates()


def generate_repair_qa(
    category: str,
    model: str = DEFAULT_MODEL,
) -> tuple[RepairQA, GenerationMeta]:
    """
    Generate a single repair Q&A pair using the specified category template.

    Args:
        category: One of the 5 repair categories (appliance_repair, etc.)
        model: LLM model identifier to use for generation.

    Returns:
        Tuple of (RepairQA, GenerationMeta) — the validated item plus its
        generation provenance.

    Raises:
        ValueError: If category is not recognized.
        Exception: If LLM generation or validation fails.

    About the Process:
    1. Retrieves the appropriate prompt template for the category
    2. Sends prompt to LLM via Instructor with response_model=RepairQA
    3. Instructor ensures the response matches RepairQA structure
    4. Returns validated RepairQA object (guaranteed to have all fields)
       together with a GenerationMeta recording provenance.
    """
    templates = get_prompt_templates()
    if category not in templates:
        raise ValueError(
            f"Unknown category: {category}. "
            f"Must be one of {list(templates.keys())}"
        )

    prompt = templates[category]
    client = _get_client()

    try:
        # Instructor enforces RepairQA structure on LLM response
        # The response_model parameter tells Instructor to validate against RepairQA
        repair_qa = client.chat.completions.create(
            model=model,
            response_model=RepairQA,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that generates realistic "
                        "home repair Q&A pairs. You MUST set the 'category' "
                        f"field to exactly '{category}'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            # Temperature 0.8 — creative diversity for synthetic data generation
            # (judge uses lower temperature for deterministic evaluation)
            temperature=GENERATOR_TEMPERATURE,
            max_tokens=1000,
        )

        meta = GenerationMeta(
            prompt_template=category,
            model=model,
        )

        return repair_qa, meta

    except Exception as e:
        print(f"Error generating Q&A for category '{category}': {e}")
        raise


def generate_dataset(
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = 42,
    model: str = DEFAULT_MODEL,
) -> list[tuple[RepairQA, GenerationMeta]]:
    """
    Generate a dataset of synthetic repair Q&A pairs.

    Args:
        num_samples: Number of Q&A pairs to generate (default: 50, per L206).
        seed: Random seed for reproducibility.
        model: LLM model identifier.

    Returns:
        List of (RepairQA, GenerationMeta) tuples.

    Process:
    1. Sets random seed for reproducible template selection
    2. For each sample:
       - Randomly selects one of 5 repair categories
       - Generates Q&A pair using that category's template
       - Validates against RepairQA model (automatic via Instructor)
    3. Returns list of all generated samples with metadata

    Why Random Selection:
    Random selection ensures diversity across repair categories while maintaining
    realistic distribution. Over 50 samples, this typically gives good coverage
    of all 5 categories without forcing exact quotas.
    """
    random.seed(seed)  # For reproducible results

    categories = list(PROMPT_TEMPLATES.keys())
    dataset: list[tuple[RepairQA, GenerationMeta]] = []

    print(f"Generating {num_samples} synthetic repair Q&A pairs...")
    print(f"Categories: {', '.join(categories)}\n")

    for i in range(num_samples):
        # Randomly select a category for diversity
        category = random.choice(categories)

        print(f"Sample {i+1}/{num_samples}: Generating {category}...")

        try:
            repair_qa, meta = generate_repair_qa(category, model=model)
            dataset.append((repair_qa, meta))
            print(f"  ✓ Generated: {repair_qa.question[:60]}...")

        except Exception as e:
            print(f"  ✗ Failed to generate sample {i+1}: {e}")
            # Continue with next sample rather than stopping entire generation
            continue

    print(f"\n✓ Successfully generated {len(dataset)} samples")
    return dataset


def save_dataset(
    dataset: list[tuple[RepairQA, GenerationMeta]],
    output_file: str = "data/baseline.jsonl",
) -> None:
    """
    Save the generated dataset to a JSONL file (_instructions.md L570).

    Each line is a JSON object containing the RepairQA fields plus a nested
    "metadata" object with generation provenance.

    Args:
        dataset: List of (RepairQA, GenerationMeta) tuples to save.
        output_file: Output path (default: data/baseline.jsonl).

    About Pydantic Serialization:
    Pydantic models have a .model_dump() method that converts them to dictionaries.
    This ensures proper JSON serialization of all fields, including lists and nested objects.

    About JSONL:
    JSON Lines format stores one JSON object per line. This is streaming-friendly,
    easy to append, and preferred over JSON arrays for generated datasets.
    """
    # Create parent directories if they don't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for repair_qa, meta in dataset:
            record = repair_qa.model_dump(mode="json")
            record["metadata"] = meta.model_dump(mode="json")
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    file_size = os.path.getsize(output_file) / 1024
    print(f"\n✓ Dataset saved to {output_file}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  File size: {file_size:.2f} KB")


def main():
    """
    Main execution function for Phase 1: Data Generation

    Steps:
    1. Generate ≥50 synthetic repair Q&A pairs
    2. Save to data/baseline.jsonl
    3. Display summary statistics including category distribution
    """
    print("=" * 70)
    print("PHASE 1: SYNTHETIC DATA GENERATION")
    print("Home DIY Repair Q&A Pairs")
    print("=" * 70)
    print()

    # Generate dataset
    dataset = generate_dataset(num_samples=DEFAULT_NUM_SAMPLES, seed=42)

    # Save to JSONL
    save_dataset(dataset, output_file="data/baseline.jsonl")

    # Display category distribution (now tracked per item)
    from collections import Counter

    cat_counts = Counter(qa.category.value for qa, _meta in dataset)
    print("\nCategory Distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review data/baseline.jsonl")
    print("2. Proceed to Phase 2: Validation")


if __name__ == "__main__":
    main()
