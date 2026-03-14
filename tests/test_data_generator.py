"""Tests for data_generator.py scaffolding — no LLM calls required.

Covers tasks #5–6 from the ranked task list:
  #5: Fix data_generator.py — 50 samples, category tracking, metadata
  #6: Update data_generator.py output to JSONL

Citations:
  - _instructions.md L206, L599  (≥50 Q&A pairs per run)
  - _instructions.md L295-309   (5 distinct prompt templates)
  - _instructions.md L311, L605  (all 5 categories represented)
  - _instructions.md L283-291   (generation metadata per item)
  - _instructions.md L570-574   (JSONL for generated data)
  - _instructions.md L576       (separate baseline/corrected filenames)
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import only non-LLM-dependent pieces to avoid needing API keys at test time
from models import RepairQA, GenerationMeta, RepairCategory


# ============================================================================
# Task #5: Prompt template coverage (5 categories)
# ============================================================================

class TestPromptTemplates:
    """Verify all 5 repair categories have prompt templates (L295-309)."""

    def test_all_five_categories_have_templates(self):
        """PROMPT_TEMPLATES must have one key per RepairCategory."""
        # Import at test time so module-level API key check doesn't block
        # We'll need to handle this — the current module raises on import
        # if OPENAI_API_KEY is missing. That's a test isolation problem.
        from data_generator import PROMPT_TEMPLATES

        expected = {c.value for c in RepairCategory}
        actual = set(PROMPT_TEMPLATES.keys())
        assert actual == expected, f"Missing templates: {expected - actual}"

    def test_each_template_is_nonempty_string(self):
        from data_generator import PROMPT_TEMPLATES

        for category, template in PROMPT_TEMPLATES.items():
            assert isinstance(template, str), f"{category} template is not a string"
            assert len(template) > 50, f"{category} template is suspiciously short"


# ============================================================================
# Task #5: Default sample count
# ============================================================================

class TestDefaultSampleCount:
    """Verify default generation produces ≥50 items (L206, L599)."""

    def test_default_num_samples_is_at_least_50(self):
        from data_generator import DEFAULT_NUM_SAMPLES

        assert DEFAULT_NUM_SAMPLES >= 50, (
            f"DEFAULT_NUM_SAMPLES is {DEFAULT_NUM_SAMPLES}, must be ≥50"
        )


# ============================================================================
# Task #6: JSONL serialization
# ============================================================================

class TestJSONLSerialization:
    """Verify save_dataset writes valid JSONL with metadata (L570-574, L283-291)."""

    def test_save_dataset_writes_jsonl(self, valid_repair_qa, tmp_path):
        """Each line must be a valid JSON object."""
        from data_generator import save_dataset

        outfile = tmp_path / "test_output.jsonl"
        meta = GenerationMeta(
            prompt_template="plumbing_repair",
            model="gpt-4o-mini",
        )

        save_dataset(
            dataset=[(valid_repair_qa, meta)],
            output_file=str(outfile),
        )

        lines = outfile.read_text().strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        # Must contain the RepairQA fields
        assert record["question"] == valid_repair_qa.question
        assert record["category"] == "plumbing_repair"
        # Must contain metadata
        assert "metadata" in record
        assert record["metadata"]["prompt_template"] == "plumbing_repair"
        assert record["metadata"]["model"] == "gpt-4o-mini"

    def test_save_dataset_multiple_items(self, valid_repair_qa, tmp_path):
        """Multiple items produce multiple JSONL lines."""
        from data_generator import save_dataset

        items = []
        for i in range(3):
            meta = GenerationMeta(
                prompt_template="plumbing_repair",
                model="gpt-4o-mini",
            )
            items.append((valid_repair_qa, meta))

        outfile = tmp_path / "multi.jsonl"
        save_dataset(dataset=items, output_file=str(outfile))

        lines = outfile.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_save_dataset_creates_parent_dirs(self, valid_repair_qa, tmp_path):
        """Output directory is created if it doesn't exist."""
        from data_generator import save_dataset

        outfile = tmp_path / "nested" / "dir" / "output.jsonl"
        meta = GenerationMeta(prompt_template="hvac_repair", model="gpt-4o-mini")
        save_dataset(dataset=[(valid_repair_qa, meta)], output_file=str(outfile))

        assert outfile.exists()
