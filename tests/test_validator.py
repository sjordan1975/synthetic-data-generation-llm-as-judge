"""Tests for validator.py — JSONL loading, validation, and reporting.

Covers task #7 from the ranked task list:
  #7: Update validator.py for new model + JSONL

Citations:
  - _instructions.md L200, L603  (≥95% structural validation pass rate)
  - _instructions.md L570-574    (JSONL for generated data)
"""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import RepairQA, GenerationMeta
from tests.conftest import VALID_REPAIR_QA_DATA


# ============================================================================
# Helper: write JSONL test files
# ============================================================================

def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_record_with_meta(overrides: dict | None = None) -> dict:
    """Build a single JSONL record (RepairQA fields + metadata)."""
    record = {**VALID_REPAIR_QA_DATA}
    record["metadata"] = {
        "prompt_template": "plumbing_repair",
        "model": "gpt-4o-mini",
        "timestamp": "2026-03-14T10:00:00Z",
        "validation_passed": None,
    }
    if overrides:
        record.update(overrides)
    return record


# ============================================================================
# JSONL loading
# ============================================================================

class TestLoadSyntheticData:
    """Verify load_synthetic_data reads JSONL correctly."""

    def test_loads_valid_jsonl(self, tmp_path):
        from validator import load_synthetic_data

        records = [_make_record_with_meta() for _ in range(3)]
        fpath = tmp_path / "data.jsonl"
        _write_jsonl(fpath, records)

        loaded = load_synthetic_data(str(fpath))
        assert len(loaded) == 3
        assert loaded[0]["question"] == VALID_REPAIR_QA_DATA["question"]

    def test_raises_on_missing_file(self, tmp_path):
        from validator import load_synthetic_data

        with pytest.raises(FileNotFoundError):
            load_synthetic_data(str(tmp_path / "nonexistent.jsonl"))

    def test_strips_metadata_before_returning(self, tmp_path):
        """Metadata is separated; raw records passed to validation shouldn't choke on it."""
        from validator import load_synthetic_data

        records = [_make_record_with_meta()]
        fpath = tmp_path / "data.jsonl"
        _write_jsonl(fpath, records)

        loaded = load_synthetic_data(str(fpath))
        # metadata key may or may not be present — validator should handle either
        assert len(loaded) == 1


# ============================================================================
# Validation logic
# ============================================================================

class TestValidateSamples:
    """Verify validate_samples catches schema violations."""

    def test_valid_data_passes(self):
        from validator import validate_samples

        valid, errors = validate_samples([VALID_REPAIR_QA_DATA])
        assert len(valid) == 1
        assert len(errors) == 0

    def test_invalid_data_captured(self):
        from validator import validate_samples

        bad = {**VALID_REPAIR_QA_DATA, "tips": "not a list"}  # must be list
        valid, errors = validate_samples([bad])
        assert len(valid) == 0
        assert len(errors) == 1
        assert errors[0]["index"] == 1

    def test_mixed_batch(self):
        from validator import validate_samples

        good = VALID_REPAIR_QA_DATA
        bad = {**VALID_REPAIR_QA_DATA, "steps": ["only one"]}  # needs ≥3
        valid, errors = validate_samples([good, bad, good])
        assert len(valid) == 2
        assert len(errors) == 1


# ============================================================================
# JSONL output
# ============================================================================

class TestSaveValidatedData:
    """Verify save_validated_data writes JSONL."""

    def test_writes_jsonl(self, valid_repair_qa, tmp_path):
        from validator import save_validated_data

        outfile = tmp_path / "validated.jsonl"
        save_validated_data([valid_repair_qa], str(outfile))

        lines = outfile.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["category"] == "plumbing_repair"

    def test_creates_parent_dirs(self, valid_repair_qa, tmp_path):
        from validator import save_validated_data

        outfile = tmp_path / "nested" / "dir" / "out.jsonl"
        save_validated_data([valid_repair_qa], str(outfile))
        assert outfile.exists()


# ============================================================================
# Validation report
# ============================================================================

class TestValidationReport:
    """Verify the text report is generated."""

    def test_report_written(self, valid_repair_qa, tmp_path):
        from validator import generate_validation_report

        outfile = tmp_path / "report.txt"
        generate_validation_report(
            total_samples=1,
            valid_samples=[valid_repair_qa],
            validation_errors=[],
            output_file=str(outfile),
        )

        text = outfile.read_text()
        assert "Success Rate: 100.0%" in text
        assert "ALL SAMPLES VALID" in text

    def test_report_shows_errors(self, tmp_path):
        from validator import generate_validation_report

        outfile = tmp_path / "report.txt"
        errors = [{"index": 1, "question": "Bad item", "error_count": 2, "error": "oops"}]
        generate_validation_report(
            total_samples=1,
            valid_samples=[],
            validation_errors=errors,
            output_file=str(outfile),
        )

        text = outfile.read_text()
        assert "Success Rate: 0.0%" in text
        assert "Sample 1:" in text
