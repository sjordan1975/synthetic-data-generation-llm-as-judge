"""
pytest configuration and fixtures for synthetic-data-generator tests.

Shared fixtures provide valid model instances that individual tests can
override to check specific constraints.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import RepairQA  # noqa: E402


# ---------------------------------------------------------------------------
# RepairQA fixtures
# ---------------------------------------------------------------------------

VALID_REPAIR_QA_DATA: dict = {
    "question": "How do I fix a leaking kitchen faucet?",
    "answer": (
        "To fix a leaking faucet, first turn off the water supply under the sink. "
        "Remove the faucet handle by unscrewing the set screw, then remove the packing "
        "nut to access the stem. Replace the worn washer or O-ring with a new one of "
        "the same size. Reassemble the faucet and turn the water back on to test."
    ),
    "equipment_problem": "Leaking kitchen faucet - worn washer",
    "tools_required": ["adjustable wrench", "screwdriver", "replacement washer"],
    "steps": [
        "Turn off water supply under sink",
        "Remove faucet handle by unscrewing set screw",
        "Replace worn washer or O-ring",
        "Reassemble faucet and test for leaks",
    ],
    "safety_info": (
        "Always turn off the water supply before starting any plumbing repair. "
        "Keep a bucket handy to catch any residual water."
    ),
    "tips": [
        "Take a photo before disassembly to remember how parts fit together.",
        "Bring the old washer to the hardware store to ensure correct replacement size.",
    ],
    "category": "plumbing_repair",
}


@pytest.fixture
def valid_repair_qa_data() -> dict:
    """Return a mutable copy of valid RepairQA payload."""
    return {**VALID_REPAIR_QA_DATA, "tips": list(VALID_REPAIR_QA_DATA["tips"])}


@pytest.fixture
def valid_repair_qa(valid_repair_qa_data) -> RepairQA:
    """Return a validated RepairQA instance."""
    return RepairQA(**valid_repair_qa_data)


# ---------------------------------------------------------------------------
# pytest configuration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
