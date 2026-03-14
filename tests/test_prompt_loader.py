"""Tests for prompt_loader.py — versioned prompt loading utility.

Covers:
- Loading generation prompts by category + version
- Loading judge prompt template with placeholders
- Version discovery
- Error handling for missing files
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import RepairCategory


# ============================================================================
# Generation prompt loading
# ============================================================================

class TestLoadGenerationPrompt:

    def test_loads_all_v1_categories(self):
        from prompt_loader import load_generation_prompt

        for cat in RepairCategory:
            prompt = load_generation_prompt(cat.value, version="v1")
            assert isinstance(prompt, str)
            assert len(prompt) > 50, f"{cat.value} prompt is suspiciously short"

    def test_raises_on_unknown_category(self):
        from prompt_loader import load_generation_prompt

        with pytest.raises(FileNotFoundError):
            load_generation_prompt("nonexistent_category", version="v1")

    def test_raises_on_unknown_version(self):
        from prompt_loader import load_generation_prompt

        with pytest.raises(FileNotFoundError):
            load_generation_prompt("appliance_repair", version="v99")


# ============================================================================
# Judge prompt loading
# ============================================================================

class TestLoadJudgeTemplate:

    def test_loads_v1_template(self):
        from prompt_loader import load_judge_template

        template = load_judge_template(version="v1")
        assert isinstance(template, str)
        assert len(template) > 100

    def test_template_has_placeholders(self):
        from prompt_loader import load_judge_template

        template = load_judge_template(version="v1")
        expected_placeholders = [
            "{trace_id}", "{category}", "{question}", "{answer}",
            "{equipment_problem}", "{tools_required}", "{steps}",
            "{safety_info}", "{tips}",
        ]
        for ph in expected_placeholders:
            assert ph in template, f"Missing placeholder: {ph}"

    def test_template_can_be_formatted(self):
        from prompt_loader import load_judge_template

        template = load_judge_template(version="v1")
        filled = template.format(
            trace_id="qa_001",
            category="plumbing_repair",
            question="How do I fix a leak?",
            answer="Turn off water, replace washer.",
            equipment_problem="Leaking faucet",
            tools_required="wrench, washer",
            steps="  1. Turn off water\n  2. Replace washer",
            safety_info="Turn off water supply first.",
            tips="  - Bring old washer to store to match size.",
        )
        assert "qa_001" in filled
        assert "plumbing_repair" in filled
        assert "{trace_id}" not in filled  # All placeholders resolved

    def test_raises_on_unknown_version(self):
        from prompt_loader import load_judge_template

        with pytest.raises(FileNotFoundError):
            load_judge_template(version="v99")


# ============================================================================
# Version discovery
# ============================================================================

class TestVersionDiscovery:

    def test_generation_versions_include_v1(self):
        from prompt_loader import available_generation_versions

        versions = available_generation_versions()
        assert "v1" in versions

    def test_judge_versions_include_v1(self):
        from prompt_loader import available_judge_versions

        versions = available_judge_versions()
        assert "v1" in versions
