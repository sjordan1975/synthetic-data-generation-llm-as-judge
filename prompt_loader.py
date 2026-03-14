"""Prompt loader for versioned prompt templates.

Loads prompt text from the ``prompts/`` directory tree, enabling:
- Clean separation of pipeline code (stable) from prompt content (iterable)
- Prompt versioning: ``prompts/generation/v1/`` → ``prompts/generation/v2/``
- Easy diffing and narrative in the README about what changed and why

Directory layout::

    prompts/
      generation/
        v1/                   # one .md per RepairCategory
          appliance_repair.md
          plumbing_repair.md
          ...
      judge/
        v1/
          system.md           # judge evaluation template (with {placeholders})

About Versioning:
Each version directory is a complete snapshot of all prompts used in a single
pipeline run. When prompt corrections are made (data-driven, post-analysis),
a new version directory is created rather than editing in place, preserving
the full iteration history.
"""

from __future__ import annotations

from pathlib import Path

# Root of the prompts directory (sibling of this file)
_PROMPTS_ROOT = Path(__file__).resolve().parent / "prompts"


def load_generation_prompt(
    category: str,
    version: str = "v1",
) -> str:
    """
    Load a generation system prompt for the given repair category.

    Args:
        category: RepairCategory value (e.g. 'plumbing_repair').
        version: Prompt version directory name (e.g. 'v1', 'v2').

    Returns:
        The prompt text as a string.

    Raises:
        FileNotFoundError: If no prompt file exists for the category/version.
    """
    path = _PROMPTS_ROOT / "generation" / version / f"{category}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Generation prompt not found: {path}\n"
            f"Expected a .md file for category '{category}' in version '{version}'."
        )
    return path.read_text(encoding="utf-8").strip()


def load_judge_template(
    version: str = "v1",
) -> str:
    """
    Load the judge system prompt template.

    The template contains ``{placeholder}`` tokens that the caller fills
    via ``str.format()`` or ``str.format_map()``.

    Args:
        version: Prompt version directory name.

    Returns:
        The template text with unfilled placeholders.

    Raises:
        FileNotFoundError: If the template file doesn't exist.
    """
    path = _PROMPTS_ROOT / "judge" / version / "system.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Judge prompt template not found: {path}\n"
            f"Expected system.md in version '{version}'."
        )
    return path.read_text(encoding="utf-8").strip()


def available_generation_versions() -> list[str]:
    """Return sorted list of available generation prompt versions."""
    gen_dir = _PROMPTS_ROOT / "generation"
    if not gen_dir.exists():
        return []
    return sorted(d.name for d in gen_dir.iterdir() if d.is_dir())


def available_judge_versions() -> list[str]:
    """Return sorted list of available judge prompt versions."""
    judge_dir = _PROMPTS_ROOT / "judge"
    if not judge_dir.exists():
        return []
    return sorted(d.name for d in judge_dir.iterdir() if d.is_dir())
