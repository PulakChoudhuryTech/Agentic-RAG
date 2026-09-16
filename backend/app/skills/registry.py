"""
Skill registry: loads backend/app/skills/<domain>/SKILL.md into Skill
objects. See docs/concepts/skills_pattern.md for the full rationale; in
short, this replicates the *pattern* behind Anthropic's Agent Skills
(progressive disclosure: cheap always-on metadata, expensive content loaded
only when triggered) without their runtime, which only exists inside
Claude's API/Code-execution container -- this project is 100% Gemini.

What each part of a Skill is used for:
  - `name` + `description`: ALWAYS loaded -- agents/supervisor.py builds its
    routing prompt from every skill's description, so adding a new domain
    is "add a SKILL.md", not "edit the routing prompt string".
  - `category` + `keywords`: loaded once and cached -- rag/metadata_router.py
    reads these to build its rule-based category filter table.
  - `body`: loaded only for the skill that was actually triggered --
    agents/rag_agent.py folds it into the final answer prompt as
    domain-specific guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

SKILLS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Skill:
    name: str  # matches a LangGraph route label, e.g. "rag_hr"
    description: str  # the routing rule shown in the supervisor prompt
    category: str | None = None  # documents.category this skill scopes retrieval to, or None
    keywords: list[str] = field(default_factory=list)
    match_priority: int = 100  # lower = checked first by metadata_router's category matching
    body: str = ""  # per-domain answer guidance, injected only when this skill is triggered


def _parse_skill_md(path: Path) -> Skill:
    text = path.read_text()
    if not text.startswith("---\n"):
        raise ValueError(f"{path} is missing YAML frontmatter")
    _, frontmatter_raw, body = text.split("---\n", 2)
    frontmatter = yaml.safe_load(frontmatter_raw) or {}
    return Skill(
        name=frontmatter["name"],
        description=frontmatter["description"],
        category=frontmatter.get("category"),
        keywords=frontmatter.get("keywords") or [],
        match_priority=frontmatter.get("match_priority", 100),
        body=body.strip(),
    )


@lru_cache(maxsize=1)
def load_skills() -> tuple[Skill, ...]:
    """Load every backend/app/skills/*/SKILL.md once per process -- cached
    like config.get_settings(), since these are static files read at
    startup, not something that changes per request."""
    skills = tuple(_parse_skill_md(skill_md) for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")))
    if not skills:
        raise RuntimeError(f"no SKILL.md files found under {SKILLS_DIR}")
    return skills


def get_skill(name: str) -> Skill | None:
    return next((s for s in load_skills() if s.name == name), None)
