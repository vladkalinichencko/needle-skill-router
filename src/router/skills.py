from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from router.paths import ARTIFACTS, RAW_SKILLS


def _frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def inventory(root: Path = RAW_SKILLS) -> list[dict]:
    skills = []
    for path in sorted(root.rglob("SKILL.md")):
        meta = _frontmatter(path.read_text(encoding="utf-8", errors="replace"))
        name = str(meta.get("name") or path.parent.name).strip()
        description = " ".join(str(meta.get("description") or "").split())
        skills.append(
            {
                "name": name,
                "dir": path.parent.name,
                "description": description,
                "path": str(path.relative_to(root)),
                "category": path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else "",
            }
        )
    return skills


def by_name(skills: list[dict]) -> dict[str, dict]:
    index = {}
    for skill in skills:
        index[skill["name"]] = skill
        index.setdefault(skill["dir"], skill)
    return index


def short_description(text: str, limit: int = 160) -> str:
    first = text.split(". ")[0].strip() or text.strip()
    if len(first) > limit:
        first = first[:limit].rstrip(" ,;:") + "…"
    return first


def tool_name(skill_name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", skill_name).strip("_").lower()
    return f"s_{slug}" if slug[:1].isdigit() else slug


def normalize_invoked(name: str) -> str:
    return name.split(":")[-1].strip()


def build_tools(skills: list[dict], out_dir: Path = ARTIFACTS) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tools = [
        {
            "name": tool_name(skill["name"]),
            "description": short_description(skill["description"]),
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
        for skill in skills
    ]
    (out_dir / "tools.json").write_text(
        json.dumps(tools, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    mapping = {tool_name(skill["name"]): skill["name"] for skill in skills}
    (out_dir / "tools.map.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return tools
