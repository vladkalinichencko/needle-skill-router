from __future__ import annotations

import json
import re
from pathlib import Path

from router.paths import PROCESSED

CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def language(text: str) -> str:
    cyr = len(CYRILLIC.findall(text))
    lat = sum(ch.isalpha() and ch.isascii() for ch in text)
    if not cyr:
        return "en" if lat else "none"
    if not lat:
        return "ru"
    ratio = cyr / (cyr + lat)
    return "ru" if ratio > 0.8 else "en" if ratio < 0.2 else "mixed"


def from_dialogue(dialogue: dict) -> list[dict]:
    turns = []
    for turn in dialogue["turns"]:
        invoked: list[str] = []
        for call in turn["skill_calls"]:
            if call["skill"] not in invoked:
                invoked.append(call["skill"])
        turns.append(
            {
                "session_id": dialogue["session_id"],
                "source": dialogue["source"],
                "project": dialogue.get("project", ""),
                "sidechain": turn.get("sidechain", False),
                "index": turn["index"],
                "query": turn["user"],
                "lang": language(turn["user"]),
                "skills_raw": invoked,
                "tool_calls": len(turn["tool_calls"]),
            }
        )
    return turns


def write_turns(turns: list[dict], out_dir: Path = PROCESSED) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "turns.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for turn in turns:
            handle.write(json.dumps(turn, ensure_ascii=False) + "\n")
    return path


def load_turns(path: Path | None = None, out_dir: Path = PROCESSED) -> list[dict]:
    path = path or out_dir / "turns.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
