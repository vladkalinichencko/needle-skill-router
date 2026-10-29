from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from router.paths import ARTIFACTS, PROCESSED
from router.skills import by_name, normalize_invoked
from router.text import redact
from router.turns import load_turns


def build(skills: list[dict], out_dir: Path = PROCESSED) -> dict:
    index = by_name(skills)
    turns = load_turns(out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, negatives, unmatched = [], [], Counter()
    turns_with_skill = 0
    redactions = 0

    for turn in turns:
        matched: list[str] = []
        for raw in turn["skills_raw"]:
            name = normalize_invoked(raw)
            if name in index:
                if name not in matched:
                    matched.append(name)
            else:
                unmatched[raw] += 1
        query, cuts = redact(turn["query"])
        redactions += cuts
        if matched:
            turns_with_skill += 1
            for name in matched:
                skill = index[name]
                pairs.append(
                    {
                        "query": query,
                        "skill": name,
                        "description": skill["description"],
                        "lang": turn["lang"],
                        "source": turn["source"],
                        "sidechain": turn["sidechain"],
                        "session_id": turn["session_id"],
                        "turn_index": turn["index"],
                    }
                )
        elif not turn["skills_raw"]:
            negatives.append(
                {
                    "query": query,
                    "lang": turn["lang"],
                    "source": turn["source"],
                    "sidechain": turn["sidechain"],
                    "session_id": turn["session_id"],
                    "turn_index": turn["index"],
                }
            )

    _dump(pairs, out_dir / "dataset.jsonl")
    _dump(negatives, out_dir / "negatives.jsonl")

    per_class = Counter(pair["skill"] for pair in pairs)
    summary = {
        "pairs": len(pairs),
        "turns": len(turns),
        "turns_with_skill": turns_with_skill,
        "negatives": len(negatives),
        "classes": len(per_class),
        "per_class": dict(per_class.most_common()),
        "unmatched": dict(unmatched.most_common()),
        "redactions": redactions,
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


QUERY_LIMIT = int(os.environ.get("ROUTER_QUERY_LIMIT", "1200"))


def clip_query(text: str, limit: int | None = None) -> tuple[str, bool]:
    limit = QUERY_LIMIT if limit is None else limit
    if not limit or len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + " …", True


def to_needle(pairs_path: Path | None = None, out_dir: Path = PROCESSED) -> Path:
    mapping = json.loads((ARTIFACTS / "tools.map.json").read_text(encoding="utf-8"))
    all_tools = json.loads((ARTIFACTS / "tools.json").read_text(encoding="utf-8"))
    by_skill = {skill: tool for tool, skill in mapping.items()}
    source = pairs_path or out_dir / "dataset.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").split("\n") if line.strip()]

    used = {row["skill"] for row in rows}
    tools = [tool for tool in all_tools if tool["name"] in {by_skill[name] for name in used}]

    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["session_id"], row["turn_index"]), []).append(row)

    path = out_dir / "needle.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for (session_id, turn_index), group in grouped.items():
            answers = [{"name": by_skill[row["skill"]], "arguments": {}} for row in group]
            query, clipped = clip_query(group[0]["query"])
            handle.write(
                json.dumps(
                    {
                        "query": query,
                        "tools": tools,
                        "answers": answers,
                        "meta": {
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "lang": group[0]["lang"],
                            "source": group[0]["source"],
                            "clipped": clipped,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def _dump(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
