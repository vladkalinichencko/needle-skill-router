from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from router.infer import Router
from router.paths import PROCESSED


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def run(
    weights: str | None = None,
    data_path: Path | None = None,
    out_dir: Path = PROCESSED,
    min_confidence: float | None = None,
    negatives_limit: int = 300,
) -> dict:
    router = Router(weights=weights)
    rows = _rows(data_path or out_dir / "val.jsonl")
    negatives = _rows(out_dir / "negatives.jsonl")

    hits = defaultdict(lambda: {"support": 0, "hit": 0})
    exact = 0
    wrong = []
    for row in rows:
        expected = [answer["name"] for answer in row["answers"]]
        result = router.predict(row["query"])
        predicted = result["calls"]
        if min_confidence is not None and result["confidence"] is not None and result["confidence"] < min_confidence:
            predicted = []
        if sorted(set(predicted)) == sorted(set(expected)):
            exact += 1
        else:
            wrong.append({"query": row["query"], "expected": expected, "predicted": predicted})
        for name in expected:
            hits[name]["support"] += 1
            hits[name]["hit"] += int(name in predicted)

    refused = 0
    for row in negatives[:negatives_limit]:
        if not router.predict(row["query"])["calls"]:
            refused += 1

    return {
        "examples": len(rows),
        "exact_match": round(exact / len(rows), 4) if rows else 0.0,
        "negatives_checked": min(len(negatives), negatives_limit),
        "refusal_rate": round(refused / min(len(negatives), negatives_limit), 4) if negatives else None,
        "per_class": {
            name: {**value, "recall": round(value["hit"] / value["support"], 3)}
            for name, value in sorted(hits.items())
        },
        "wrong": wrong[:30],
    }
