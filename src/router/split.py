from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from router.paths import PROCESSED

SEED = 42
VAL_SHARE = 0.1


def _tokenizer():
    from needle.model.run import load_checkpoint
    from needle.model.tokenizer import get_tokenizer

    from router.paths import ROOT

    params, config = load_checkpoint(str(ROOT / "checkpoints" / "needle3.safetensors"))
    return get_tokenizer(config.vocab_size)


def _lengths(rows: list[dict]) -> list[int]:
    from needle.model.finetune import render_example
    from needle.model.run import load_checkpoint
    from needle.model.tokenizer import get_tokenizer

    from router.paths import ROOT

    params, config = load_checkpoint(str(ROOT / "checkpoints" / "needle3.safetensors"))
    tokenizer = get_tokenizer(config.vocab_size)
    return [
        len(tokenizer.encode(render_example(row)[0])) + len(tokenizer.encode(render_example(row)[1])) + 2
        for row in rows
    ]


def split(
    out_dir: Path = PROCESSED,
    seed: int = SEED,
    val_share: float = VAL_SHARE,
    max_tokens: int | None = None,
) -> dict:
    rows = [
        json.loads(line)
        for line in (out_dir / "needle.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    truncated = 0
    if max_tokens:
        from needle.model.finetune import render_example

        from router.text import fit_query

        tokenizer = _tokenizer()
        fixed = []
        for row in rows:
            if len(tokenizer.encode(render_example(row)[0])) + len(tokenizer.encode(render_example(row)[1])) + 2 <= max_tokens:
                fixed.append(row)
                continue
            empty = {**row, "query": ""}
            budget = max_tokens - len(tokenizer.encode(render_example(empty)[0])) - len(tokenizer.encode(render_example(empty)[1])) - 8
            query, cut = fit_query(row["query"], tokenizer, budget)
            fixed.append({**row, "query": query, "meta": {**row["meta"], "truncated": cut}})
            truncated += int(cut)
        rows = fixed

    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_session[row["meta"]["session_id"]].append(row)

    sessions = sorted(by_session)
    random.Random(seed).shuffle(sessions)

    target = int(len(rows) * val_share)
    val_sessions: set[str] = set()
    taken = 0
    for session in sessions:
        if taken >= target:
            break
        val_sessions.add(session)
        taken += len(by_session[session])

    train_rows = [row for row in rows if row["meta"]["session_id"] not in val_sessions]
    val_rows = [row for row in rows if row["meta"]["session_id"] in val_sessions]

    _dump(train_rows, out_dir / "train.jsonl")
    _dump(val_rows, out_dir / "val.jsonl")
    return {
        "train": len(train_rows),
        "val": len(val_rows),
        "val_sessions": len(val_sessions),
        "truncated_middle": truncated,
        "max_tokens": max_tokens,
        "seed": seed,
    }


def _dump(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
