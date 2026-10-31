from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from router.paths import PROCESSED, dump_jsonl

SEED = 42
VAL_SHARE = 0.1


def _tokenizer():
    from needle.model.run import load_checkpoint
    from needle.model.tokenizer import get_tokenizer

    from router.paths import ROOT

    params, config = load_checkpoint(str(ROOT / "checkpoints" / "needle3.safetensors"))
    return get_tokenizer(config.vocab_size)


def _answers(row: dict) -> set[str]:
    return {answer["name"] for answer in row["answers"]}


def _stratify(rows: list[dict], seed: int, val_share: float) -> tuple[list[dict], list[dict], int]:
    by_session: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_session[row["meta"]["session_id"]].append(row)

    session_classes = {
        session: set().union(*(_answers(row) for row in session_rows))
        for session, session_rows in by_session.items()
    }
    class_sessions: dict[str, set[str]] = defaultdict(set)
    for session, classes in session_classes.items():
        for cls in classes:
            class_sessions[cls].add(session)

    rng = random.Random(seed)
    sessions = sorted(by_session)
    rng.shuffle(sessions)

    # класс, чьи пары живут в одной сессии, весь уходит в train: иначе train его не увидит
    locked = {s for class_sessions_ in class_sessions.values() if len(class_sessions_) == 1 for s in class_sessions_}

    val_sessions: set[str] = set()

    def orphans(session: str) -> bool:
        return any(
            class_sessions[cls] <= val_sessions | {session}
            for cls in session_classes[session]
        )

    # каждому классу с парой в нескольких сессиях — одна сессия в val, начиная с редких
    class_rows = Counter(cls for row in rows for cls in _answers(row))
    for cls in sorted(class_rows, key=lambda c: (class_rows[c], c)):
        candidates = [
            s for s in class_sessions[cls]
            if s not in locked and s not in val_sessions and not orphans(s)
        ]
        if candidates:
            val_sessions.add(min(candidates, key=lambda s: (len(by_session[s]), s)))

    target = int(len(rows) * val_share)
    taken = sum(len(by_session[s]) for s in val_sessions)
    for session in sessions:
        if taken >= target:
            break
        if session in locked or session in val_sessions or orphans(session):
            continue
        val_sessions.add(session)
        taken += len(by_session[session])

    train_rows = [row for row in rows if row["meta"]["session_id"] not in val_sessions]
    val_rows = [row for row in rows if row["meta"]["session_id"] in val_sessions]
    return train_rows, val_rows, len(val_sessions)


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

    train_rows, val_rows, n_val_sessions = _stratify(rows, seed, val_share)

    dump_jsonl(train_rows, out_dir / "train.jsonl")
    dump_jsonl(val_rows, out_dir / "val.jsonl")
    train_classes = set().union(*(_answers(row) for row in train_rows))
    val_classes = set().union(*(_answers(row) for row in val_rows))
    return {
        "train": len(train_rows),
        "val": len(val_rows),
        "val_sessions": n_val_sessions,
        "classes_train": len(train_classes),
        "classes_val": len(val_classes),
        "classes_val_only": sorted(val_classes - train_classes),
        "truncated_middle": truncated,
        "max_tokens": max_tokens,
        "seed": seed,
    }
