from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from router.paths import CLAUDE_PROJECTS, CODEX_SESSIONS, RAW_DIALOGUES, SERVICE_PREFIXES

SKILL_PATH = re.compile(r"([\w.@+-]+)/SKILL\.md")
CODEX_NOISE = ("<environment_context", "<user_instructions", "# AGENTS.md")
MAX_INPUT_CHARS = 4000


def _clip(value, limit: int = MAX_INPUT_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def _blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _join_text(blocks: list[dict], kinds: tuple[str, ...]) -> str:
    return "\n".join(
        str(b.get("text") or "").strip()
        for b in blocks
        if b.get("type") in kinds and str(b.get("text") or "").strip()
    ).strip()


def _is_service(text: str) -> bool:
    return text.startswith(SERVICE_PREFIXES) or text.startswith(CODEX_NOISE)


def _new_turn(dialogue: dict, text: str, timestamp: str | None, sidechain: bool) -> dict:
    turn = {
        "index": len(dialogue["turns"]),
        "user": text,
        "timestamp": timestamp,
        "sidechain": sidechain,
        "assistant_text": [],
        "tool_calls": [],
        "skill_calls": [],
    }
    dialogue["turns"].append(turn)
    return turn


def parse_claude(path: Path) -> dict:
    dialogue = {
        "source": "claude",
        "session_id": path.stem,
        "project": path.parent.name,
        "path": str(path),
        "title": "",
        "cwd": "",
        "started_at": None,
        "sidechain": False,
        "turns": [],
    }
    current: dict | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = record.get("type")
            if kind == "ai-title":
                dialogue["title"] = record.get("aiTitle") or ""
                continue
            if kind not in ("user", "assistant"):
                continue
            dialogue["cwd"] = dialogue["cwd"] or record.get("cwd") or ""
            sidechain = bool(record.get("isSidechain"))
            dialogue["sidechain"] = dialogue["sidechain"] or sidechain
            blocks = _blocks((record.get("message") or {}).get("content"))

            if kind == "user":
                text = _join_text(blocks, ("text",))
                if not text or _is_service(text):
                    continue
                current = _new_turn(dialogue, text, record.get("timestamp"), sidechain)
                dialogue["started_at"] = dialogue["started_at"] or record.get("timestamp")
                continue

            if current is None:
                continue
            assistant = _join_text(blocks, ("text",))
            if assistant:
                current["assistant_text"].append(assistant)
            for block in blocks:
                if block.get("type") != "tool_use":
                    continue
                payload = block.get("input") or {}
                name = block.get("name") or ""
                current["tool_calls"].append({"name": name, "input": payload})
                if name == "Skill" and payload.get("skill"):
                    current["skill_calls"].append(
                        {"skill": payload["skill"], "args": _clip(payload.get("args") or "", 500)}
                    )
    return dialogue


def parse_codex(path: Path) -> dict:
    dialogue = {
        "source": "codex",
        "session_id": path.stem,
        "project": "-".join(path.parts[-4:-1]),
        "path": str(path),
        "title": "",
        "cwd": "",
        "started_at": None,
        "sidechain": False,
        "turns": [],
    }
    current: dict | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = record.get("type")
            payload = record.get("payload") or {}

            if kind == "session_meta":
                dialogue["session_id"] = payload.get("id") or dialogue["session_id"]
                dialogue["cwd"] = payload.get("cwd") or ""
                dialogue["started_at"] = payload.get("timestamp")
                continue
            if kind != "response_item":
                continue

            item = payload.get("type")
            if item == "message":
                role = payload.get("role")
                text = _join_text(_blocks(payload.get("content")), ("text", "input_text", "output_text"))
                if role == "user":
                    if not text or _is_service(text):
                        continue
                    current = _new_turn(dialogue, text, payload.get("timestamp"), False)
                elif role == "assistant" and current and text:
                    current["assistant_text"].append(text)
                continue

            if current is None:
                continue
            if item in ("function_call", "custom_tool_call"):
                raw = payload.get("arguments") or payload.get("input") or ""
                name = payload.get("name") or item
                current["tool_calls"].append({"name": name, "input": _clip(raw)})
                for skill_dir in SKILL_PATH.findall(str(raw)):
                    call = {"skill": skill_dir, "args": ""}
                    if call not in current["skill_calls"]:
                        current["skill_calls"].append(call)
    return dialogue


def claude_files(root: Path = CLAUDE_PROJECTS) -> list[Path]:
    return sorted(root.rglob("*.jsonl"))


def codex_files(root: Path = CODEX_SESSIONS) -> list[Path]:
    return sorted(root.rglob("*.jsonl"))


def dialogue_path(dialogue: dict, root: Path = RAW_DIALOGUES) -> Path:
    folder = dialogue.get("project") or "unknown"
    target = root / dialogue["source"] / folder / f"{dialogue['session_id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write(dialogue: dict, root: Path = RAW_DIALOGUES) -> Path:
    target = dialogue_path(dialogue, root)
    target.write_text(json.dumps(dialogue, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return target


def load_all(root: Path = RAW_DIALOGUES) -> Iterator[dict]:
    for path in sorted(root.rglob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


UUID = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")


def uuid_from_stem(stem: str) -> str:
    found = UUID.findall(stem)
    return found[-1] if found else stem


def _readonly(db: Path) -> "sqlite3.Connection":
    import sqlite3

    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def _thread_meta(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    conn = _readonly(path)
    try:
        rows = conn.execute("select id, title, cwd, archived from threads").fetchall()
    finally:
        conn.close()
    return {row[0]: {"title": row[1] or "", "cwd": row[2] or "", "archived": bool(row[3])} for row in rows}


def iter_codex_threads(db: Path | None = None, state_db: Path | None = None) -> Iterator[dict]:
    from router.paths import CODEX_STATE_DB, CODEX_THREADS_DB

    db = db or CODEX_THREADS_DB
    if not db.exists():
        return
    meta = _thread_meta(state_db or CODEX_STATE_DB)
    conn = _readonly(db)
    try:
        thread_ids = [row[0] for row in conn.execute("select distinct thread_id from thread_items order by thread_id")]
        for thread_id in thread_ids:
            info = meta.get(thread_id, {})
            dialogue = {
                "source": "codex",
                "session_id": thread_id,
                "project": "threads",
                "path": str(db),
                "title": info.get("title", ""),
                "cwd": info.get("cwd", ""),
                "started_at": None,
                "sidechain": False,
                "turns": [],
            }
            current = None
            rows = conn.execute(
                "select item_type, item_json, created_at_ms from thread_items"
                " where thread_id = ? order by rollout_ordinal, created_at_ms",
                (thread_id,),
            )
            for item_type, raw, created_at in rows:
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                stamp = str(created_at) if created_at else None
                if item_type == "userMessage":
                    text = _join_text(_blocks(payload.get("content")), ("text",))
                    if not text or _is_service(text):
                        continue
                    current = _new_turn(dialogue, text, stamp, False)
                    dialogue["started_at"] = dialogue["started_at"] or stamp
                    continue
                if current is None:
                    continue
                if item_type == "agentMessage":
                    text = str(payload.get("text") or "").strip()
                    if text:
                        current["assistant_text"].append(text)
                    continue
                if item_type == "commandExecution":
                    command = payload.get("command") or ""
                    current["tool_calls"].append({"name": "command", "input": _clip(command)})
                    dialogue["cwd"] = dialogue["cwd"] or payload.get("cwd") or ""
                    for skill_dir in SKILL_PATH.findall(str(command)):
                        call = {"skill": skill_dir, "args": ""}
                        if call not in current["skill_calls"]:
                            current["skill_calls"].append(call)
            if dialogue["turns"]:
                yield dialogue
    finally:
        conn.close()
