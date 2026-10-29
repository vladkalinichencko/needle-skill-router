from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from router import dataset as dataset_mod
from router import eda as eda_mod
from router import evaluate as evaluate_mod
from router import sessions as sessions_mod
from router import skills as skills_mod
from router import turns as turns_mod
from router.paths import ARTIFACTS, PROCESSED, RAW_DIALOGUES


def cmd_skills(args: argparse.Namespace) -> None:
    inventory = skills_mod.inventory()
    tools = skills_mod.build_tools(inventory)
    print(f"{len(inventory)} скиллов, {len(tools)} инструментов")


def cmd_collect(args: argparse.Namespace) -> None:
    dialogues = 0
    turns: list[dict] = []
    manifest: list[dict] = []
    seen: set[str] = set()

    def add(dialogue: dict) -> None:
        nonlocal dialogues
        if not dialogue["turns"]:
            return
        sessions_mod.write(dialogue)
        dialogues += 1
        turns.extend(turns_mod.from_dialogue(dialogue))
        manifest.append(
            {
                "session_id": dialogue["session_id"],
                "project": dialogue.get("project", ""),
                "source": dialogue["source"],
                "title": dialogue.get("title", ""),
                "cwd": dialogue.get("cwd", ""),
                "sidechain": dialogue["sidechain"],
                "turns": len(dialogue["turns"]),
            }
        )

    for dialogue in sessions_mod.iter_codex_threads():
        seen.add(dialogue["session_id"])
        add(dialogue)

    for path in sessions_mod.claude_files():
        dialogue = sessions_mod.parse_claude(path)
        if dialogue["session_id"] in seen:
            continue
        add(dialogue)

    for path in sessions_mod.codex_files():
        key = sessions_mod.uuid_from_stem(path.stem)
        if key in seen:
            continue
        seen.add(key)
        add(sessions_mod.parse_codex(path))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    turns_mod.write_turns(turns)
    (PROCESSED / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({"dialogues": dialogues, "turns": len(turns)}, ensure_ascii=False))


def cmd_dataset(args: argparse.Namespace) -> None:
    inventory = skills_mod.inventory()
    summary = dataset_mod.build(inventory)
    needle = dataset_mod.to_needle()
    summary["needle_rows"] = sum(1 for _ in needle.open(encoding="utf-8"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    report = evaluate_mod.run(
        weights=args.weights,
        negatives_limit=args.negatives,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_eda(args: argparse.Namespace) -> None:
    print(json.dumps(eda_mod.run(), ensure_ascii=False, indent=2))


def cmd_all(args: argparse.Namespace) -> None:
    cmd_skills(args)
    cmd_collect(args)
    cmd_dataset(args)
    cmd_eda(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="router")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, help_text in (
        ("skills", cmd_skills, "инвентарь скиллов и tools.json"),
        ("collect", cmd_collect, "нормализованные копии диалогов и ходы"),
        ("dataset", cmd_dataset, "датасет пар и негативы"),
        ("eda", cmd_eda, "статистика и графики"),
        ("eval", cmd_eval, "оценка модели на отложенных диалогах"),
        ("all", cmd_all, "всё подряд"),
    ):
        parser_ = sub.add_parser(name, help=help_text)
        parser_.set_defaults(func=func)
        if name == "eval":
            parser_.add_argument("--weights")
            parser_.add_argument("--negatives", type=int, default=60)
            parser_.add_argument("--min-confidence", type=float)
            parser_.add_argument("--out")

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
