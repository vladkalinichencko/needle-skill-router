from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from router.paths import FIGURES, PROCESSED, REPORTS


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _save(fig, name: str) -> str:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path.relative_to(REPORTS))


def _bar(counts: Counter, title: str, name: str, xlabel: str, rotate: int = 60, top: int = 30) -> str:
    items = counts.most_common(top)
    fig, ax = plt.subplots(figsize=(max(7, len(items) * 0.4), 4.2))
    ax.bar(range(len(items)), [v for _, v in items], color="#3b6ea5")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([k for k, _ in items], rotation=rotate, ha="right", fontsize=8)
    ax.set_ylabel("ходов")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, name)


def _share(part: int, total: int, name: str, title: str, ylabel: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    values = [part / total if total else 0, 1 - (part / total if total else 0)]
    ax.bar(["со скиллом", "без скилла"], values, color=["#3b6ea5", "#c9d6e3"])
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: {part} из {total}")
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.1%}", ha="center", fontsize=10)
    return _save(fig, name)


def run(out_dir: Path = PROCESSED) -> dict:
    turns = _rows(out_dir / "turns.jsonl")
    pairs = _rows(out_dir / "dataset.jsonl")
    summary = json.loads((out_dir / "dataset_summary.json").read_text(encoding="utf-8"))

    per_skill = Counter(summary["per_class"])
    unmatched = Counter(summary["unmatched"])
    per_source = Counter(turn["source"] for turn in turns)
    per_lang = Counter(turn["lang"] for turn in turns)
    sidechain = Counter("сабагент" if turn["sidechain"] else "основной" for turn in turns)

    per_dialogue: dict[str, int] = defaultdict(int)
    for turn in turns:
        per_dialogue[turn["session_id"]] += 1
    touched = {pair["session_id"] for pair in pairs}

    figures = {
        "skills": _bar(per_skill, "Сколько ходов вызвало каждый скилл", "skills", "скилл"),
        "unmatched": _bar(unmatched, "Вызовы скиллов вне инвентаря ~/.agents/skills", "unmatched", "имя вызова", top=15),
        "sources": _bar(per_source, "Ходы по источникам", "sources", "источник", rotate=0),
        "languages": _bar(per_lang, "Ходы по языку промпта", "languages", "язык", rotate=0),
        "sidechain": _bar(sidechain, "Ходы: основной агент и сабагенты", "sidechain", "", rotate=0),
        "turns": _share(summary["turns_with_skill"], len(turns), "turns", "Затронуто ходов", "доля ходов"),
        "dialogues": _share(len(touched), len(per_dialogue), "dialogues", "Диалогов со скиллом", "доля диалогов"),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(list(per_dialogue.values()), bins=min(30, max(3, len(per_dialogue))), color="#3b6ea5")
    ax.set_xlabel("ходов в диалоге")
    ax.set_ylabel("диалогов")
    ax.set_title("Длина диалогов")
    figures["lengths"] = _save(fig, "lengths")

    report = {
        "turns": len(turns),
        "turns_with_skill": summary["turns_with_skill"],
        "turns_with_skill_share": round(summary["turns_with_skill"] / len(turns), 4),
        "dialogues": len(per_dialogue),
        "dialogues_with_skill": len(touched),
        "dialogues_with_skill_share": round(len(touched) / len(per_dialogue), 4),
        "turns_per_dialogue": round(len(turns) / len(per_dialogue), 2),
        "pairs": summary["pairs"],
        "classes": summary["classes"],
        "sources": dict(per_source),
        "languages": dict(per_lang),
        "sidechain": dict(sidechain),
        "skills": dict(per_skill.most_common()),
        "unmatched": dict(unmatched.most_common()),
        "figures": figures,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "eda.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
