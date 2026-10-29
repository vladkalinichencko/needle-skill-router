from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_SKILLS = ROOT / "data" / "raw" / "skills"
RAW_DIALOGUES = ROOT / "data" / "raw" / "dialogues"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "eda"

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
CODEX_SESSIONS = Path.home() / ".codex" / "sessions"

SERVICE_PREFIXES = ("<system-reminder>", "Current date:", "Ответ на вопрос")

CODEX_THREADS_DB = Path.home() / ".codex" / "thread_history_1.sqlite"
CODEX_STATE_DB = Path.home() / ".codex" / "state_5.sqlite"
