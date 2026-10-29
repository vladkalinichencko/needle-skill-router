from __future__ import annotations

import re

SECRET_PATTERNS = (
    r"sk-[A-Za-z0-9_-]{20,}",
    r"ghp_[A-Za-z0-9]{20,}",
    r"gho_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"needle_ft_[A-Za-z0-9]{10,}",
    r"hf_[A-Za-z0-9]{30,}",
    r"hf_[A-Za-z0-9]{20,}",
    r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}",
)
SECRET_RE = re.compile("|".join(f"(?:{pattern})" for pattern in SECRET_PATTERNS))
REDACTED = "[секрет вырезан]"
CUT = " […] "


def redact(text: str) -> tuple[str, int]:
    return SECRET_RE.subn(REDACTED, text)


def fit_query(query: str, tokenizer, budget: int) -> tuple[str, bool]:
    if budget <= 0 or len(tokenizer.encode(query)) <= budget:
        return query, False
    low, high = 0, len(query)
    while low < high:
        middle = (low + high + 1) // 2
        head = middle // 2
        tail = middle - head
        candidate = query[:head] + CUT + query[len(query) - tail:] if middle else CUT
        if len(tokenizer.encode(candidate)) <= budget:
            low = middle
        else:
            high = middle - 1
    head = low // 2
    tail = low - head
    return query[:head] + CUT + query[len(query) - tail:], True
