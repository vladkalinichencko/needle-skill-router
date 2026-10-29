from __future__ import annotations

import json
from pathlib import Path

from router.paths import ARTIFACTS


class Router:
    def __init__(self, tools_path: Path | None = None, weights: str | None = None):
        import needle

        self.tools = json.loads((tools_path or ARTIFACTS / "tools.json").read_text(encoding="utf-8"))
        self.agent = needle.Needle(tools=self.tools, weights=weights, stateless=True)

    def predict(self, query: str) -> dict:
        result = self.agent.complete(query)
        return {
            "calls": [call["name"] for call in result.get("function_calls") or []],
            "confidence": result.get("confidence"),
            "suppressed": [call["name"] for call in result.get("suppressed_calls") or []],
        }
