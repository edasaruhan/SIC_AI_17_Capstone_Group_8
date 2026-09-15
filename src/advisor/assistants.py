"""The assistants the advisor can ask.

Gemini 3.5 Flash Lite is the primary assistant: the controlled test measured its effect
sizes, so the diagnosis and the advice are always read from its answers. gpt-oss-120b on
Cerebras is the optional second assistant, the other half of the description experiment:
asking it the same questions shows whether a finding is one assistant's habit.
"""

from __future__ import annotations

PRIMARY = "gemini"
ASSISTANTS: dict[str, dict] = {
    "gemini": {"label": "Gemini 3.5 Flash Lite", "service": "gemini"},
    "cerebras": {
        "label": "gpt-oss-120b (Cerebras)",
        "service": "cerebras",
        "model": "gpt-oss-120b",
        # A reasoning model: its hidden reasoning counts towards max_tokens.
        "max_tokens": 6144,
        "extra": {"reasoning_effort": "low"},
    },
}


def label(name: str) -> str:
    return ASSISTANTS.get(name, {}).get("label", name)


def service(name: str) -> str:
    return ASSISTANTS[name]["service"]
