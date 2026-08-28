"""Model catalog filtering and validation for the browser model selector."""

from __future__ import annotations

import re
from typing import Any, Iterable


MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")

LOCAL_RECOMMENDED_MODELS: tuple[dict[str, str], ...] = (
    {
        "id": "qwen3.5:9b",
        "label": "Qwen 3.5 9B",
        "size": "6.6 GB",
        "note": "預設，201 種語言／方言",
    },
    {
        "id": "qwen3:14b",
        "label": "Qwen 3 14B",
        "size": "9.3 GB",
        "note": "成熟的多語與指令遵循",
    },
    {
        "id": "gemma3:12b",
        "label": "Gemma 3 12B",
        "size": "8.1 GB",
        "note": "140+ 語言的通用模型",
    },
)

_TEXT_PREFIXES = ("gpt-", "o1", "o3", "o4")
_TEXT_EXCLUSIONS = (
    "audio",
    "codex",
    "deep-research",
    "embedding",
    "image",
    "moderation",
    "realtime",
    "search",
    "sora",
    "transcribe",
    "tts",
    "whisper",
)


def validate_model_id(value: str, default: str) -> str:
    """Return a safe model identifier suitable for upstream API requests."""

    model_id = (value or default).strip()
    if not MODEL_ID_PATTERN.fullmatch(model_id):
        raise ValueError("Invalid model id")
    return model_id


def classify_openai_models(models: Iterable[Any]) -> tuple[list[str], list[str]]:
    """Split the basic Models API response into text and Realtime choices.

    The Models API exposes identifiers and ownership but not endpoint capability,
    so classification is intentionally conservative.
    """

    ids: set[str] = set()
    for item in models:
        model_id = item if isinstance(item, str) else getattr(item, "id", "")
        if isinstance(model_id, str) and MODEL_ID_PATTERN.fullmatch(model_id):
            ids.add(model_id)

    realtime = sorted(model_id for model_id in ids if "realtime" in model_id.lower())
    text = sorted(
        model_id
        for model_id in ids
        if model_id.lower().startswith(_TEXT_PREFIXES)
        and not any(part in model_id.lower() for part in _TEXT_EXCLUSIONS)
    )
    return text, realtime


def preferred_first(model_ids: Iterable[str], preferred: Iterable[str]) -> list[str]:
    """Keep every model while presenting useful aliases before snapshots."""

    unique = sorted(set(model_ids))
    ranks = {model_id: index for index, model_id in enumerate(preferred)}
    return sorted(
        unique,
        key=lambda model_id: (
            ranks.get(model_id, len(ranks)),
            bool(re.search(r"-\d{4}-\d{2}-\d{2}$", model_id)),
            model_id,
        ),
    )


def choose_default(available: list[str], configured: str) -> str:
    if configured in available:
        return configured
    return available[0] if available else configured
