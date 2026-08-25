"""Pure configuration and validation helpers for the voice application."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Answer accurately, naturally, and concisely."
)

SUPPORTED_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "marin",
    "sage",
    "shimmer",
    "verse",
}

LANGUAGE_LABELS = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
    "zh-CN": "Mandarin Chinese (Simplified)",
    "zh-HK": "Cantonese Chinese (Hong Kong)",
    "zh-TW": "Mandarin Chinese (Traditional)",
}

_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


@dataclass(frozen=True)
class TurnSettings:
    system_prompt: str
    language_a: str
    language_b: str
    voice: str


def normalize_language_code(value: str, field_name: str) -> str:
    """Validate and normalize an ISO/BCP-47-like language code."""

    normalized = value.strip().replace("_", "-")
    if not normalized or not _LANGUAGE_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a language code such as en or zh-TW")

    parts = normalized.split("-")
    parts[0] = parts[0].lower()
    for index in range(1, len(parts)):
        part = parts[index]
        parts[index] = part.upper() if len(part) == 2 and part.isalpha() else part
    return "-".join(parts)


def validate_turn_settings(
    system_prompt: str,
    language_a: str,
    language_b: str,
    voice: str,
) -> TurnSettings:
    prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    if len(prompt) > 12_000:
        raise ValueError("System prompt must be 12,000 characters or fewer")

    normalized_voice = voice.strip().lower()
    if normalized_voice not in SUPPORTED_VOICES:
        raise ValueError("Unsupported voice")

    return TurnSettings(
        system_prompt=prompt,
        language_a=normalize_language_code(language_a, "Language A"),
        language_b=normalize_language_code(language_b, "Language B"),
        voice=normalized_voice,
    )


def language_label(code: str) -> str:
    return LANGUAGE_LABELS.get(code, code)


def build_instructions(settings: TurnSettings) -> str:
    """Combine the configurable prompt with non-negotiable language behavior."""

    input_language = language_label(settings.language_a)
    output_language = language_label(settings.language_b)
    return f"""# Assistant instructions
{settings.system_prompt}

# Application language policy
- The user speaks {input_language} ({settings.language_a}).
- Always answer in {output_language} ({settings.language_b}), even when the user uses another language.
- Respond to the user's meaning; do not merely repeat or translate their words unless they ask for translation.
- If the audio is unclear, ask a short clarification question in {output_language}.
- Make the response easy to understand when spoken aloud and avoid unnecessary formatting.
""".strip()


def parse_history(raw_history: str) -> list[dict[str, str]]:
    """Validate browser-supplied chat history before sending it to Responses."""

    if not raw_history.strip():
        return []
    try:
        parsed: Any = json.loads(raw_history)
    except json.JSONDecodeError as exc:
        raise ValueError("History must be valid JSON") from exc

    if not isinstance(parsed, list):
        raise ValueError("History must be a JSON list")

    clean: list[dict[str, str]] = []
    for item in parsed[-20:]:
        if not isinstance(item, dict):
            raise ValueError("Each history item must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError("History items require a user/assistant role and text content")
        content = content.strip()
        if not content:
            continue
        if len(content) > 8_000:
            raise ValueError("A history message is too long")
        clean.append({"role": role, "content": content})
    return clean


def transcription_language(code: str) -> str:
    """The transcription endpoint accepts the ISO-639 primary language code."""

    return code.split("-", 1)[0]


def build_realtime_session(
    settings: TurnSettings,
    realtime_model: str,
    realtime_transcription_model: str,
) -> dict[str, Any]:
    """Create the GA Realtime session object sent with the browser SDP."""

    return {
        "type": "realtime",
        "model": realtime_model,
        "instructions": build_instructions(settings),
        "output_modalities": ["audio"],
        "reasoning": {"effort": "low"},
        "audio": {
            "input": {
                "transcription": {
                    "model": realtime_transcription_model,
                    "language": transcription_language(settings.language_a),
                },
                "noise_reduction": {"type": "near_field"},
                "turn_detection": {
                    "type": "semantic_vad",
                    "create_response": True,
                    "interrupt_response": True,
                    "eagerness": "auto",
                },
            },
            "output": {"voice": settings.voice},
        },
    }

