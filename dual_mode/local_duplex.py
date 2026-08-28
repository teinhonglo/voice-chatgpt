"""Configuration and protocol guards for the local speech-to-speech runtime."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .core import TurnSettings, build_instructions


LOCAL_DUPLEX_MODEL = os.getenv(
    "LOCAL_DUPLEX_MODEL",
    "openbmb/MiniCPM-o-4_5-GPTQ",
).strip()
LOCAL_DUPLEX_WS_URL = os.getenv(
    "LOCAL_DUPLEX_WS_URL",
    "ws://127.0.0.1:8099/v1/realtime",
).strip()
LOCAL_DUPLEX_REF_AUDIO = os.getenv(
    "LOCAL_DUPLEX_REF_AUDIO",
    str(Path(__file__).resolve().parents[1] / "runtime/ref_audio/ref_minicpm_signature.wav"),
).strip()
LOCAL_DUPLEX_INPUT_SAMPLE_RATE = 16_000
LOCAL_DUPLEX_OUTPUT_SAMPLE_RATE = 24_000
LOCAL_DUPLEX_LANGUAGES = frozenset({"en", "zh", "zh-CN", "zh-TW"})
_MAX_AUDIO_EVENT_CHARS = 1_000_000


def validate_local_duplex_languages(settings: TurnSettings) -> None:
    """MiniCPM-o 4.5 documents realtime speech for Chinese and English."""

    if settings.language_a not in LOCAL_DUPLEX_LANGUAGES:
        raise ValueError("Local Full Duplex Language A supports Chinese or English only")
    if settings.language_b not in LOCAL_DUPLEX_LANGUAGES:
        raise ValueError("Local Full Duplex Language B supports Chinese or English only")


def local_duplex_upstream_url(base_url: str = LOCAL_DUPLEX_WS_URL) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError("LOCAL_DUPLEX_WS_URL must be a ws:// or wss:// URL")
    path = parsed.path.rstrip("/") or "/v1/realtime"
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(
        {
            "duplex": "1",
            "model": LOCAL_DUPLEX_MODEL,
            "minicpmo45_native_duplex": "1",
        }
    )
    return urlunsplit(
        (parsed.scheme, parsed.netloc, path, urlencode(query, safe="/"), "")
    )


def local_duplex_health_url(base_url: str = LOCAL_DUPLEX_WS_URL) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError("LOCAL_DUPLEX_WS_URL must be a ws:// or wss:// URL")
    scheme = "https" if parsed.scheme == "wss" else "http"
    return urlunsplit((scheme, parsed.netloc, "/health", "", ""))


def reference_audio_data_url(path: str = LOCAL_DUPLEX_REF_AUDIO) -> str:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"MiniCPM-o reference audio is missing: {audio_path}"
        )
    content = audio_path.read_bytes()
    if not content or len(content) > 5 * 1024 * 1024:
        raise ValueError("MiniCPM-o reference audio must be between 1 byte and 5 MB")
    return "data:audio/wav;base64," + base64.b64encode(content).decode("ascii")


def build_local_duplex_session(
    settings: TurnSettings,
    ref_audio: str,
) -> dict[str, Any]:
    validate_local_duplex_languages(settings)
    return {
        "type": "session.update",
        "session": {
            "model": LOCAL_DUPLEX_MODEL,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": None,
            "overlap_policy": "listen_only",
            "playback_commit_policy": "ack_only",
            "instructions": build_instructions(settings, rag_enabled=False),
            "ref_audio": ref_audio,
            "extra_body": {
                "auto_response": True,
                "minicpmo45_native_duplex": True,
                "force_listen_count": 0,
            },
        },
    }


def parse_local_duplex_config(raw: str) -> dict[str, str]:
    if len(raw) > 20_000:
        raise ValueError("Local Full Duplex configuration is too large")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Local Full Duplex configuration must be JSON") from exc
    if not isinstance(payload, dict) or payload.get("type") != "voice_chat.configure":
        raise ValueError("The first Local Full Duplex message must configure the session")
    fields = {}
    for name in ("system_prompt", "language_a", "language_b", "voice", "llm_model"):
        value = payload.get(name, "")
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        fields[name] = value
    return fields


def sanitize_local_duplex_client_event(raw: str) -> str:
    """Allow only bounded audio and playback events through the public proxy."""

    if len(raw) > _MAX_AUDIO_EVENT_CHARS + 2_000:
        raise ValueError("Local Full Duplex event is too large")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Local Full Duplex event must be JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Local Full Duplex event must be an object")

    event_type = payload.get("type")
    if event_type == "input_audio_buffer.append":
        audio = payload.get("audio")
        if not isinstance(audio, str) or not audio or len(audio) > _MAX_AUDIO_EVENT_CHARS:
            raise ValueError("Invalid PCM16 audio payload")
        duration_ms = payload.get("duration_ms")
        audio_end_ms = payload.get("audio_end_ms")
        if not isinstance(duration_ms, (int, float)) or not 1 <= duration_ms <= 1_000:
            raise ValueError("Invalid audio duration")
        if not isinstance(audio_end_ms, (int, float)) or audio_end_ms < 0:
            raise ValueError("Invalid audio position")
        clean: dict[str, Any] = {
            "type": event_type,
            "audio": audio,
            "input_audio_format": "pcm16",
            "sample_rate_hz": LOCAL_DUPLEX_INPUT_SAMPLE_RATE,
            "duration_ms": round(float(duration_ms), 3),
            "audio_end_ms": round(float(audio_end_ms), 3),
        }
    elif event_type == "input_audio_buffer.commit":
        clean = {"type": event_type, "final": bool(payload.get("final", False))}
    elif event_type == "playback.ack":
        response_id = payload.get("response_id")
        played_ms = payload.get("played_ms")
        if not isinstance(response_id, str) or not response_id or len(response_id) > 200:
            raise ValueError("Invalid playback response id")
        if not isinstance(played_ms, (int, float)) or played_ms < 0:
            raise ValueError("Invalid playback position")
        item_id = payload.get("item_id")
        if not isinstance(item_id, str) or not item_id or len(item_id) > 220:
            item_id = f"item_{response_id}"
        clean = {
            "type": event_type,
            "response_id": response_id,
            "item_id": item_id,
            "played_ms": round(float(played_ms), 3),
            "committed_ms": round(float(played_ms), 3),
        }
    elif event_type == "session.close":
        clean = {"type": event_type}
    else:
        raise ValueError("Unsupported Local Full Duplex event")
    return json.dumps(clean, separators=(",", ":"))
