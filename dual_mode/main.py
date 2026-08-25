"""FastAPI server for Pipeline and Full Duplex OpenAI voice chat."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from starlette.concurrency import run_in_threadpool

from .core import (
    TurnSettings,
    build_instructions,
    build_realtime_session,
    parse_history,
    transcription_language,
    validate_turn_settings,
)


LOGGER = logging.getLogger("voice-chatgpt")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

PIPELINE_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
PIPELINE_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-transcribe")
PIPELINE_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1")
REALTIME_TRANSCRIBE_MODEL = os.getenv(
    "OPENAI_REALTIME_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"
)
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))

app = FastAPI(title="OpenAI Dual-mode Voice Chat", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
    return key


def _settings(
    system_prompt: str,
    language_a: str,
    language_b: str,
    voice: str,
) -> TurnSettings:
    try:
        return validate_turn_settings(system_prompt, language_a, language_b, voice)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _binary_content(response: Any) -> bytes:
    if hasattr(response, "read"):
        content = response.read()
    else:
        content = response.content
    if not isinstance(content, bytes):
        raise RuntimeError("The speech API returned an unexpected response")
    return content


def _pipeline_turn(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    settings: TurnSettings,
    history: list[dict[str, str]],
    api_key: str,
) -> dict[str, str]:
    client = OpenAI(api_key=api_key)
    audio_file = BytesIO(audio_bytes)
    audio_file.name = filename

    transcript_response = client.audio.transcriptions.create(
        model=PIPELINE_TRANSCRIBE_MODEL,
        file=(filename, audio_file, content_type),
        language=transcription_language(settings.language_a),
    )
    transcript = transcript_response.text.strip()
    if not transcript:
        raise RuntimeError("No speech was detected in the recording")

    model_response = client.responses.create(
        model=PIPELINE_TEXT_MODEL,
        instructions=build_instructions(settings),
        input=[*history, {"role": "user", "content": transcript}],
    )
    reply = model_response.output_text.strip()
    if not reply:
        raise RuntimeError("The language model returned an empty response")

    speech_response = client.audio.speech.create(
        model=PIPELINE_TTS_MODEL,
        voice=settings.voice,
        input=reply,
        instructions=(
            f"Speak naturally and clearly in {settings.language_b}. "
            "Use a warm conversational pace."
        ),
        response_format="mp3",
    )
    speech_bytes = _binary_content(speech_response)

    return {
        "transcript": transcript,
        "reply": reply,
        "audio_base64": base64.b64encode(speech_bytes).decode("ascii"),
        "audio_mime": "audio/mpeg",
    }


def _safe_upstream_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message")
        if isinstance(message, str) and message:
            return message[:500]
    except (ValueError, AttributeError):
        pass
    return f"OpenAI Realtime returned HTTP {response.status_code}"


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "modes": ["pipeline", "realtime"],
        "models": {
            "text": PIPELINE_TEXT_MODEL,
            "transcribe": PIPELINE_TRANSCRIBE_MODEL,
            "tts": PIPELINE_TTS_MODEL,
            "realtime": REALTIME_MODEL,
        },
    }


@app.post("/api/pipeline/turn", response_class=JSONResponse)
async def pipeline_turn(
    audio: UploadFile = File(...),
    system_prompt: str = Form(""),
    language_a: str = Form("zh-TW"),
    language_b: str = Form("en"),
    voice: str = Form("marin"),
    history_json: str = Form("[]"),
) -> dict[str, str]:
    settings = _settings(system_prompt, language_a, language_b, voice)
    try:
        history = parse_history(history_json)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    audio_bytes = await audio.read(MAX_AUDIO_BYTES + 1)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The recording is empty")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="The recording is too large")

    filename = Path(audio.filename or "recording.webm").name
    content_type = audio.content_type or "audio/webm"
    try:
        return await run_in_threadpool(
            _pipeline_turn,
            audio_bytes,
            filename,
            content_type,
            settings,
            history,
            _api_key(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Pipeline turn failed")
        raise HTTPException(status_code=502, detail="OpenAI pipeline request failed") from exc


@app.post("/api/realtime/session", response_class=PlainTextResponse)
async def realtime_session(
    sdp: str = Form(...),
    system_prompt: str = Form(""),
    language_a: str = Form("zh-TW"),
    language_b: str = Form("en"),
    voice: str = Form("marin"),
) -> PlainTextResponse:
    if not sdp.strip() or len(sdp) > 200_000:
        raise HTTPException(status_code=400, detail="Invalid SDP offer")

    settings = _settings(system_prompt, language_a, language_b, voice)
    session = build_realtime_session(
        settings,
        realtime_model=REALTIME_MODEL,
        realtime_transcription_model=REALTIME_TRANSCRIBE_MODEL,
    )
    files = {
        "sdp": (None, sdp, "application/sdp"),
        "session": (None, json.dumps(session), "application/json"),
    }
    headers = {"Authorization": f"Bearer {_api_key()}"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/realtime/calls",
                headers=headers,
                files=files,
            )
    except httpx.HTTPError as exc:
        LOGGER.exception("Realtime SDP exchange failed")
        raise HTTPException(status_code=502, detail="Could not reach OpenAI Realtime") from exc

    if response.is_error:
        detail = _safe_upstream_detail(response)
        LOGGER.error("Realtime session rejected: %s", detail)
        raise HTTPException(status_code=502, detail=detail)

    return PlainTextResponse(response.text, media_type="application/sdp")
