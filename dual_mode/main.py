"""FastAPI server for Pipeline and Full Duplex OpenAI voice chat."""

from __future__ import annotations

import base64
import hashlib
import hmac
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from starlette.concurrency import run_in_threadpool

from .core import (
    RAG_TOOL_NAME,
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
MAX_RAG_FILE_BYTES = int(os.getenv("MAX_RAG_FILE_BYTES", str(20 * 1024 * 1024)))
MAX_RAG_TOTAL_BYTES = int(os.getenv("MAX_RAG_TOTAL_BYTES", str(50 * 1024 * 1024)))
MAX_RAG_FILES_PER_UPLOAD = int(os.getenv("MAX_RAG_FILES_PER_UPLOAD", "10"))
MAX_RAG_FILES_PER_KNOWLEDGE_BASE = int(
    os.getenv("MAX_RAG_FILES_PER_KNOWLEDGE_BASE", "50")
)
RAG_MAX_RESULTS = int(os.getenv("RAG_MAX_RESULTS", "5"))
RAG_EXPIRY_DAYS = int(os.getenv("RAG_EXPIRY_DAYS", "7"))
RAG_INDEX_TIMEOUT_SECONDS = int(os.getenv("RAG_INDEX_TIMEOUT_SECONDS", "120"))

RAG_MIME_TYPES = {
    ".c": "text/x-c",
    ".cpp": "text/x-c++",
    ".cs": "text/x-csharp",
    ".css": "text/css",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".go": "text/x-golang",
    ".html": "text/html",
    ".java": "text/x-java",
    ".js": "text/javascript",
    ".json": "application/json",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".php": "text/x-php",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".py": "text/x-python",
    ".rb": "text/x-ruby",
    ".sh": "application/x-sh",
    ".tex": "text/x-tex",
    ".ts": "application/typescript",
    ".txt": "text/plain",
}
_VECTOR_STORE_ID_PATTERN = re.compile(r"^vs_[A-Za-z0-9_-]{6,200}$")

app = FastAPI(title="OpenAI Dual-mode Voice Chat", version="1.1.0")
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


def _rag_token_secret(api_key: str) -> bytes:
    """Use a dedicated signing secret when provided, with the API key as a safe fallback."""

    return (os.getenv("RAG_TOKEN_SECRET", "").strip() or api_key).encode("utf-8")


def _encode_rag_token(vector_store_id: str, api_key: str) -> str:
    payload = json.dumps(
        {"version": 1, "vector_store_id": vector_store_id},
        separators=(",", ":"),
    ).encode("utf-8")
    body = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    signature = hmac.new(
        _rag_token_secret(api_key), body.encode("ascii"), hashlib.sha256
    ).digest()
    signature_text = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"{body}.{signature_text}"


def _decode_rag_token(token: str, api_key: str) -> str:
    if not token or len(token) > 2_048 or token.count(".") != 1:
        raise ValueError("Invalid knowledge base token")
    body, signature_text = token.split(".", 1)
    expected = hmac.new(
        _rag_token_secret(api_key), body.encode("ascii"), hashlib.sha256
    ).digest()
    try:
        supplied = base64.urlsafe_b64decode(signature_text + "=" * (-len(signature_text) % 4))
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid knowledge base token") from exc
    if not hmac.compare_digest(expected, supplied):
        raise ValueError("Invalid knowledge base token")
    try:
        decoded = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        payload = json.loads(decoded)
        vector_store_id = payload["vector_store_id"]
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid knowledge base token") from exc
    if payload.get("version") != 1 or not isinstance(vector_store_id, str):
        raise ValueError("Invalid knowledge base token")
    if not _VECTOR_STORE_ID_PATTERN.fullmatch(vector_store_id):
        raise ValueError("Invalid knowledge base token")
    return vector_store_id


def _optional_vector_store_id(rag_token: str, api_key: str) -> str | None:
    if not rag_token.strip():
        return None
    try:
        return _decode_rag_token(rag_token.strip(), api_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _cleanup_uploaded_files(
    client: OpenAI,
    vector_store_id: str,
    attached_file_ids: list[str],
    uploaded_file_ids: list[str],
) -> None:
    for file_id in attached_file_ids:
        try:
            client.vector_stores.files.delete(
                file_id=file_id,
                vector_store_id=vector_store_id,
            )
        except Exception:
            LOGGER.warning("Could not detach failed RAG file %s", file_id)
    for file_id in uploaded_file_ids:
        try:
            client.files.delete(file_id=file_id)
        except Exception:
            LOGGER.warning("Could not delete failed RAG upload %s", file_id)


def _upload_knowledge_files(
    files: list[tuple[str, bytes, str]],
    vector_store_id: str | None,
    api_key: str,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key)
    created_store = vector_store_id is None
    if created_store:
        vector_store = client.vector_stores.create(
            name="voice-chatgpt knowledge base",
            expires_after={"anchor": "last_active_at", "days": RAG_EXPIRY_DAYS},
        )
        vector_store_id = vector_store.id
        current_file_count = 0
    else:
        vector_store = client.vector_stores.retrieve(vector_store_id=vector_store_id)
        current_file_count = int(vector_store.file_counts.total)

    if current_file_count + len(files) > MAX_RAG_FILES_PER_KNOWLEDGE_BASE:
        if created_store:
            client.vector_stores.delete(vector_store_id=vector_store_id)
        raise ValueError(
            f"A knowledge base can contain at most {MAX_RAG_FILES_PER_KNOWLEDGE_BASE} files"
        )

    uploaded_file_ids: list[str] = []
    attached_file_ids: list[str] = []
    try:
        for filename, content, content_type in files:
            stream = BytesIO(content)
            stream.name = filename
            uploaded = client.files.create(
                file=(filename, stream, content_type),
                purpose="assistants",
            )
            uploaded_file_ids.append(uploaded.id)
            attached = client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=uploaded.id,
            )
            attached_file_ids.append(attached.id)

        deadline = time.monotonic() + RAG_INDEX_TIMEOUT_SECONDS
        pending = set(attached_file_ids)
        while pending and time.monotonic() < deadline:
            for file_id in list(pending):
                status = client.vector_stores.files.retrieve(
                    file_id=file_id,
                    vector_store_id=vector_store_id,
                )
                if status.status == "completed":
                    pending.remove(file_id)
                elif status.status in {"failed", "cancelled"}:
                    message = getattr(getattr(status, "last_error", None), "message", "")
                    raise RuntimeError(message or f"OpenAI could not index {file_id}")
            if pending:
                time.sleep(1)
        if pending:
            raise TimeoutError("Knowledge base indexing timed out")
    except Exception:
        _cleanup_uploaded_files(
            client,
            vector_store_id,
            attached_file_ids,
            uploaded_file_ids,
        )
        if created_store:
            try:
                client.vector_stores.delete(vector_store_id=vector_store_id)
            except Exception:
                LOGGER.warning("Could not delete failed RAG vector store %s", vector_store_id)
        raise

    return {
        "vector_store_id": vector_store_id,
        "files": [filename for filename, _, _ in files],
    }


def _search_knowledge_base(
    vector_store_id: str,
    query: str,
    api_key: str,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key)
    response = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query,
        max_num_results=max(1, min(RAG_MAX_RESULTS, 20)),
        rewrite_query=True,
    )
    results: list[dict[str, Any]] = []
    remaining_characters = 12_000
    for item in response.data:
        text_parts = [
            part.text.strip()
            for part in item.content
            if getattr(part, "type", None) == "text"
            and isinstance(getattr(part, "text", None), str)
            and part.text.strip()
        ]
        text = "\n".join(text_parts)
        if not text or remaining_characters <= 0:
            continue
        text = text[: min(3_000, remaining_characters)]
        remaining_characters -= len(text)
        results.append(
            {
                "filename": item.filename,
                "score": round(float(item.score), 4),
                "text": text,
            }
        )
    return {"query": query, "results": results}


def _delete_knowledge_base(vector_store_id: str, api_key: str) -> None:
    client = OpenAI(api_key=api_key)
    page = client.vector_stores.files.list(vector_store_id=vector_store_id, limit=100)
    file_ids = [item.id for item in page.data]
    client.vector_stores.delete(vector_store_id=vector_store_id)
    for file_id in file_ids:
        try:
            client.files.delete(file_id=file_id)
        except Exception:
            LOGGER.warning("Vector store deleted, but OpenAI file cleanup failed for %s", file_id)


def _pipeline_turn(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    settings: TurnSettings,
    history: list[dict[str, str]],
    vector_store_id: str | None,
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

    response_options: dict[str, Any] = {
        "model": PIPELINE_TEXT_MODEL,
        "instructions": build_instructions(
            settings,
            rag_enabled=bool(vector_store_id),
        ),
        "input": [*history, {"role": "user", "content": transcript}],
    }
    if vector_store_id:
        response_options["tools"] = [
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": max(1, min(RAG_MAX_RESULTS, 20)),
            }
        ]
    model_response = client.responses.create(**response_options)
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
        "rag": {
            "enabled": True,
            "max_files_per_upload": MAX_RAG_FILES_PER_UPLOAD,
            "max_files_per_knowledge_base": MAX_RAG_FILES_PER_KNOWLEDGE_BASE,
            "max_file_bytes": MAX_RAG_FILE_BYTES,
            "max_total_bytes": MAX_RAG_TOTAL_BYTES,
            "expires_after_days": RAG_EXPIRY_DAYS,
        },
    }


@app.post("/api/rag/upload", response_class=JSONResponse)
async def rag_upload(
    files: list[UploadFile] = File(...),
    rag_token: str = Form(""),
) -> dict[str, Any]:
    if not files or len(files) > MAX_RAG_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=422,
            detail=f"Upload between 1 and {MAX_RAG_FILES_PER_UPLOAD} files at a time",
        )

    prepared: list[tuple[str, bytes, str]] = []
    total_bytes = 0
    seen_names: set[str] = set()
    for upload in files:
        filename = Path(upload.filename or "").name.strip()
        if not filename or len(filename) > 180:
            raise HTTPException(status_code=422, detail="A knowledge file has an invalid name")
        extension = Path(filename).suffix.lower()
        content_type = RAG_MIME_TYPES.get(extension)
        if not content_type:
            supported = ", ".join(sorted(RAG_MIME_TYPES))
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported knowledge file type. Supported: {supported}",
            )
        if filename.casefold() in seen_names:
            raise HTTPException(status_code=422, detail=f"Duplicate filename: {filename}")
        seen_names.add(filename.casefold())

        content = await upload.read(MAX_RAG_FILE_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail=f"{filename} is empty")
        if len(content) > MAX_RAG_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the per-file upload limit",
            )
        total_bytes += len(content)
        if total_bytes > MAX_RAG_TOTAL_BYTES:
            raise HTTPException(status_code=413, detail="Knowledge files exceed the total limit")
        prepared.append((filename, content, content_type))

    api_key = _api_key()
    vector_store_id = _optional_vector_store_id(rag_token, api_key)
    try:
        result = await run_in_threadpool(
            _upload_knowledge_files,
            prepared,
            vector_store_id,
            api_key,
        )
    except Exception as exc:
        LOGGER.exception("RAG upload or indexing failed")
        raise HTTPException(
            status_code=502,
            detail="OpenAI could not upload or index the knowledge files",
        ) from exc

    return {
        "rag_token": _encode_rag_token(result["vector_store_id"], api_key),
        "files": result["files"],
        "expires_after_days": RAG_EXPIRY_DAYS,
    }


@app.post("/api/rag/search", response_class=JSONResponse)
async def rag_search(
    query: str = Form(...),
    rag_token: str = Form(...),
) -> dict[str, Any]:
    clean_query = query.strip()
    if not clean_query or len(clean_query) > 2_000:
        raise HTTPException(status_code=422, detail="Invalid knowledge base query")
    api_key = _api_key()
    vector_store_id = _optional_vector_store_id(rag_token, api_key)
    if not vector_store_id:
        raise HTTPException(status_code=422, detail="No knowledge base is active")
    try:
        result = await run_in_threadpool(
            _search_knowledge_base,
            vector_store_id,
            clean_query,
            api_key,
        )
    except Exception as exc:
        LOGGER.exception("RAG search failed")
        raise HTTPException(status_code=502, detail="Knowledge base search failed") from exc
    return {"tool": RAG_TOOL_NAME, **result}


@app.post("/api/rag/delete", response_class=JSONResponse)
async def rag_delete(rag_token: str = Form(...)) -> dict[str, bool]:
    api_key = _api_key()
    vector_store_id = _optional_vector_store_id(rag_token, api_key)
    if not vector_store_id:
        raise HTTPException(status_code=422, detail="No knowledge base is active")
    try:
        await run_in_threadpool(_delete_knowledge_base, vector_store_id, api_key)
    except Exception as exc:
        LOGGER.exception("RAG deletion failed")
        raise HTTPException(status_code=502, detail="Knowledge base deletion failed") from exc
    return {"deleted": True}


@app.post("/api/pipeline/turn", response_class=JSONResponse)
async def pipeline_turn(
    audio: UploadFile = File(...),
    system_prompt: str = Form(""),
    language_a: str = Form("zh-TW"),
    language_b: str = Form("en"),
    voice: str = Form("marin"),
    history_json: str = Form("[]"),
    rag_token: str = Form(""),
) -> dict[str, str]:
    settings = _settings(system_prompt, language_a, language_b, voice)
    api_key = _api_key()
    vector_store_id = _optional_vector_store_id(rag_token, api_key)
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
            vector_store_id,
            api_key,
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
    rag_token: str = Form(""),
) -> PlainTextResponse:
    if not sdp.strip() or len(sdp) > 200_000:
        raise HTTPException(status_code=400, detail="Invalid SDP offer")

    settings = _settings(system_prompt, language_a, language_b, voice)
    api_key = _api_key()
    vector_store_id = _optional_vector_store_id(rag_token, api_key)
    session = build_realtime_session(
        settings,
        realtime_model=REALTIME_MODEL,
        realtime_transcription_model=REALTIME_TRANSCRIBE_MODEL,
        rag_enabled=bool(vector_store_id),
    )
    files = {
        "sdp": (None, sdp, "application/sdp"),
        "session": (None, json.dumps(session), "application/json"),
    }
    headers = {"Authorization": f"Bearer {api_key}"}

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
