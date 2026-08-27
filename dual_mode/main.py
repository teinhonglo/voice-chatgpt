"""FastAPI server for OpenAI and local Pipeline / Full Duplex voice chat."""

from __future__ import annotations

import asyncio
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
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from starlette.concurrency import run_in_threadpool

from .core import (
    RAG_TOOL_NAME,
    TurnSettings,
    build_instructions,
    build_prompt_enhancement_instructions,
    build_realtime_session,
    build_realtime_transcription_session,
    clean_enhanced_prompt,
    parse_history,
    transcription_language,
    validate_turn_settings,
)
from .local_service import (
    LOCAL_EMBEDDING_MODEL,
    LOCAL_LLM_API_KEY,
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MODEL,
    LOCAL_RAG_EXTENSIONS,
    QDRANT_URL,
    decode_local_rag_token,
    delete_local_knowledge_base,
    encode_local_rag_token,
    enhance_local_prompt,
    generate_local_reply,
    local_health,
    setup_ollama_model,
    stream_local_reply,
    upload_local_knowledge_files,
)
from .model_catalog import (
    LOCAL_RECOMMENDED_MODELS,
    choose_default,
    classify_openai_models,
    preferred_first,
    validate_model_id,
)


LOGGER = logging.getLogger("voice-chatgpt")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
_BACKEND_VALUE = os.getenv("BACKEND", "openai").strip().lower() or "openai"
DEPLOYMENT_BACKEND = _BACKEND_VALUE if _BACKEND_VALUE in {"openai", "local"} else "openai"

PIPELINE_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.6-luna")
PIPELINE_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-transcribe")
PIPELINE_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
PROMPT_ENHANCER_MODEL = os.getenv("OPENAI_PROMPT_ENHANCER_MODEL", PIPELINE_TEXT_MODEL)
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
REALTIME_TRANSCRIBE_MODEL = os.getenv(
    "OPENAI_REALTIME_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"
)
REALTIME_ASR_MODEL = os.getenv("OPENAI_REALTIME_ASR_MODEL", "gpt-live-transcribe")
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
ENABLE_LOCAL_MODEL_SETUP = os.getenv(
    "ENABLE_LOCAL_MODEL_SETUP",
    os.getenv("MANAGE_LOCAL_SERVICES", "1"),
).strip().lower() not in {"0", "false", "no"}

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

app = FastAPI(title="Cloud and Local Voice Chat", version="1.5.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_LOCAL_MODEL_SETUP_LOCK = asyncio.Lock()


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


def _model_id(value: str, default: str) -> str:
    try:
        return validate_model_id(value, default)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _require_backend(expected: str) -> None:
    if DEPLOYMENT_BACKEND != expected:
        raise HTTPException(status_code=404, detail=f"This mode is unavailable with BACKEND={DEPLOYMENT_BACKEND}")


def _openai_models(api_key: str) -> list[Any]:
    with OpenAI(api_key=api_key) as client:
        return list(client.models.list().data)


def _enhance_openai_prompt(
    draft: str,
    target_model: str,
    mode: str,
    api_key: str,
) -> str:
    instructions = build_prompt_enhancement_instructions(target_model, mode, "openai")
    with OpenAI(api_key=api_key) as client:
        response = client.responses.create(
            model=PROMPT_ENHANCER_MODEL,
            instructions=instructions,
            input=(
                "Rewrite the following draft system prompt. The draft is provided as "
                f"a JSON string:\n{json.dumps(draft, ensure_ascii=False)}"
            ),
            max_output_tokens=3_000,
        )
    return clean_enhanced_prompt(response.output_text or "")


async def _installed_local_models() -> list[str]:
    headers = {"Authorization": f"Bearer {LOCAL_LLM_API_KEY}"}
    async with httpx.AsyncClient(timeout=5.0, headers=headers) as client:
        response = await client.get(f"{LOCAL_LLM_BASE_URL}/models")
        response.raise_for_status()
    payload = response.json()
    return sorted(
        {
            item["id"]
            for item in payload.get("data", [])
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and item["id"].strip()
        }
    )


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
    try:
        body_bytes = body.encode("ascii")
        expected = hmac.new(
            _rag_token_secret(api_key), body_bytes, hashlib.sha256
        ).digest()
        supplied = base64.urlsafe_b64decode(signature_text + "=" * (-len(signature_text) % 4))
    except (UnicodeError, ValueError, TypeError) as exc:
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


def _optional_local_knowledge_base_id(rag_token: str, api_key: str) -> str | None:
    if not rag_token.strip():
        return None
    try:
        return decode_local_rag_token(rag_token.strip(), api_key)
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
    text_model: str,
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
        "model": text_model,
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


def _local_pipeline_turn(
    audio_bytes: bytes,
    filename: str,
    content_type: str,
    settings: TurnSettings,
    history: list[dict[str, str]],
    knowledge_base_id: str | None,
    api_key: str,
    llm_model: str,
) -> dict[str, str]:
    """Use OpenAI for speech I/O and local services for retrieval and reasoning."""

    openai_client = OpenAI(api_key=api_key)
    audio_file = BytesIO(audio_bytes)
    audio_file.name = filename
    transcript_response = openai_client.audio.transcriptions.create(
        model=PIPELINE_TRANSCRIBE_MODEL,
        file=(filename, audio_file, content_type),
        language=transcription_language(settings.language_a),
    )
    transcript = transcript_response.text.strip()
    if not transcript:
        raise RuntimeError("No speech was detected in the recording")

    reply, _ = generate_local_reply(
        settings,
        history,
        transcript,
        knowledge_base_id,
        llm_model,
    )
    speech_response = openai_client.audio.speech.create(
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


async def _tts_segment(segment: str, settings: TurnSettings, api_key: str) -> bytes:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": PIPELINE_TTS_MODEL,
                "voice": settings.voice,
                "input": segment,
                "instructions": (
                    f"Speak naturally and clearly in {settings.language_b}. "
                    "Keep a consistent warm conversational pace."
                ),
                "response_format": "mp3",
            },
        )
        response.raise_for_status()
        return response.content


def _ndjson_event(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


async def _local_duplex_events(
    settings: TurnSettings,
    history: list[dict[str, str]],
    transcript: str,
    knowledge_base_id: str | None,
    api_key: str,
    llm_model: str,
):
    """Stream local text and ordered OpenAI TTS chunks as newline-delimited JSON."""

    output_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
    reply_parts: list[str] = []

    async def produce_text() -> None:
        pending_speech = ""
        try:
            async for delta in stream_local_reply(
                settings,
                history,
                transcript,
                knowledge_base_id,
                llm_model,
            ):
                reply_parts.append(delta)
                pending_speech += delta
                await output_queue.put({"type": "text_delta", "delta": delta})
                punctuation = bool(re.search(r"[。！？!?；;.]\s*$", pending_speech))
                if (punctuation and len(pending_speech) >= 24) or len(pending_speech) >= 260:
                    await tts_queue.put(pending_speech.strip())
                    pending_speech = ""
            if pending_speech.strip():
                await tts_queue.put(pending_speech.strip())
        finally:
            await tts_queue.put(None)
            await output_queue.put({"type": "text_done"})

    async def produce_audio() -> None:
        try:
            while True:
                segment = await tts_queue.get()
                if segment is None:
                    break
                audio = await _tts_segment(segment, settings, api_key)
                await output_queue.put(
                    {
                        "type": "audio",
                        "audio_base64": base64.b64encode(audio).decode("ascii"),
                        "audio_mime": "audio/mpeg",
                    }
                )
        finally:
            await output_queue.put({"type": "audio_done"})

    text_task = asyncio.create_task(produce_text())
    audio_task = asyncio.create_task(produce_audio())
    completed = set()
    try:
        while len(completed) < 2:
            event = await output_queue.get()
            if event["type"] in {"text_done", "audio_done"}:
                completed.add(event["type"])
                continue
            yield _ndjson_event(event)
        await text_task
        await audio_task
        reply = "".join(reply_parts).strip()
        if not reply:
            raise RuntimeError("The local language model returned an empty response")
        yield _ndjson_event({"type": "done", "reply": reply})
    except asyncio.CancelledError:
        text_task.cancel()
        audio_task.cancel()
        raise
    except Exception as exc:
        LOGGER.exception("Local duplex turn failed")
        yield _ndjson_event({"type": "error", "message": str(exc)[:500]})
    finally:
        for task in (text_task, audio_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(text_task, audio_task, return_exceptions=True)


async def _local_model_setup_events(model: str, already_installed: bool):
    """Serialize model setup and report Ollama progress as NDJSON."""

    async with _LOCAL_MODEL_SETUP_LOCK:
        try:
            async for event in setup_ollama_model(
                model,
                already_installed=already_installed,
            ):
                yield _ndjson_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("Local model setup failed for %s", model)
            yield _ndjson_event({"type": "error", "message": str(exc)[:500]})


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
        "backend": DEPLOYMENT_BACKEND,
        "conda_environment": os.getenv("VOICE_CHATGPT_CONDA_ENV", ""),
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "modes": (
            ["local-pipeline", "local-realtime"]
            if DEPLOYMENT_BACKEND == "local"
            else ["pipeline", "realtime"]
        ),
        "models": {
            "text": PIPELINE_TEXT_MODEL,
            "transcribe": PIPELINE_TRANSCRIBE_MODEL,
            "tts": PIPELINE_TTS_MODEL,
            "prompt_enhancer": PROMPT_ENHANCER_MODEL,
            "realtime": REALTIME_MODEL,
            "realtime_asr": REALTIME_ASR_MODEL,
            "local_llm": LOCAL_LLM_MODEL,
            "local_embeddings": LOCAL_EMBEDDING_MODEL,
        },
        "rag": {
            "enabled": True,
            "max_files_per_upload": MAX_RAG_FILES_PER_UPLOAD,
            "max_files_per_knowledge_base": MAX_RAG_FILES_PER_KNOWLEDGE_BASE,
            "max_file_bytes": MAX_RAG_FILE_BYTES,
            "max_total_bytes": MAX_RAG_TOTAL_BYTES,
            "expires_after_days": RAG_EXPIRY_DAYS,
        },
        "local": {
            "qdrant_url": QDRANT_URL,
            "health_url": "/api/local/health",
        },
    }


@app.get("/api/models")
async def model_catalog() -> dict[str, Any]:
    """Return mode-compatible model choices without exposing the API key."""

    api_key = _api_key()
    warnings: list[str] = []
    text_models: list[str] = []
    realtime_models: list[str] = []
    installed_local_models: list[str] = []

    if DEPLOYMENT_BACKEND == "local":
        try:
            local_result = await _installed_local_models()
        except Exception:
            warnings.append("Local model server is not running; showing recommended RTX 3090 models.")
        else:
            for model_id in local_result:
                try:
                    installed_local_models.append(validate_model_id(model_id, LOCAL_LLM_MODEL))
                except ValueError:
                    continue
    else:
        try:
            openai_result = await run_in_threadpool(_openai_models, api_key)
        except Exception:
            warnings.append("OpenAI model list could not be loaded; using the configured defaults.")
            text_models = [PIPELINE_TEXT_MODEL]
            realtime_models = [REALTIME_MODEL]
        else:
            text_models, realtime_models = classify_openai_models(openai_result)
        if not text_models:
            text_models = [PIPELINE_TEXT_MODEL]
        if not realtime_models:
            realtime_models = [REALTIME_MODEL]

        text_models = preferred_first(
            text_models,
            [PIPELINE_TEXT_MODEL, "gpt-5.6-luna", "gpt-5.4-mini", "gpt-4.1-mini", "gpt-4o-mini"],
        )
        realtime_models = preferred_first(
            realtime_models,
            [REALTIME_MODEL, "gpt-realtime-2", "gpt-realtime", "gpt-realtime-mini"],
        )

    return {
        "backend": DEPLOYMENT_BACKEND,
        "modes": (
            ["local-pipeline", "local-realtime"]
            if DEPLOYMENT_BACKEND == "local"
            else ["pipeline", "realtime"]
        ),
        "defaults": {
            "text": choose_default(text_models, PIPELINE_TEXT_MODEL),
            "realtime": choose_default(realtime_models, REALTIME_MODEL),
            "local": (
                LOCAL_LLM_MODEL
                if LOCAL_LLM_MODEL in installed_local_models or not installed_local_models
                else installed_local_models[0]
            ),
        },
        "openai": {
            "text": text_models,
            "realtime": realtime_models,
        },
        "local": {
            "installed": sorted(set(installed_local_models)),
            "recommended": list(LOCAL_RECOMMENDED_MODELS),
            "setup_enabled": ENABLE_LOCAL_MODEL_SETUP,
        },
        "warnings": warnings,
    }


@app.post("/api/local/models/setup")
async def local_model_setup(model: str = Form(...)) -> StreamingResponse:
    """Download an approved Ollama model and preload it for local inference."""

    _require_backend("local")
    if not ENABLE_LOCAL_MODEL_SETUP:
        raise HTTPException(status_code=404, detail="Local model setup is disabled")
    model_id = _model_id(model, LOCAL_LLM_MODEL)
    try:
        installed = set(await _installed_local_models())
    except Exception as exc:
        LOGGER.exception("Could not query Ollama before model setup")
        raise HTTPException(status_code=502, detail="Ollama is not available") from exc

    approved = {item["id"] for item in LOCAL_RECOMMENDED_MODELS}
    approved.add(LOCAL_LLM_MODEL)
    if model_id not in approved and model_id not in installed:
        raise HTTPException(status_code=403, detail="This model is not approved for browser download")
    if _LOCAL_MODEL_SETUP_LOCK.locked():
        raise HTTPException(status_code=409, detail="Another model is currently being configured")

    return StreamingResponse(
        _local_model_setup_events(model_id, model_id in installed),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/api/prompt/enhance", response_class=JSONResponse)
async def prompt_enhance(
    prompt: str = Form(...),
    llm_model: str = Form(...),
    mode: str = Form(...),
) -> dict[str, str]:
    """Rewrite the current system prompt for the selected model and interaction type."""

    draft = prompt.strip()
    if not draft:
        raise HTTPException(status_code=422, detail="System prompt cannot be empty")
    if len(draft) > 12_000:
        raise HTTPException(status_code=422, detail="System prompt must be 12,000 characters or fewer")

    allowed_modes = (
        {"local-pipeline", "local-realtime"}
        if DEPLOYMENT_BACKEND == "local"
        else {"pipeline", "realtime"}
    )
    if mode not in allowed_modes:
        raise HTTPException(status_code=422, detail="Prompt mode does not match the active backend")
    default_model = (
        LOCAL_LLM_MODEL
        if DEPLOYMENT_BACKEND == "local"
        else REALTIME_MODEL if mode == "realtime" else PIPELINE_TEXT_MODEL
    )
    target_model = _model_id(llm_model, default_model)

    try:
        if DEPLOYMENT_BACKEND == "local":
            enhanced = await run_in_threadpool(
                enhance_local_prompt,
                draft,
                target_model,
                mode,
            )
            enhancer_model = target_model
        else:
            enhanced = await run_in_threadpool(
                _enhance_openai_prompt,
                draft,
                target_model,
                mode,
                _api_key(),
            )
            enhancer_model = PROMPT_ENHANCER_MODEL
    except Exception as exc:
        LOGGER.exception("Prompt enhancement failed for %s", target_model)
        raise HTTPException(
            status_code=502,
            detail=f"Prompt enhancement failed: {str(exc)[:300]}",
        ) from exc

    return {
        "prompt": enhanced,
        "target_model": target_model,
        "enhancer_model": enhancer_model,
    }


@app.get("/api/local/health")
async def local_services_health() -> dict[str, Any]:
    _require_backend("local")
    return await local_health()


@app.post("/api/rag/upload", response_class=JSONResponse)
async def rag_upload(
    files: list[UploadFile] = File(...),
    rag_token: str = Form(""),
) -> dict[str, Any]:
    _require_backend("openai")
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
    _require_backend("openai")
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
    _require_backend("openai")
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


@app.post("/api/local/rag/upload", response_class=JSONResponse)
async def local_rag_upload(
    files: list[UploadFile] = File(...),
    rag_token: str = Form(""),
) -> dict[str, Any]:
    _require_backend("local")
    if not files or len(files) > MAX_RAG_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=422,
            detail=f"Upload between 1 and {MAX_RAG_FILES_PER_UPLOAD} files at a time",
        )
    prepared: list[tuple[str, bytes]] = []
    total_bytes = 0
    seen_names: set[str] = set()
    for upload in files:
        filename = Path(upload.filename or "").name.strip()
        extension = Path(filename).suffix.lower()
        if not filename or len(filename) > 180:
            raise HTTPException(status_code=422, detail="A knowledge file has an invalid name")
        if extension not in LOCAL_RAG_EXTENSIONS:
            supported = ", ".join(sorted(LOCAL_RAG_EXTENSIONS))
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported local knowledge file type. Supported: {supported}",
            )
        if filename.casefold() in seen_names:
            raise HTTPException(status_code=422, detail=f"Duplicate filename: {filename}")
        seen_names.add(filename.casefold())
        content = await upload.read(MAX_RAG_FILE_BYTES + 1)
        if not content:
            raise HTTPException(status_code=400, detail=f"{filename} is empty")
        if len(content) > MAX_RAG_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"{filename} exceeds the per-file upload limit")
        total_bytes += len(content)
        if total_bytes > MAX_RAG_TOTAL_BYTES:
            raise HTTPException(status_code=413, detail="Knowledge files exceed the total limit")
        prepared.append((filename, content))

    api_key = _api_key()
    knowledge_base_id = _optional_local_knowledge_base_id(rag_token, api_key)
    try:
        result = await run_in_threadpool(
            upload_local_knowledge_files,
            prepared,
            knowledge_base_id,
        )
    except Exception as exc:
        LOGGER.exception("Local RAG upload or indexing failed")
        raise HTTPException(
            status_code=502,
            detail=f"Local RAG could not index the files: {str(exc)[:300]}",
        ) from exc
    return {
        "rag_token": encode_local_rag_token(result["knowledge_base_id"], api_key),
        "files": result["files"],
        "chunks": result["chunks"],
    }


@app.post("/api/local/rag/delete", response_class=JSONResponse)
async def local_rag_delete(rag_token: str = Form(...)) -> dict[str, bool]:
    _require_backend("local")
    api_key = _api_key()
    knowledge_base_id = _optional_local_knowledge_base_id(rag_token, api_key)
    if not knowledge_base_id:
        raise HTTPException(status_code=422, detail="No local knowledge base is active")
    try:
        await run_in_threadpool(delete_local_knowledge_base, knowledge_base_id)
    except Exception as exc:
        LOGGER.exception("Local RAG deletion failed")
        raise HTTPException(status_code=502, detail="Local knowledge base deletion failed") from exc
    return {"deleted": True}


@app.post("/api/pipeline/turn", response_class=JSONResponse)
async def pipeline_turn(
    audio: UploadFile = File(...),
    system_prompt: str = Form(""),
    language_a: str = Form("zh-TW"),
    language_b: str = Form("en"),
    voice: str = Form("marin"),
    llm_model: str = Form(PIPELINE_TEXT_MODEL),
    history_json: str = Form("[]"),
    rag_token: str = Form(""),
) -> dict[str, str]:
    _require_backend("openai")
    settings = _settings(system_prompt, language_a, language_b, voice)
    text_model = _model_id(llm_model, PIPELINE_TEXT_MODEL)
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
            text_model,
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Pipeline turn failed")
        raise HTTPException(status_code=502, detail="OpenAI pipeline request failed") from exc


@app.post("/api/local/pipeline/turn", response_class=JSONResponse)
async def local_pipeline_turn(
    audio: UploadFile = File(...),
    system_prompt: str = Form(""),
    language_a: str = Form("zh-TW"),
    language_b: str = Form("en"),
    voice: str = Form("marin"),
    llm_model: str = Form(LOCAL_LLM_MODEL),
    history_json: str = Form("[]"),
    rag_token: str = Form(""),
) -> dict[str, str]:
    _require_backend("local")
    settings = _settings(system_prompt, language_a, language_b, voice)
    local_model = _model_id(llm_model, LOCAL_LLM_MODEL)
    api_key = _api_key()
    knowledge_base_id = _optional_local_knowledge_base_id(rag_token, api_key)
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
            _local_pipeline_turn,
            audio_bytes,
            filename,
            content_type,
            settings,
            history,
            knowledge_base_id,
            api_key,
            local_model,
        )
    except Exception as exc:
        LOGGER.exception("Local pipeline turn failed")
        raise HTTPException(
            status_code=502,
            detail=f"Local pipeline request failed: {str(exc)[:300]}",
        ) from exc


@app.post("/api/realtime/session", response_class=PlainTextResponse)
async def realtime_session(
    sdp: str = Form(...),
    system_prompt: str = Form(""),
    language_a: str = Form("zh-TW"),
    language_b: str = Form("en"),
    voice: str = Form("marin"),
    llm_model: str = Form(REALTIME_MODEL),
    rag_token: str = Form(""),
) -> PlainTextResponse:
    _require_backend("openai")
    if not sdp.strip() or len(sdp) > 200_000:
        raise HTTPException(status_code=400, detail="Invalid SDP offer")

    settings = _settings(system_prompt, language_a, language_b, voice)
    realtime_model = _model_id(llm_model, REALTIME_MODEL)
    if "realtime" not in realtime_model.lower():
        raise HTTPException(status_code=422, detail="OpenAI Full Duplex requires a Realtime model")
    api_key = _api_key()
    vector_store_id = _optional_vector_store_id(rag_token, api_key)
    session = build_realtime_session(
        settings,
        realtime_model=realtime_model,
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


@app.post("/api/local/realtime/session", response_class=PlainTextResponse)
async def local_realtime_transcription_session(
    sdp: str = Form(...),
    system_prompt: str = Form(""),
    language_a: str = Form("zh-TW"),
    language_b: str = Form("en"),
    voice: str = Form("marin"),
    rag_token: str = Form(""),
) -> PlainTextResponse:
    _require_backend("local")
    if not sdp.strip() or len(sdp) > 200_000:
        raise HTTPException(status_code=400, detail="Invalid SDP offer")
    settings = _settings(system_prompt, language_a, language_b, voice)
    api_key = _api_key()
    # Validate the local token before opening a metered ASR session.
    _optional_local_knowledge_base_id(rag_token, api_key)
    session = build_realtime_transcription_session(settings, REALTIME_ASR_MODEL)
    files = {
        "sdp": (None, sdp, "application/sdp"),
        "session": (None, json.dumps(session), "application/json"),
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/realtime/calls",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
            )
    except httpx.HTTPError as exc:
        LOGGER.exception("Realtime transcription SDP exchange failed")
        raise HTTPException(status_code=502, detail="Could not reach OpenAI Realtime ASR") from exc
    if response.is_error:
        detail = _safe_upstream_detail(response)
        LOGGER.error("Realtime transcription session rejected: %s", detail)
        raise HTTPException(status_code=502, detail=detail)
    return PlainTextResponse(response.text, media_type="application/sdp")


@app.post("/api/local/realtime/turn")
async def local_realtime_turn(
    transcript: str = Form(...),
    system_prompt: str = Form(""),
    language_a: str = Form("zh-TW"),
    language_b: str = Form("en"),
    voice: str = Form("marin"),
    llm_model: str = Form(LOCAL_LLM_MODEL),
    history_json: str = Form("[]"),
    rag_token: str = Form(""),
) -> StreamingResponse:
    _require_backend("local")
    clean_transcript = transcript.strip()
    if not clean_transcript or len(clean_transcript) > 8_000:
        raise HTTPException(status_code=422, detail="Invalid transcript")
    settings = _settings(system_prompt, language_a, language_b, voice)
    local_model = _model_id(llm_model, LOCAL_LLM_MODEL)
    api_key = _api_key()
    knowledge_base_id = _optional_local_knowledge_base_id(rag_token, api_key)
    try:
        history = parse_history(history_json)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return StreamingResponse(
        _local_duplex_events(
            settings,
            history,
            clean_transcript,
            knowledge_base_id,
            api_key,
            local_model,
        ),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
