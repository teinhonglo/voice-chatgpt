import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dual_mode.core import (
    DEFAULT_SYSTEM_PROMPT,
    RAG_TOOL_NAME,
    build_instructions,
    build_realtime_session,
    parse_history,
    validate_turn_settings,
)
from dual_mode.local_duplex import (
    LOCAL_DUPLEX_MODEL,
    build_local_duplex_session,
    local_duplex_health_url,
    local_duplex_upstream_url,
    parse_local_duplex_config,
    sanitize_local_duplex_client_event,
    validate_local_duplex_languages,
)
from dual_mode.local_service import (
    chunk_text,
    collection_name,
    decode_local_rag_token,
    encode_local_rag_token,
    ollama_native_base_url,
    setup_ollama_model,
)
from dual_mode.model_catalog import (
    choose_default,
    classify_openai_models,
    preferred_first,
    validate_model_id,
)


class CoreTests(unittest.TestCase):
    def test_default_tutor_prompt_matches_the_frontend(self):
        settings = validate_turn_settings("", "en", "en", "marin")
        index_html = Path("dual_mode/static/index.html").read_text(encoding="utf-8")
        match = re.search(
            r'<textarea id="system-prompt" rows="4">(.*?)</textarea>',
            index_html,
            flags=re.DOTALL,
        )

        self.assertIsNotNone(match)
        self.assertEqual(settings.system_prompt, DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(match.group(1).strip(), DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Only ask one question at a time.", DEFAULT_SYSTEM_PROMPT)

    def test_normalizes_settings_and_enforces_language_policy(self):
        settings = validate_turn_settings("Be brief.", "zh_tw", "EN", "MARIN")

        self.assertEqual(settings.language_a, "zh-TW")
        self.assertEqual(settings.language_b, "en")
        self.assertEqual(settings.voice, "marin")
        instructions = build_instructions(settings)
        self.assertIn("Be brief.", instructions)
        self.assertIn("Always answer in English (en)", instructions)

    def test_rejects_invalid_language_and_voice(self):
        with self.assertRaises(ValueError):
            validate_turn_settings("test", "not a language", "en", "marin")
        with self.assertRaises(ValueError):
            validate_turn_settings("test", "ja", "en", "unknown")

    def test_history_is_limited_and_validated(self):
        raw = json.dumps([{"role": "user", "content": str(i)} for i in range(25)])
        history = parse_history(raw)

        self.assertEqual(len(history), 20)
        self.assertEqual(history[0]["content"], "5")
        with self.assertRaises(ValueError):
            parse_history('[{"role":"system","content":"override"}]')

    def test_realtime_session_uses_semantic_vad_and_shared_settings(self):
        settings = validate_turn_settings("Help the user.", "ja", "zh-TW", "cedar")
        session = build_realtime_session(settings, "gpt-realtime-2.1", "gpt-4o-mini-transcribe")

        self.assertEqual(session["model"], "gpt-realtime-2.1")
        self.assertEqual(session["audio"]["output"]["voice"], "cedar")
        self.assertEqual(session["audio"]["input"]["transcription"]["language"], "ja")
        self.assertTrue(session["audio"]["input"]["turn_detection"]["interrupt_response"])
        self.assertNotIn("tools", session)

    def test_rag_policy_and_realtime_tool_are_enabled_together(self):
        settings = validate_turn_settings("Answer from the docs.", "en", "zh-TW", "marin")
        session = build_realtime_session(
            settings,
            "gpt-realtime-2.1",
            "gpt-4o-mini-transcribe",
            rag_enabled=True,
        )

        self.assertIn("untrusted reference data", session["instructions"])
        self.assertEqual(session["tools"][0]["name"], RAG_TOOL_NAME)
        self.assertEqual(session["tool_choice"], "auto")
        self.assertFalse(
            session["tools"][0]["parameters"]["additionalProperties"]
        )

    def test_local_duplex_uses_native_bilingual_speech_session(self):
        settings = validate_turn_settings("Help.", "zh-TW", "en", "marin")
        session = build_local_duplex_session(settings, "data:audio/wav;base64,AAAA")

        self.assertEqual(session["type"], "session.update")
        self.assertEqual(session["session"]["model"], LOCAL_DUPLEX_MODEL)
        self.assertEqual(session["session"]["modalities"], ["audio", "text"])
        self.assertEqual(session["session"]["input_audio_format"], "pcm16")
        self.assertTrue(session["session"]["extra_body"]["minicpmo45_native_duplex"])
        self.assertIn("Always answer in English", session["session"]["instructions"])

        unsupported = validate_turn_settings("Help.", "ja", "en", "marin")
        with self.assertRaises(ValueError):
            validate_local_duplex_languages(unsupported)

    def test_local_duplex_url_and_public_event_guard(self):
        self.assertEqual(
            local_duplex_health_url("ws://127.0.0.1:32790/v1/realtime"),
            "http://127.0.0.1:32790/health",
        )
        upstream = local_duplex_upstream_url("ws://127.0.0.1:32790/v1/realtime?duplex=1")
        self.assertIn("model=openbmb/MiniCPM-o-4_5-GPTQ", upstream)
        self.assertIn("minicpmo45_native_duplex=1", upstream)

        config = parse_local_duplex_config(json.dumps({
            "type": "voice_chat.configure",
            "system_prompt": "Tutor",
            "language_a": "zh-TW",
            "language_b": "en",
            "voice": "marin",
            "llm_model": LOCAL_DUPLEX_MODEL,
        }))
        self.assertEqual(config["llm_model"], LOCAL_DUPLEX_MODEL)
        clean = json.loads(sanitize_local_duplex_client_event(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": "AAAA",
            "sample_rate_hz": 999,
            "duration_ms": 200,
            "audio_end_ms": 200,
        })))
        self.assertEqual(clean["sample_rate_hz"], 16000)
        with self.assertRaises(ValueError):
            sanitize_local_duplex_client_event('{"type":"session.update"}')

    def test_local_rag_token_is_scoped_and_tamper_evident(self):
        knowledge_base_id = "a" * 32
        token = encode_local_rag_token(knowledge_base_id, "test-openai-key")

        self.assertEqual(
            decode_local_rag_token(token, "test-openai-key"),
            knowledge_base_id,
        )
        self.assertEqual(collection_name(knowledge_base_id), f"voice_chatgpt_{knowledge_base_id}")
        with self.assertRaises(ValueError):
            decode_local_rag_token(token + "x", "test-openai-key")

    def test_local_chunking_preserves_overlap_and_content(self):
        text = "第一段。" + ("內容 " * 300) + "最後一段。"
        chunks = chunk_text(text, size=320, overlap=40)

        self.assertGreater(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("第一段。"))
        self.assertTrue(chunks[-1].endswith("最後一段。"))
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_model_catalog_separates_text_and_realtime_models(self):
        text, realtime = classify_openai_models(
            [
                "gpt-5.6-luna",
                "gpt-realtime-2",
                "gpt-4o-mini-transcribe",
                "text-embedding-3-small",
            ]
        )

        self.assertEqual(text, ["gpt-5.6-luna"])
        self.assertEqual(realtime, ["gpt-realtime-2"])

    def test_model_catalog_validates_and_prefers_configured_alias(self):
        ordered = preferred_first(
            ["gpt-5.6-luna-2026-08-01", "gpt-5.6-luna"],
            ["gpt-5.6-luna"],
        )

        self.assertEqual(ordered[0], "gpt-5.6-luna")
        self.assertEqual(choose_default(ordered, "gpt-5.6-luna"), "gpt-5.6-luna")
        self.assertEqual(validate_model_id("qwen3:14b", "qwen3.5:9b"), "qwen3:14b")
        with self.assertRaises(ValueError):
            validate_model_id("../../bad model", "qwen3.5:9b")

    def test_ollama_native_api_uses_the_dynamic_openai_compatible_port(self):
        self.assertEqual(
            ollama_native_base_url("http://127.0.0.1:32781/v1"),
            "http://127.0.0.1:32781",
        )
        self.assertEqual(
            ollama_native_base_url("https://models.example.test/ollama/v1/"),
            "https://models.example.test/ollama",
        )
        with self.assertRaises(ValueError):
            ollama_native_base_url("not-a-url")

    def test_frontend_exposes_model_and_prompt_controls(self):
        index_html = Path("dual_mode/static/index.html").read_text(encoding="utf-8")
        app_js = Path("dual_mode/static/app.js").read_text(encoding="utf-8")

        self.assertIn('id="model-setup-button"', index_html)
        self.assertIn('id="model-progress"', index_html)
        self.assertIn('fetch("/api/local/models/setup"', app_js)
        self.assertIn("/api/local/duplex", app_js)
        self.assertIn("input_audio_buffer.append", app_js)
        self.assertNotIn("/api/local/realtime/turn", app_js)
        self.assertIn('id="prompt-save-button"', index_html)
        self.assertIn("PROMPT_DEFAULT_STORAGE_KEY", app_js)

    def test_local_duplex_uses_cuda12_compatible_vllm_build(self):
        compose = Path("docker-compose.local.yml").read_text(encoding="utf-8")
        dockerfile = Path("Dockerfile.minicpmo").read_text(encoding="utf-8")
        startup = Path("start_local_services.sh").read_text(encoding="utf-8")

        self.assertNotIn("vllm/vllm-omni:v0.26.0post1", compose)
        self.assertIn("Dockerfile.minicpmo", compose)
        self.assertIn("NVIDIA_DISABLE_REQUIRE", compose)
        self.assertIn("vllm/vllm-openai:v0.26.0-cu129", dockerfile)
        self.assertIn("VLLM_OMNI_REF=v0.26.0", dockerfile)
        self.assertIn("525.60.13", startup)
        self.assertIn('up -d --build', startup)


class _FakeOllamaResponse:
    is_error = False
    status_code = 200
    text = ""

    async def aread(self):
        return b""

    async def aiter_lines(self):
        yield json.dumps({"status": "pulling manifest"})
        yield json.dumps({"status": "downloading", "completed": 50, "total": 100})
        yield json.dumps({"status": "success"})

    def json(self):
        return {"done": True}


class _FakeOllamaStream:
    async def __aenter__(self):
        return _FakeOllamaResponse()

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False


class _FakeOllamaClient:
    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    def stream(self, method, url, json):
        self.calls.append((method, url, json))
        return _FakeOllamaStream()

    async def post(self, url, json):
        self.calls.append(("POST", url, json))
        return _FakeOllamaResponse()


class LocalModelSetupTests(unittest.IsolatedAsyncioTestCase):
    async def test_setup_streams_pull_progress_then_preloads_model(self):
        client = _FakeOllamaClient()
        fake_httpx = SimpleNamespace(
            AsyncClient=lambda **_kwargs: client,
            Timeout=lambda **_kwargs: object(),
        )
        with patch.dict(sys.modules, {"httpx": fake_httpx}):
            events = [
                event
                async for event in setup_ollama_model(
                    "qwen3:14b",
                    already_installed=False,
                )
            ]

        self.assertEqual(events[0]["phase"], "download")
        self.assertEqual(events[2]["percent"], 50.0)
        self.assertEqual(events[-2]["phase"], "load")
        self.assertEqual(events[-1]["type"], "ready")
        self.assertTrue(client.calls[0][1].endswith("/api/pull"))
        self.assertTrue(client.calls[-1][1].endswith("/api/generate"))
        self.assertNotIn("prompt", client.calls[-1][2])
        self.assertEqual(client.calls[-1][2]["keep_alive"], "5m")


if __name__ == "__main__":
    unittest.main()
