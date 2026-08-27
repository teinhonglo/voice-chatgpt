import json
from pathlib import Path
import re
import unittest

from dual_mode.core import (
    DEFAULT_SYSTEM_PROMPT,
    RAG_TOOL_NAME,
    build_instructions,
    build_realtime_session,
    build_realtime_transcription_session,
    parse_history,
    validate_turn_settings,
)
from dual_mode.local_service import (
    chunk_text,
    collection_name,
    decode_local_rag_token,
    encode_local_rag_token,
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

    def test_local_duplex_uses_transcription_only_realtime_session(self):
        settings = validate_turn_settings("Help.", "zh-TW", "en", "marin")
        session = build_realtime_transcription_session(settings, "gpt-live-transcribe")

        self.assertEqual(session["type"], "transcription")
        self.assertEqual(
            session["audio"]["input"]["transcription"]["model"],
            "gpt-live-transcribe",
        )
        self.assertEqual(session["audio"]["input"]["transcription"]["language"], "zh")
        self.assertEqual(session["audio"]["input"]["turn_detection"]["type"], "server_vad")
        self.assertNotIn("output", session["audio"])

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
        self.assertEqual(validate_model_id("qwen3:14b", "qwen3:8b"), "qwen3:14b")
        with self.assertRaises(ValueError):
            validate_model_id("../../bad model", "qwen3:8b")


if __name__ == "__main__":
    unittest.main()
