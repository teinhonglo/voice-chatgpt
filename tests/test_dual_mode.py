import json
import unittest

from dual_mode.core import (
    RAG_TOOL_NAME,
    build_instructions,
    build_realtime_session,
    parse_history,
    validate_turn_settings,
)


class CoreTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
