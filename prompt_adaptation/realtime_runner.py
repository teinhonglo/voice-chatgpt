from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import quote

import websocket

from prompt_adaptation.dataset import TutorTrajectory, audio_path


@dataclass(frozen=True)
class RealtimeTurn:
    index: int
    student_text: str | None
    assistant_text: str


@dataclass(frozen=True)
class RealtimeTrajectoryResult:
    trajectory_id: str
    turns: tuple[RealtimeTurn, ...]  # index 0 is the lesson opening


def build_realtime_instructions(candidate_prompt: str, trajectory: TutorTrajectory) -> str:
    """Keep dynamic book context outside the text optimized by GEPA."""
    return f"""{candidate_prompt.strip()}

# CURRENT BOOK CONTEXT — AUTHORITATIVE
{trajectory.book_context}

# RUNTIME INTERFACE
- The application will first send the control token [START_LESSON]. It is not a learner utterance. Begin the assigned reading lesson immediately when it arrives.
- After that, every user item is the learner's spoken turn for this same lesson.
- Keep the lesson state internally. Do not output JSON, state labels, scores, or hidden reasoning.
- Produce only the natural child-facing tutor utterance that should be spoken aloud.
- The final-report/assessment stage after lesson completion belongs to another model and is outside your responsibility.
""".strip()


class RealtimeEpisodeClient:
    def __init__(
        self,
        *,
        model: str,
        voice: str,
        reasoning_effort: str,
        max_output_tokens: int,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.voice = voice
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = int(max_output_tokens)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY must be configured")
        self.timeout = timeout
        self.ws: websocket.WebSocket | None = None

    def __enter__(self) -> "RealtimeEpisodeClient":
        url = f"wss://api.openai.com/v1/realtime?model={quote(self.model)}"
        self.ws = websocket.create_connection(
            url,
            header=[f"Authorization: Bearer {self.api_key}"],
            timeout=self.timeout,
        )
        self._wait_for_type("session.created")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def _send(self, event: dict[str, Any]) -> None:
        if self.ws is None:
            raise RuntimeError("Realtime socket is not connected")
        self.ws.send(json.dumps(event, ensure_ascii=False))

    def _recv(self) -> dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("Realtime socket is not connected")
        raw = self.ws.recv()
        if not isinstance(raw, str):
            raw = raw.decode("utf-8")
        event = json.loads(raw)
        if event.get("type") == "error":
            error = event.get("error", {})
            raise RuntimeError(
                f"Realtime error: {error.get('code', '')} {error.get('message', event)}"
            )
        return event

    def _wait_for_type(self, event_type: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            event = self._recv()
            if event.get("type") == event_type:
                return event
        raise TimeoutError(f"Timed out waiting for {event_type}")

    def configure(self, instructions: str) -> None:
        self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.model,
                    "instructions": instructions,
                    "output_modalities": ["audio"],
                    "reasoning": {"effort": self.reasoning_effort},
                    "max_output_tokens": self.max_output_tokens,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "turn_detection": None,
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "voice": self.voice,
                        },
                    },
                },
            }
        )
        self._wait_for_type("session.updated")

    def add_text_user_item(self, text: str) -> None:
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    def add_audio_user_item(self, pcm_path: str | Path) -> None:
        audio = base64.b64encode(Path(pcm_path).read_bytes()).decode("ascii")
        self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": audio}],
                },
            }
        )

    def create_response(self) -> str:
        self._send(
            {
                "type": "response.create",
                "response": {"output_modalities": ["audio"]},
            }
        )
        transcript_parts: list[str] = []
        final_transcript: str | None = None
        deadline = time.monotonic() + self.timeout

        while time.monotonic() < deadline:
            event = self._recv()
            event_type = event.get("type")
            if event_type == "response.output_audio_transcript.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    transcript_parts.append(delta)
            elif event_type == "response.output_audio_transcript.done":
                transcript = event.get("transcript")
                if isinstance(transcript, str) and transcript.strip():
                    final_transcript = transcript.strip()
            elif event_type == "response.done":
                response = event.get("response", {})
                status = response.get("status")
                if status not in {None, "completed"}:
                    raise RuntimeError(f"Realtime response ended with status={status}")
                text = final_transcript or "".join(transcript_parts).strip()
                if not text:
                    text = _extract_transcript_from_done(response)
                if not text:
                    raise RuntimeError("Realtime response completed without a transcript")
                return text
        raise TimeoutError("Timed out waiting for response.done")


def _extract_transcript_from_done(response: dict[str, Any]) -> str:
    pieces: list[str] = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            for key in ("transcript", "text"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    pieces.append(value.strip())
    return "\n".join(pieces).strip()


def run_realtime_trajectory(
    *,
    candidate_prompt: str,
    trajectory: TutorTrajectory,
    audio_dir: str | Path,
    model: str,
    voice: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> RealtimeTrajectoryResult:
    results: list[RealtimeTurn] = []
    instructions = build_realtime_instructions(candidate_prompt, trajectory)

    with RealtimeEpisodeClient(
        model=model,
        voice=voice,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    ) as client:
        client.configure(instructions)

        client.add_text_user_item("[START_LESSON]")
        opening = client.create_response()
        results.append(RealtimeTurn(index=0, student_text=None, assistant_text=opening))

        for turn in trajectory.turns:
            pcm_path = audio_path(audio_dir, trajectory.trajectory_id, turn.turn_index)
            if not pcm_path.exists():
                raise FileNotFoundError(
                    f"Missing fixed student audio: {pcm_path}. "
                    "Run python -m prompt_adaptation.prepare_audio first."
                )
            client.add_audio_user_item(pcm_path)
            assistant = client.create_response()
            results.append(
                RealtimeTurn(
                    index=turn.turn_index,
                    student_text=turn.student_text,
                    assistant_text=assistant,
                )
            )

    return RealtimeTrajectoryResult(
        trajectory_id=trajectory.trajectory_id,
        turns=tuple(results),
    )
