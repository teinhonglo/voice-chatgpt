from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TutorTurn:
    turn_index: int
    student_text: str
    expected_teacher_state: str
    expected_dimension: int | None
    expected_mastery: str | None
    expected_question_type: str | None
    expected_completed: bool
    expected_ib_dacc: str | None
    source_tutor_response: str


@dataclass(frozen=True)
class TutorOpening:
    expected_teacher_state: str
    expected_dimension: int | None
    expected_mastery: str | None
    source_tutor_response: str


@dataclass(frozen=True)
class TutorTrajectory:
    trajectory_id: str
    split: str
    source_session_id: str
    book: str
    dacc: int
    book_context: str
    opening: TutorOpening
    turns: tuple[TutorTurn, ...]


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_trajectories(path: str | Path, split: str | None = None) -> list[TutorTrajectory]:
    items: list[TutorTrajectory] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            item_split = str(payload["split"])
            if split is not None and item_split != split:
                continue

            opening_raw = payload["opening"]
            opening = TutorOpening(
                expected_teacher_state=str(opening_raw.get("expected_teacher_state", "")),
                expected_dimension=_optional_int(opening_raw.get("expected_dimension")),
                expected_mastery=_optional_str(opening_raw.get("expected_mastery")),
                source_tutor_response=str(opening_raw["source_tutor_response"]).strip(),
            )

            turns = tuple(
                TutorTurn(
                    turn_index=int(turn["turn_index"]),
                    student_text=str(turn["student_text"]).strip(),
                    expected_teacher_state=str(turn.get("expected_teacher_state", "")),
                    expected_dimension=_optional_int(turn.get("expected_dimension")),
                    expected_mastery=_optional_str(turn.get("expected_mastery")),
                    expected_question_type=_optional_str(turn.get("expected_question_type")),
                    expected_completed=bool(turn.get("expected_completed", False)),
                    expected_ib_dacc=_optional_str(turn.get("expected_ib_dacc")),
                    source_tutor_response=str(turn["source_tutor_response"]).strip(),
                )
                for turn in payload.get("turns", [])
            )
            if not turns:
                raise ValueError(f"Trajectory on line {line_number} has no turns")
            items.append(
                TutorTrajectory(
                    trajectory_id=str(payload["trajectory_id"]),
                    split=item_split,
                    source_session_id=str(payload.get("source_session_id", "")),
                    book=str(payload["book"]),
                    dacc=int(payload["dacc"]),
                    book_context=str(payload["book_context"]).strip(),
                    opening=opening,
                    turns=turns,
                )
            )

    if not items:
        label = f" split={split}" if split else ""
        raise ValueError(f"No trajectories found in {path}{label}")
    return items


def audio_path(audio_dir: str | Path, trajectory_id: str, turn_index: int) -> Path:
    return Path(audio_dir) / trajectory_id / f"turn_{turn_index:02d}.pcm"
