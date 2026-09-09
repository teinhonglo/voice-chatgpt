from __future__ import annotations

from dataclasses import dataclass
import json
from statistics import mean
from typing import Any

from openai import OpenAI

from prompt_adaptation.dataset import TutorTrajectory
from prompt_adaptation.realtime_runner import RealtimeTrajectoryResult


@dataclass(frozen=True)
class DimensionScore:
    score: float
    reason: str


@dataclass(frozen=True)
class TutorTurnScore:
    turn_index: int
    process_adherence: DimensionScore
    pedagogical_quality: DimensionScore
    naturalness_encouragement: DimensionScore
    feedback: str

    def weighted_score(self, weights: dict[str, float]) -> float:
        return (
            float(weights["process_adherence"]) * self.process_adherence.score
            + float(weights["pedagogical_quality"]) * self.pedagogical_quality.score
            + float(weights["naturalness_encouragement"])
            * self.naturalness_encouragement.score
        )


EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "process_adherence": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
        "pedagogical_quality": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
        "naturalness_encouragement": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
        "feedback": {"type": "string"},
    },
    "required": [
        "process_adherence",
        "pedagogical_quality",
        "naturalness_encouragement",
        "feedback",
    ],
    "additionalProperties": False,
}


def _history_text(
    trajectory: TutorTrajectory,
    target: RealtimeTrajectoryResult,
    current_index: int,
) -> str:
    lines: list[str] = []
    if target.turns:
        lines.append(f"Tutor opening: {target.turns[0].assistant_text}")
    for i in range(1, current_index):
        source_turn = trajectory.turns[i - 1]
        target_turn = target.turns[i]
        lines.append(f"Learner: {source_turn.student_text}")
        lines.append(f"Tutor: {target_turn.assistant_text}")
    return "\n".join(lines) if lines else "[No previous learner turns]"


def _source_history_text(trajectory: TutorTrajectory, current_index: int) -> str:
    lines = [f"Tutor opening: {trajectory.opening.source_tutor_response}"]
    for i in range(1, current_index):
        turn = trajectory.turns[i - 1]
        lines.append(f"Learner: {turn.student_text}")
        lines.append(f"Tutor: {turn.source_tutor_response}")
    return "\n".join(lines)


def _make_prompt(
    *,
    trajectory: TutorTrajectory,
    target: RealtimeTrajectoryResult,
    turn_index: int,
) -> str:
    if turn_index == 0:
        student_text = "[START_LESSON application control; no learner answer yet]"
        expected_state = trajectory.opening.expected_teacher_state
        source_response = trajectory.opening.source_tutor_response
        expected_completed = False
        target_response = target.turns[0].assistant_text
        target_history = "[Lesson opening]"
        source_history = "[Lesson opening]"
    else:
        source_turn = trajectory.turns[turn_index - 1]
        student_text = source_turn.student_text
        expected_state = source_turn.expected_teacher_state
        source_response = source_turn.source_tutor_response
        expected_completed = source_turn.expected_completed
        target_response = target.turns[turn_index].assistant_text
        target_history = _history_text(trajectory, target, turn_index)
        source_history = _source_history_text(trajectory, turn_index)

    return f"""You are evaluating one turn of a controlled multi-turn AI reading-tutor experiment.

The SOURCE response comes from a previously validated modular Pipeline trajectory. It is a behavioral reference, NOT text that the TARGET must imitate. Reward semantically equivalent tutoring behavior with different wording.

The experiment evaluates only the interactive reading tutor. A separate final-report model is OUT OF SCOPE.

BOOK CONTEXT (authoritative):
{trajectory.book_context}

DACC LEVEL: {trajectory.dacc}

EXPECTED PIPELINE TEACHER STATE FOR THIS TURN:
{expected_state}

EXPECTED LESSON COMPLETED AFTER THIS TURN: {expected_completed}

SOURCE PIPELINE HISTORY BEFORE THIS TURN:
{source_history}

TARGET REALTIME HISTORY BEFORE THIS TURN:
{target_history}

CURRENT LEARNER TURN:
{student_text}

SOURCE PIPELINE TUTOR RESPONSE (reference behavior):
{source_response}

TARGET REALTIME TUTOR RESPONSE (evaluate this):
{target_response}

Score ONLY the current TARGET response on 0–100:

1. process_adherence
   - Does it stay at the required reading objective/dimension?
   - Does it correctly react to whether the learner has met that objective?
   - If the source state says the learner has NOT/partly met the objective, the tutor should explain, hint, or clarify and ask again about the SAME objective rather than advancing.
   - If the objective is met, it should advance to the appropriate next objective.
   - It must not finish early. If completion is expected, it should close instead of starting a new learning objective.
   - For advanced DACC, respect the IB-DACC progression encoded in the expected state.
   - Internal JSON/state labels should not be spoken to the child.

2. pedagogical_quality
   - Is the correction/explanation/question factually grounded in the assigned book?
   - Is the scaffolding useful and appropriate for this learner level?
   - Does it correct misconceptions gently and handle possible ASR errors without inventing facts?
   - Does the question actually test or remediate the intended objective?

3. naturalness_encouragement
   - Does it sound like natural spoken Traditional Chinese for a child/learner?
   - Is encouragement specific and proportionate rather than robotic or excessive?
   - Is it concise enough for spoken interaction and does it avoid unnecessary multi-question overload?

The most important research criterion is strict pedagogical PROCESS adherence. Do not award a high process score merely because the response is friendly or factually correct if it changes the required flow.

Return structured scores plus concise diagnostic feedback useful for automatic prompt optimization.
"""


def judge_turn(
    client: OpenAI,
    *,
    judge_model: str,
    trajectory: TutorTrajectory,
    target: RealtimeTrajectoryResult,
    turn_index: int,
) -> TutorTurnScore:
    response = client.responses.create(
        model=judge_model,
        instructions=(
            "Act as a strict, evidence-based evaluator of AI tutoring behavior. "
            "Use the supplied expected state and book context. Do not reward lexical imitation."
        ),
        input=_make_prompt(
            trajectory=trajectory,
            target=target,
            turn_index=turn_index,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "TutorTurnEvaluation",
                "strict": True,
                "schema": EVAL_SCHEMA,
            }
        },
    )
    payload = json.loads(response.output_text)

    def dimension(name: str) -> DimensionScore:
        value = payload[name]
        return DimensionScore(
            score=max(0.0, min(100.0, float(value["score"]))),
            reason=str(value["reason"]).strip(),
        )

    return TutorTurnScore(
        turn_index=turn_index,
        process_adherence=dimension("process_adherence"),
        pedagogical_quality=dimension("pedagogical_quality"),
        naturalness_encouragement=dimension("naturalness_encouragement"),
        feedback=str(payload["feedback"]).strip(),
    )


def judge_trajectory(
    client: OpenAI,
    *,
    judge_model: str,
    trajectory: TutorTrajectory,
    target: RealtimeTrajectoryResult,
    weights: dict[str, float],
    repeats: int = 1,
) -> tuple[float, list[dict[str, Any]]]:
    if len(target.turns) != len(trajectory.turns) + 1:
        raise ValueError("Target trajectory length does not match source trajectory")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    turn_details: list[dict[str, Any]] = []
    turn_scores: list[float] = []

    for turn_index in range(len(target.turns)):
        results = [
            judge_turn(
                client,
                judge_model=judge_model,
                trajectory=trajectory,
                target=target,
                turn_index=turn_index,
            )
            for _ in range(repeats)
        ]

        process = mean(item.process_adherence.score for item in results)
        pedagogy = mean(item.pedagogical_quality.score for item in results)
        natural = mean(item.naturalness_encouragement.score for item in results)
        weighted = (
            float(weights["process_adherence"]) * process
            + float(weights["pedagogical_quality"]) * pedagogy
            + float(weights["naturalness_encouragement"]) * natural
        )
        turn_scores.append(weighted)
        turn_details.append(
            {
                "turn_index": turn_index,
                "is_opening": turn_index == 0,
                "process_adherence": process,
                "pedagogical_quality": pedagogy,
                "naturalness_encouragement": natural,
                "score": weighted,
                "feedback": [item.feedback for item in results],
                "reasons": {
                    "process_adherence": [
                        item.process_adherence.reason for item in results
                    ],
                    "pedagogical_quality": [
                        item.pedagogical_quality.reason for item in results
                    ],
                    "naturalness_encouragement": [
                        item.naturalness_encouragement.reason for item in results
                    ],
                },
            }
        )

    return mean(turn_scores), turn_details
