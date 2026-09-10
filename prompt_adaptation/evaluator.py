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
    pedagogical_action_alignment: DimensionScore
    semantic_content_alignment: DimensionScore
    response_form_alignment: DimensionScore
    process_adherence: DimensionScore
    feedback: str
    prompt_advice: str

    def reference_alignment(self, weights: dict[str, float]) -> float:
        return (
            float(weights["pedagogical_action_alignment"])
            * self.pedagogical_action_alignment.score
            + float(weights["semantic_content_alignment"])
            * self.semantic_content_alignment.score
            + float(weights["response_form_alignment"])
            * self.response_form_alignment.score
        )


EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pedagogical_action_alignment": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
        "semantic_content_alignment": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
        "response_form_alignment": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
        "process_adherence": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
        "feedback": {"type": "string"},
        "prompt_advice": {"type": "string"},
    },
    "required": [
        "pedagogical_action_alignment",
        "semantic_content_alignment",
        "response_form_alignment",
        "process_adherence",
        "feedback",
        "prompt_advice",
    ],
    "additionalProperties": False,
}


def _target_history_text(
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


def _teacher_history_text(
    trajectory: TutorTrajectory,
    current_index: int,
) -> str:
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
        expected_completed = False
        teacher_response = trajectory.opening.source_tutor_response
        target_response = target.turns[0].assistant_text
        target_history = "[Lesson opening]"
        teacher_history = "[Lesson opening]"
    else:
        source_turn = trajectory.turns[turn_index - 1]
        student_text = source_turn.student_text
        expected_state = source_turn.expected_teacher_state
        expected_completed = source_turn.expected_completed
        teacher_response = source_turn.source_tutor_response
        target_response = target.turns[turn_index].assistant_text
        target_history = _target_history_text(trajectory, target, turn_index)
        teacher_history = _teacher_history_text(trajectory, turn_index)

    return f"""You are evaluating black-box sequence distillation from a validated TEXT LLM
reading-tutor trajectory to a GPT-Realtime audio tutor.

The HISTORICAL TEXT TEACHER RESPONSE is the fixed distillation target for this turn.
The goal is NOT to judge whether the TARGET is independently a good tutor. The goal is to
measure how closely the TARGET preserves the teacher's behavior while allowing natural
paraphrasing. A response that is reasonable but takes a different pedagogical action,
asks a different question, adds/removes important guidance, or changes the progression
must lose alignment score.

BOOK CONTEXT (authoritative):
{trajectory.book_context}

DACC LEVEL:
{trajectory.dacc}

EXPECTED TEACHER STATE:
{expected_state}

EXPECTED LESSON COMPLETED AFTER THIS TURN:
{expected_completed}

TEXT TEACHER HISTORY BEFORE THIS TURN:
{teacher_history}

TARGET REALTIME HISTORY BEFORE THIS TURN:
{target_history}

CURRENT LEARNER TURN:
{student_text}

HISTORICAL TEXT TEACHER RESPONSE (fixed target):
{teacher_response}

TARGET REALTIME RESPONSE:
{target_response}

Score the TARGET on 0-100 for these dimensions:

1. pedagogical_action_alignment
   - Does TARGET make the same accept/remediate/clarify/advance/complete decision?
   - Does it stay on or move to the same learning objective?
   - Does it ask the same kind of next question or provide the same kind of scaffold?
   - Major state-transition disagreement should receive a low score even if TARGET sounds good.

2. semantic_content_alignment
   - Does TARGET preserve the teacher's key factual content, correction, hint, evidence,
     and intended question semantics?
   - Penalize missing important teacher content, contradictory content, or important
     extra content that changes what the learner is being asked to do.
   - Do not penalize harmless paraphrases or wording differences.

3. response_form_alignment
   - Does TARGET roughly match the teacher's response granularity, amount of guidance,
     number of questions, directness, encouragement, and spoken interaction style?
   - Exact sentence structure or lexical copying is NOT required.

4. process_adherence
   - Independently diagnose whether TARGET is compatible with EXPECTED TEACHER STATE.
   - This is a diagnostic metric only. It is not the main distillation reward.

Important:
- The fixed TEXT TEACHER response is the primary target.
- Do not give a high alignment score merely because TARGET is pedagogically plausible.
- Do not reward stylistic polish for compensating for a different pedagogical action.
- Do not require word-for-word copying.
- Use the histories to diagnose accumulated multi-turn drift.

In feedback, state the most important teacher-target mismatch.
In prompt_advice, give one concise GENERAL system-prompt change that could reduce this
kind of mismatch. Do not encode this book's names, facts, learner utterances, or answer.
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
            "Act as a strict reference-based distillation evaluator. "
            "The historical Text LLM response is the fixed teacher target. "
            "Judge behavioral and semantic closeness, not independent response quality. "
            "Natural paraphrases are allowed."
        ),
        input=_make_prompt(
            trajectory=trajectory,
            target=target,
            turn_index=turn_index,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "TutorDistillationEvaluation",
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
        pedagogical_action_alignment=dimension(
            "pedagogical_action_alignment"
        ),
        semantic_content_alignment=dimension(
            "semantic_content_alignment"
        ),
        response_form_alignment=dimension("response_form_alignment"),
        process_adherence=dimension("process_adherence"),
        feedback=str(payload["feedback"]).strip(),
        prompt_advice=str(payload["prompt_advice"]).strip(),
    )


def judge_trajectory(
    client: OpenAI,
    *,
    judge_model: str,
    trajectory: TutorTrajectory,
    target: RealtimeTrajectoryResult,
    alignment_weights: dict[str, float],
    repeats: int = 1,
    include_opening_in_reward: bool = False,
) -> tuple[float, list[dict[str, Any]]]:
    if len(target.turns) != len(trajectory.turns) + 1:
        raise ValueError("Target trajectory length does not match teacher trajectory")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    turn_details: list[dict[str, Any]] = []
    reward_scores: list[float] = []

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

        action = mean(
            item.pedagogical_action_alignment.score for item in results
        )
        semantic = mean(
            item.semantic_content_alignment.score for item in results
        )
        form = mean(item.response_form_alignment.score for item in results)
        process = mean(item.process_adherence.score for item in results)
        reference_alignment = (
            float(alignment_weights["pedagogical_action_alignment"]) * action
            + float(alignment_weights["semantic_content_alignment"]) * semantic
            + float(alignment_weights["response_form_alignment"]) * form
        )

        if include_opening_in_reward or turn_index > 0:
            reward_scores.append(reference_alignment)

        turn_details.append(
            {
                "turn_index": turn_index,
                "is_opening": turn_index == 0,
                "reference_alignment": reference_alignment,
                "pedagogical_action_alignment": action,
                "semantic_content_alignment": semantic,
                "response_form_alignment": form,
                "process_adherence": process,
                "feedback": [item.feedback for item in results],
                "prompt_advice": [item.prompt_advice for item in results],
                "reasons": {
                    "pedagogical_action_alignment": [
                        item.pedagogical_action_alignment.reason
                        for item in results
                    ],
                    "semantic_content_alignment": [
                        item.semantic_content_alignment.reason
                        for item in results
                    ],
                    "response_form_alignment": [
                        item.response_form_alignment.reason
                        for item in results
                    ],
                    "process_adherence": [
                        item.process_adherence.reason for item in results
                    ],
                },
            }
        )

    if not reward_scores:
        raise ValueError("No turns available for trajectory reward")
    return mean(reward_scores), turn_details
