from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import gepa.optimize_anything as oa
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)
from openai import OpenAI

from prompt_adaptation.dataset import TutorTrajectory, load_trajectories
from prompt_adaptation.evaluator import judge_trajectory
from prompt_adaptation.realtime_runner import run_realtime_trajectory
from prompt_adaptation.settings import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a Full-Duplex reading-tutor system prompt against curated "
            "multi-turn Pipeline trajectories."
        )
    )
    parser.add_argument("--config", default="prompt_adaptation/config.json")
    return parser.parse_args()


def read_prompt(path: str | Path) -> str:
    prompt = Path(path).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Empty prompt: {path}")
    return prompt


def _candidate_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate.strip()
    if isinstance(candidate, dict):
        if "current_candidate" in candidate:
            return str(candidate["current_candidate"]).strip()
        if len(candidate) == 1:
            return str(next(iter(candidate.values()))).strip()
    raise TypeError(f"Unexpected GEPA candidate type: {type(candidate)!r}")


def _run_and_judge(
    *,
    prompt: str,
    trajectory: TutorTrajectory,
    config: dict[str, Any],
    judge_client: OpenAI,
    repeats: int = 1,
) -> tuple[float, list[dict[str, Any]], list[str]]:
    realtime = run_realtime_trajectory(
        candidate_prompt=prompt,
        trajectory=trajectory,
        audio_dir=config["audio_dir"],
        model=config["target_model"],
        voice=config["realtime_voice"],
        reasoning_effort=config["reasoning_effort"],
        max_output_tokens=config["max_output_tokens"],
    )
    score, details = judge_trajectory(
        judge_client,
        judge_model=config["judge_model"],
        trajectory=trajectory,
        target=realtime,
        alignment_weights=config["alignment_weights"],
        repeats=repeats,
        include_opening_in_reward=bool(config["include_opening_in_reward"]),
    )
    responses = [turn.assistant_text for turn in realtime.turns]
    return score, details, responses


def _dimension_means(
    details: list[dict[str, Any]],
    *,
    include_opening: bool,
) -> dict[str, float]:
    scored = [
        item for item in details
        if include_opening or not item["is_opening"]
    ]
    if not scored:
        raise ValueError("No scored turns")
    return {
        "reference_alignment": mean(
            item["reference_alignment"] for item in scored
        ),
        "pedagogical_action_alignment": mean(
            item["pedagogical_action_alignment"] for item in scored
        ),
        "semantic_content_alignment": mean(
            item["semantic_content_alignment"] for item in scored
        ),
        "response_form_alignment": mean(
            item["response_form_alignment"] for item in scored
        ),
        "process_adherence": mean(
            item["process_adherence"] for item in scored
        ),
    }


def evaluate_split(
    *,
    prompt: str,
    trajectories: list[TutorTrajectory],
    config: dict[str, Any],
    judge_client: OpenAI,
    repeats: int = 1,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    all_turns: list[dict[str, Any]] = []

    for trajectory in trajectories:
        score, details, responses = _run_and_judge(
            prompt=prompt,
            trajectory=trajectory,
            config=config,
            judge_client=judge_client,
            repeats=repeats,
        )
        all_turns.extend(details)
        items.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "book": trajectory.book,
                "dacc": trajectory.dacc,
                "score": score,
                "turns": details,
                "realtime_responses": responses,
            }
        )
        print(
            f"{trajectory.split}/{trajectory.trajectory_id}: "
            f"{score:.2f}"
        )

    return {
        "aggregate": _dimension_means(
            all_turns,
            include_opening=bool(config["include_opening_in_reward"]),
        ),
        "trajectories": items,
    }


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY must be configured")

    config = load_config(args.config)
    source_prompt = read_prompt(config["source_prompt"])
    train = load_trajectories(config["data_path"], split="train")
    dev = load_trajectories(config["data_path"], split="dev")

    judge_client = OpenAI()
    trajectory_by_id = {item.trajectory_id: item for item in [*train, *dev]}

    print("Evaluating direct-transfer prompt on dev trajectories...")
    direct_dev = evaluate_split(
        prompt=source_prompt,
        trajectories=dev,
        config=config,
        judge_client=judge_client,
        repeats=1,
    )

    def evaluator(candidate: str, example: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        prompt = _candidate_text(candidate)
        trajectory = trajectory_by_id[str(example["trajectory_id"])]
        try:
            score, details, responses = _run_and_judge(
                prompt=prompt,
                trajectory=trajectory,
                config=config,
                judge_client=judge_client,
                repeats=1,
            )
        except Exception as exc:
            oa.log(f"Realtime rollout failed: {exc}")
            return 0.0, {
                "Error": str(exc),
                "Feedback": "The candidate could not complete the Realtime trajectory.",
            }

        metrics = _dimension_means(
            details,
            include_opening=bool(config["include_opening_in_reward"]),
        )
        feedback_lines = []
        for item, target_response in zip(details, responses):
            if item["is_opening"]:
                teacher_response = trajectory.opening.source_tutor_response
            else:
                teacher_response = trajectory.turns[
                    item["turn_index"] - 1
                ].source_tutor_response
            turn_feedback = " | ".join(item.get("feedback", []))
            prompt_advice = " | ".join(item.get("prompt_advice", []))
            feedback_lines.append(
                "\n".join(
                    [
                        (
                            f"Turn {item['turn_index']}: "
                            f"reference_alignment="
                            f"{item['reference_alignment']:.1f}"
                        ),
                        f"TEXT TEACHER: {teacher_response}",
                        f"REALTIME: {target_response}",
                        f"MISMATCH: {turn_feedback}",
                        f"PROMPT ADVICE: {prompt_advice}",
                    ]
                )
            )

        side_info = {
            "scores": {
                "reference_alignment": (
                    metrics["reference_alignment"] / 100.0
                ),
                "pedagogical_action_alignment": (
                    metrics["pedagogical_action_alignment"] / 100.0
                ),
                "semantic_content_alignment": (
                    metrics["semantic_content_alignment"] / 100.0
                ),
                "response_form_alignment": (
                    metrics["response_form_alignment"] / 100.0
                ),
                "process_adherence": (
                    metrics["process_adherence"] / 100.0
                ),
            },
            "Trajectory": (
                f"DACC={trajectory.dacc}, "
                f"{len(trajectory.turns)} learner turns"
            ),
            "Feedback": "\n\n".join(feedback_lines),
        }
        return score / 100.0, side_info

    objective = """Perform black-box sequence distillation from a validated Text LLM
reading tutor to GPT-Realtime-2 by optimizing only the Realtime system prompt.

For every trajectory, the stored historical Pipeline tutor response is the FIXED teacher
target. The primary scalar reward is reference_alignment. Improve the Realtime response so
it matches the teacher's pedagogical action, semantic content, next-question intent,
amount of scaffolding, and response form. Natural paraphrasing is allowed; lexical copying
is not required.

Treat each mismatch as a contrastive teacher-vs-student error. Read the supplied TEXT
TEACHER, REALTIME, MISMATCH, and PROMPT ADVICE fields and infer a GENERAL prompt rule that
would make future Realtime outputs closer to the teacher. Do not optimize for an
independently plausible tutoring style when it differs from the teacher.

The Realtime model runs on its own accumulated multi-turn history, so prompt changes must
also prevent context drift across later turns. Never copy book titles, character names,
story facts, learner utterances, expected answers, trajectory IDs, or teacher wording into
the system prompt. Do not add JSON output or final-report behavior."""


    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    gepa_dir = output_dir / "gepa_reference_distillation"

    print("Running GEPA prompt optimization...")
    result = optimize_anything(
        seed_candidate=source_prompt,
        evaluator=evaluator,
        dataset=[{"trajectory_id": item.trajectory_id} for item in train],
        valset=[{"trajectory_id": item.trajectory_id} for item in dev],
        objective=objective,
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=int(config["max_metric_calls"]),
                cache_evaluation=True,
                parallel=False,
                max_workers=1,
                run_dir=str(gepa_dir),
                seed=42,
                display_progress_bar=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm=str(config["reflection_model"]),
            ),
        ),
    )

    best_prompt = _candidate_text(result.best_candidate)
    prompt_path = output_dir / "system_prompt.full_duplex.txt"
    prompt_path.write_text(best_prompt + "\n", encoding="utf-8")

    print("Evaluating optimized prompt on dev trajectories...")
    adapted_dev = evaluate_split(
        prompt=best_prompt,
        trajectories=dev,
        config=config,
        judge_client=judge_client,
        repeats=1,
    )

    report = {
        "target_model": config["target_model"],
        "judge_model": config["judge_model"],
        "alignment_weights": config["alignment_weights"],
        "include_opening_in_reward": config["include_opening_in_reward"],
        "reflection_model": config["reflection_model"],
        "source_prompt": config["source_prompt"],
        "train_ids": [item.trajectory_id for item in train],
        "dev_ids": [item.trajectory_id for item in dev],
        "direct_transfer_dev": direct_dev,
        "adapted_dev": adapted_dev,
        "dev_reference_alignment_gain": (
            adapted_dev["aggregate"]["reference_alignment"]
            - direct_dev["aggregate"]["reference_alignment"]
        ),
        "gepa": {
            "best_idx": result.best_idx,
            "best_validation_score": result.val_aggregate_scores[result.best_idx],
            "total_metric_calls": result.total_metric_calls,
            "candidate_validation_scores": result.val_aggregate_scores,
            "best_so_far_validation_scores": [
                max(result.val_aggregate_scores[: index + 1])
                for index in range(len(result.val_aggregate_scores))
            ],
        },
    }
    report_path = output_dir / "optimization_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Final prompt: {prompt_path}")
    print(f"Optimization report: {report_path}")
    print(
        "Dev reference-alignment gain: "
        f"{report['dev_reference_alignment_gain']:+.2f}"
    )


if __name__ == "__main__":
    main()
