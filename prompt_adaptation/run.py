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
        weights=config["weights"],
        repeats=repeats,
    )
    responses = [turn.assistant_text for turn in realtime.turns]
    return score, details, responses


def _dimension_means(details: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "process_adherence": mean(item["process_adherence"] for item in details),
        "pedagogical_quality": mean(item["pedagogical_quality"] for item in details),
        "naturalness_encouragement": mean(
            item["naturalness_encouragement"] for item in details
        ),
        "score": mean(item["score"] for item in details),
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
        "aggregate": _dimension_means(all_turns),
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
            score, details, _ = _run_and_judge(
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

        metrics = _dimension_means(details)
        feedback_lines = []
        for item in details:
            turn_feedback = " | ".join(item.get("feedback", []))
            if turn_feedback:
                feedback_lines.append(
                    f"Turn {item['turn_index']}: score={item['score']:.1f}; "
                    f"{turn_feedback}"
                )

        side_info = {
            "scores": {
                "process_adherence": metrics["process_adherence"] / 100.0,
                "pedagogical_quality": metrics["pedagogical_quality"] / 100.0,
                "naturalness_encouragement": metrics[
                    "naturalness_encouragement"
                ]
                / 100.0,
            },
            "Trajectory": (
                f"DACC={trajectory.dacc}, {len(trajectory.turns)} learner turns"
            ),
            "Feedback": "\n".join(feedback_lines),
        }
        return score / 100.0, side_info

    objective = """Adapt the supplied reading-tutor system prompt for GPT-Realtime-2.

The target model receives continuous multi-turn learner AUDIO in one persistent Realtime
session. Preserve the validated Pipeline tutor policy, especially the required reading
sequence, mastery-based transitions, same-objective remediation after incorrect or
incomplete answers, book-grounded corrections, DACC-appropriate difficulty, natural
encouragement, and correct lesson completion.

The optimized artifact must remain a GENERAL system prompt. Never copy or encode book
titles, character names, story facts, learner utterances, expected answers, trajectory IDs,
or wording from individual Pipeline responses. Book-specific context is supplied separately
at runtime. Do not add JSON output requirements or final-report behavior. Prefer clear,
Realtime-friendly labeled sections and explicit trigger -> action -> exception rules where
they improve adherence."""

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    gepa_dir = output_dir / "gepa"

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
        "reflection_model": config["reflection_model"],
        "source_prompt": config["source_prompt"],
        "train_ids": [item.trajectory_id for item in train],
        "dev_ids": [item.trajectory_id for item in dev],
        "direct_transfer_dev": direct_dev,
        "adapted_dev": adapted_dev,
        "dev_transfer_gain": (
            adapted_dev["aggregate"]["score"]
            - direct_dev["aggregate"]["score"]
        ),
        "gepa": {
            "best_idx": result.best_idx,
            "best_validation_score": result.val_aggregate_scores[result.best_idx],
            "total_metric_calls": result.total_metric_calls,
        },
    }
    report_path = output_dir / "optimization_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Final prompt: {prompt_path}")
    print(f"Optimization report: {report_path}")
    print(f"Dev transfer gain: {report['dev_transfer_gain']:+.2f}")


if __name__ == "__main__":
    main()
