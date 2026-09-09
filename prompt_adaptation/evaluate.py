from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI

from prompt_adaptation.dataset import TutorTrajectory, load_trajectories
from prompt_adaptation.evaluator import judge_trajectory
from prompt_adaptation.realtime_runner import run_realtime_trajectory
from prompt_adaptation.settings import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate direct-transfer and optimized GPT-Realtime-2 prompts on a "
            "held-out trajectory split."
        )
    )
    parser.add_argument("--config", default="prompt_adaptation/config.json")
    parser.add_argument(
        "--candidate-prompt",
        default=None,
        help=(
            "Optimized prompt path. Defaults to "
            "<output_dir>/system_prompt.full_duplex.txt."
        ),
    )
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def read_prompt(path: str | Path) -> str:
    prompt = Path(path).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Empty prompt: {path}")
    return prompt


def run_condition(
    *,
    prompt: str,
    trajectories: list[TutorTrajectory],
    config: dict[str, Any],
    judge_client: OpenAI,
    repeats: int,
) -> dict[str, Any]:
    trajectory_rows: list[dict[str, Any]] = []
    all_turns: list[dict[str, Any]] = []

    for trajectory in trajectories:
        target = run_realtime_trajectory(
            candidate_prompt=prompt,
            trajectory=trajectory,
            audio_dir=config["audio_dir"],
            model=config["target_model"],
            voice=config["realtime_voice"],
            reasoning_effort=config["reasoning_effort"],
            max_output_tokens=config["max_output_tokens"],
        )
        score, turns = judge_trajectory(
            judge_client,
            judge_model=config["judge_model"],
            trajectory=trajectory,
            target=target,
            weights=config["weights"],
            repeats=repeats,
        )
        all_turns.extend(turns)
        trajectory_rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "book": trajectory.book,
                "dacc": trajectory.dacc,
                "score": score,
                "turns": turns,
                "realtime_responses": [
                    item.assistant_text for item in target.turns
                ],
            }
        )
        print(f"{trajectory.trajectory_id}: {score:.2f}")

    aggregate = {
        "score": mean(item["score"] for item in all_turns),
        "process_adherence": mean(
            item["process_adherence"] for item in all_turns
        ),
        "pedagogical_quality": mean(
            item["pedagogical_quality"] for item in all_turns
        ),
        "naturalness_encouragement": mean(
            item["naturalness_encouragement"] for item in all_turns
        ),
    }

    by_turn: dict[str, dict[str, list[float]]] = {}
    for item in all_turns:
        if item["is_opening"]:
            continue
        key = str(item["turn_index"])
        bucket = by_turn.setdefault(
            key,
            {
                "score": [],
                "process_adherence": [],
                "pedagogical_quality": [],
                "naturalness_encouragement": [],
            },
        )
        for metric in bucket:
            bucket[metric].append(float(item[metric]))

    turn_curve = {
        key: {
            metric: mean(values)
            for metric, values in metrics.items()
        }
        for key, metrics in sorted(by_turn.items(), key=lambda pair: int(pair[0]))
    }

    return {
        "aggregate": aggregate,
        "learner_turn_curve": turn_curve,
        "trajectories": trajectory_rows,
    }


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY must be configured")

    config = load_config(args.config)
    trajectories = load_trajectories(config["data_path"], split=args.split)
    source_prompt = read_prompt(config["source_prompt"])
    candidate_path = Path(
        args.candidate_prompt
        or Path(config["output_dir"]) / "system_prompt.full_duplex.txt"
    )
    candidate_prompt = read_prompt(candidate_path)
    repeats = int(config["judge_repeats_final"])
    judge_client = OpenAI()

    print(f"Evaluating direct transfer on {args.split}...")
    direct = run_condition(
        prompt=source_prompt,
        trajectories=trajectories,
        config=config,
        judge_client=judge_client,
        repeats=repeats,
    )

    print(f"Evaluating optimized prompt on {args.split}...")
    adapted = run_condition(
        prompt=candidate_prompt,
        trajectories=trajectories,
        config=config,
        judge_client=judge_client,
        repeats=repeats,
    )

    report = {
        "split": args.split,
        "target_model": config["target_model"],
        "judge_model": config["judge_model"],
        "judge_repeats": repeats,
        "source_prompt": config["source_prompt"],
        "candidate_prompt": str(candidate_path),
        "direct_transfer": direct,
        "adapted": adapted,
        "transfer_gain": (
            adapted["aggregate"]["score"] - direct["aggregate"]["score"]
        ),
        "process_adherence_gain": (
            adapted["aggregate"]["process_adherence"]
            - direct["aggregate"]["process_adherence"]
        ),
    }

    output_path = Path(config["output_dir"]) / f"{args.split}_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluation report: {output_path}")
    print(f"Transfer gain: {report['transfer_gain']:+.2f}")
    print(
        "Process-adherence gain: "
        f"{report['process_adherence_gain']:+.2f}"
    )


if __name__ == "__main__":
    main()
