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



def _delta_text(value: float) -> str:
    return f"{value:+.2f}"


def _improvement_label(value: float) -> str:
    if value > 0:
        return "提升"
    if value < 0:
        return "下降"
    return "持平"


def _markdown_quote(text: str) -> str:
    value = str(text).strip()
    if not value:
        return "> [空]"
    return "\n".join(f"> {line}" if line else ">" for line in value.splitlines())


def _feedback_text(turn: dict[str, Any]) -> str:
    items = [
        str(item).strip()
        for item in turn.get("feedback", [])
        if str(item).strip()
    ]
    if not items:
        return "[無]"
    return " / ".join(dict.fromkeys(items))


def write_markdown_report(
    *,
    report: dict[str, Any],
    trajectories: list[TutorTrajectory],
    output_path: str | Path,
) -> None:
    direct = report["direct_transfer"]
    adapted = report["adapted"]
    direct_aggregate = direct["aggregate"]
    adapted_aggregate = adapted["aggregate"]

    metrics = (
        ("Overall score", "score"),
        ("Process adherence", "process_adherence"),
        ("Pedagogical quality", "pedagogical_quality"),
        ("Naturalness & encouragement", "naturalness_encouragement"),
    )

    lines: list[str] = [
        f"# Prompt Adaptation {report['split'].title()} Report",
        "",
        "Before = GPT-Realtime-2 + initial prompt.  ",
        "After = GPT-Realtime-2 + optimized prompt.",
        "",
        "## 1. Performance Summary",
        "",
        "| Metric | Before | After | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]

    for label, key in metrics:
        before = float(direct_aggregate[key])
        after = float(adapted_aggregate[key])
        lines.append(
            f"| {label} | {before:.2f} | {after:.2f} | "
            f"{_delta_text(after - before)} |"
        )

    overall_delta = (
        float(adapted_aggregate["score"]) - float(direct_aggregate["score"])
    )
    process_delta = (
        float(adapted_aggregate["process_adherence"])
        - float(direct_aggregate["process_adherence"])
    )
    lines.extend(
        [
            "",
            (
                f"**整體 Performance：{_improvement_label(overall_delta)} "
                f"{_delta_text(overall_delta)} 分。**  "
            ),
            (
                f"**Process Adherence：{_improvement_label(process_delta)} "
                f"{_delta_text(process_delta)} 分。**"
            ),
            "",
            "## 2. Trajectory Summary",
            "",
            "| Trajectory | Book | DACC | Before | After | Delta |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    direct_by_id = {
        item["trajectory_id"]: item for item in direct["trajectories"]
    }
    adapted_by_id = {
        item["trajectory_id"]: item for item in adapted["trajectories"]
    }

    for trajectory in trajectories:
        before_row = direct_by_id[trajectory.trajectory_id]
        after_row = adapted_by_id[trajectory.trajectory_id]
        delta = float(after_row["score"]) - float(before_row["score"])
        book = trajectory.book.replace("|", r"\|")
        lines.append(
            f"| {trajectory.trajectory_id} | {book} | {trajectory.dacc} | "
            f"{float(before_row['score']):.2f} | "
            f"{float(after_row['score']):.2f} | {_delta_text(delta)} |"
        )

    lines.extend(["", "## 3. Turn-by-Turn Qualitative Comparison", ""])

    for trajectory in trajectories:
        before_row = direct_by_id[trajectory.trajectory_id]
        after_row = adapted_by_id[trajectory.trajectory_id]
        before_responses = before_row["realtime_responses"]
        after_responses = after_row["realtime_responses"]
        before_turns = before_row["turns"]
        after_turns = after_row["turns"]

        lines.extend(
            [
                (
                    f"### {trajectory.trajectory_id} — "
                    f"{trajectory.book} (DACC {trajectory.dacc})"
                ),
                "",
            ]
        )

        for position in range(len(before_responses)):
            if position == 0:
                turn_label = "Opening"
                learner_text = "[START_LESSON]"
                expected_state = trajectory.opening.expected_teacher_state
                pipeline_reference = trajectory.opening.source_tutor_response
            else:
                source_turn = trajectory.turns[position - 1]
                turn_label = f"Turn {source_turn.turn_index}"
                learner_text = source_turn.student_text
                expected_state = source_turn.expected_teacher_state
                pipeline_reference = source_turn.source_tutor_response

            before_turn = before_turns[position]
            after_turn = after_turns[position]

            lines.extend(
                [
                    f"#### {turn_label}",
                    "",
                    "**Learner**",
                    "",
                    _markdown_quote(learner_text),
                    "",
                    "**Expected teacher state**",
                    "",
                    _markdown_quote(expected_state),
                    "",
                    "**Pipeline reference**",
                    "",
                    _markdown_quote(pipeline_reference),
                    "",
                    "**Before — Initial prompt**",
                    "",
                    _markdown_quote(before_responses[position]),
                    "",
                    "**After — Optimized prompt**",
                    "",
                    _markdown_quote(after_responses[position]),
                    "",
                    "| Metric | Before | After | Delta |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )

            for metric_label, key in metrics:
                before_score = float(before_turn[key])
                after_score = float(after_turn[key])
                lines.append(
                    f"| {metric_label} | {before_score:.2f} | "
                    f"{after_score:.2f} | "
                    f"{_delta_text(after_score - before_score)} |"
                )

            lines.extend(
                [
                    "",
                    (
                        "**Before judge feedback:** "
                        f"{_feedback_text(before_turn)}"
                    ),
                    "",
                    (
                        "**After judge feedback:** "
                        f"{_feedback_text(after_turn)}"
                    ),
                    "",
                ]
            )

    Path(output_path).write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


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
    markdown_path = Path(config["output_dir"]) / f"{args.split}_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(
        report=report,
        trajectories=trajectories,
        output_path=markdown_path,
    )
    print(f"Evaluation JSON: {output_path}")
    print(f"Readable report: {markdown_path}")
    print(f"Transfer gain: {report['transfer_gain']:+.2f}")
    print(
        "Process-adherence gain: "
        f"{report['process_adherence_gain']:+.2f}"
    )


if __name__ == "__main__":
    main()
