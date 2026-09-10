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
            "Evaluate direct-transfer and optimized GPT-Realtime-2 prompts "
            "against fixed historical Text LLM teacher trajectories."
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


def _scored_turns(
    turns: list[dict[str, Any]],
    *,
    include_opening: bool,
) -> list[dict[str, Any]]:
    scored = [
        item for item in turns
        if include_opening or not item["is_opening"]
    ]
    if not scored:
        raise ValueError("No scored turns")
    return scored


def _aggregate(
    turns: list[dict[str, Any]],
    *,
    include_opening: bool,
) -> dict[str, float]:
    scored = _scored_turns(turns, include_opening=include_opening)
    keys = (
        "reference_alignment",
        "pedagogical_action_alignment",
        "semantic_content_alignment",
        "response_form_alignment",
        "process_adherence",
    )
    return {
        key: mean(float(item[key]) for item in scored)
        for key in keys
    }


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
            alignment_weights=config["alignment_weights"],
            repeats=repeats,
            include_opening_in_reward=bool(
                config["include_opening_in_reward"]
            ),
        )
        all_turns.extend(turns)
        trajectory_rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "book": trajectory.book,
                "dacc": trajectory.dacc,
                "reference_alignment": score,
                "turns": turns,
                "realtime_responses": [
                    item.assistant_text for item in target.turns
                ],
            }
        )
        print(
            f"{trajectory.trajectory_id}: "
            f"reference_alignment={score:.2f}"
        )

    by_turn: dict[str, dict[str, list[float]]] = {}
    for item in all_turns:
        if item["is_opening"]:
            continue
        key = str(item["turn_index"])
        bucket = by_turn.setdefault(
            key,
            {
                "reference_alignment": [],
                "pedagogical_action_alignment": [],
                "semantic_content_alignment": [],
                "response_form_alignment": [],
                "process_adherence": [],
            },
        )
        for metric in bucket:
            bucket[metric].append(float(item[metric]))

    turn_curve = {
        key: {
            metric: mean(values)
            for metric, values in metrics.items()
        }
        for key, metrics in sorted(
            by_turn.items(),
            key=lambda pair: int(pair[0]),
        )
    }

    return {
        "aggregate": _aggregate(
            all_turns,
            include_opening=bool(config["include_opening_in_reward"]),
        ),
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
    return "\n".join(
        f"> {line}" if line else ">"
        for line in value.splitlines()
    )


def _unique_text(items: list[Any]) -> str:
    values = [
        str(item).strip()
        for item in items
        if str(item).strip()
    ]
    if not values:
        return "[無]"
    return " / ".join(dict.fromkeys(values))


def _trajectory_lookup(
    condition: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        item["trajectory_id"]: item
        for item in condition["trajectories"]
    }


def write_markdown_report(
    *,
    report: dict[str, Any],
    trajectories: list[TutorTrajectory],
    output_path: str | Path,
) -> None:
    direct = report["direct_transfer"]
    adapted = report["adapted"]
    before_agg = direct["aggregate"]
    after_agg = adapted["aggregate"]

    metrics = (
        ("Reference alignment (primary)", "reference_alignment"),
        (
            "Pedagogical action alignment",
            "pedagogical_action_alignment",
        ),
        ("Semantic content alignment", "semantic_content_alignment"),
        ("Response-form alignment", "response_form_alignment"),
        ("Process adherence (diagnostic)", "process_adherence"),
    )

    lines: list[str] = [
        f"# Prompt Distillation {report['split'].title()} Report",
        "",
        "Teacher = fixed historical Text LLM / Pipeline response.  ",
        "Before = GPT-Realtime-2 + initial prompt.  ",
        "After = GPT-Realtime-2 + optimized prompt.",
        "",
        (
            "Primary optimization target: **Reference Alignment**. "
            "Process Adherence is diagnostic and is not mixed into the "
            "GEPA scalar reward."
        ),
        "",
        "## 1. Performance Summary",
        "",
        "| Metric | Before | After | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]

    for label, key in metrics:
        before = float(before_agg[key])
        after = float(after_agg[key])
        lines.append(
            f"| {label} | {before:.2f} | {after:.2f} | "
            f"{_delta_text(after - before)} |"
        )

    primary_delta = (
        float(after_agg["reference_alignment"])
        - float(before_agg["reference_alignment"])
    )
    lines.extend(
        [
            "",
            (
                f"**Reference Alignment："
                f"{_improvement_label(primary_delta)} "
                f"{_delta_text(primary_delta)} 分。**"
            ),
            "",
            "## 2. Trajectory Summary",
            "",
            "| Trajectory | Book | DACC | Before | After | Delta |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    direct_by_id = _trajectory_lookup(direct)
    adapted_by_id = _trajectory_lookup(adapted)

    for trajectory in trajectories:
        before_row = direct_by_id[trajectory.trajectory_id]
        after_row = adapted_by_id[trajectory.trajectory_id]
        before = float(before_row["reference_alignment"])
        after = float(after_row["reference_alignment"])
        book = trajectory.book.replace("|", r"\|")
        lines.append(
            f"| {trajectory.trajectory_id} | {book} | "
            f"{trajectory.dacc} | {before:.2f} | {after:.2f} | "
            f"{_delta_text(after - before)} |"
        )

    lines.extend(
        ["", "## 3. Turn-by-Turn Teacher–Realtime Comparison", ""]
    )

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
                expected_state = (
                    trajectory.opening.expected_teacher_state
                )
                teacher_response = (
                    trajectory.opening.source_tutor_response
                )
            else:
                source_turn = trajectory.turns[position - 1]
                turn_label = f"Turn {source_turn.turn_index}"
                learner_text = source_turn.student_text
                expected_state = source_turn.expected_teacher_state
                teacher_response = source_turn.source_tutor_response

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
                    "**Fixed Text Teacher target**",
                    "",
                    _markdown_quote(teacher_response),
                    "",
                    "**Before — Realtime + initial prompt**",
                    "",
                    _markdown_quote(before_responses[position]),
                    "",
                    "**After — Realtime + optimized prompt**",
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
                        "**Before mismatch:** "
                        f"{_unique_text(before_turn.get('feedback', []))}"
                    ),
                    "",
                    (
                        "**After mismatch:** "
                        f"{_unique_text(after_turn.get('feedback', []))}"
                    ),
                    "",
                    (
                        "**Before prompt advice:** "
                        f"{_unique_text(before_turn.get('prompt_advice', []))}"
                    ),
                    "",
                    (
                        "**After prompt advice:** "
                        f"{_unique_text(after_turn.get('prompt_advice', []))}"
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
    trajectories = load_trajectories(
        config["data_path"],
        split=args.split,
    )
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
        "alignment_weights": config["alignment_weights"],
        "include_opening_in_reward": config[
            "include_opening_in_reward"
        ],
        "direct_transfer": direct,
        "adapted": adapted,
        "reference_alignment_gain": (
            adapted["aggregate"]["reference_alignment"]
            - direct["aggregate"]["reference_alignment"]
        ),
        "pedagogical_action_alignment_gain": (
            adapted["aggregate"]["pedagogical_action_alignment"]
            - direct["aggregate"]["pedagogical_action_alignment"]
        ),
        "semantic_content_alignment_gain": (
            adapted["aggregate"]["semantic_content_alignment"]
            - direct["aggregate"]["semantic_content_alignment"]
        ),
        "response_form_alignment_gain": (
            adapted["aggregate"]["response_form_alignment"]
            - direct["aggregate"]["response_form_alignment"]
        ),
        "process_adherence_gain": (
            adapted["aggregate"]["process_adherence"]
            - direct["aggregate"]["process_adherence"]
        ),
    }

    output_path = (
        Path(config["output_dir"])
        / f"{args.split}_evaluation.json"
    )
    markdown_path = (
        Path(config["output_dir"])
        / f"{args.split}_report.md"
    )
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
    print(
        "Reference-alignment gain: "
        f"{report['reference_alignment_gain']:+.2f}"
    )
    print(
        "Action-alignment gain: "
        f"{report['pedagogical_action_alignment_gain']:+.2f}"
    )
    print(
        "Semantic-content gain: "
        f"{report['semantic_content_alignment_gain']:+.2f}"
    )


if __name__ == "__main__":
    main()
