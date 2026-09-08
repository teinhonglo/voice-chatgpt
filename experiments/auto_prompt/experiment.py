#!/usr/bin/env python3
"""Automatic prompt migration from the text tutor to OpenAI Full Duplex.

The target model is called through the Realtime API in text-in/text-out mode so
that prompt adaptation is measured independently of ASR/TTS variability. The
optimized prompt is intended to be copied back into the existing Full Duplex UI
for final audio validation.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

# Allow running this file directly from experiments/auto_prompt/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dual_mode.core import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    TurnSettings,
    build_instructions,
    validate_turn_settings,
)


class TutorJudgeResult(BaseModel):
    """MRBench-inspired tutor evaluation plus source-behavior preservation."""

    instruction_adherence: int = Field(ge=0, le=4)
    behavior_preservation: int = Field(ge=0, le=4)
    pedagogical_correctness: int = Field(ge=0, le=4)
    providing_guidance: int = Field(ge=0, le=4)
    answer_control: int = Field(ge=0, le=4)
    actionability: int = Field(ge=0, le=4)
    coherence: int = Field(ge=0, le=4)
    tutor_tone: int = Field(ge=0, le=4)
    human_likeness: int = Field(ge=0, le=4)
    source_teaching_action: str
    target_teaching_action: str
    hard_constraint_violations: list[str]
    feedback: str


@dataclass(frozen=True)
class ExperimentConfig:
    source_model: str
    target_model: str
    judge_model: str
    reflection_model: str
    language_a: str
    language_b: str
    voice: str
    reasoning_effort: str
    max_prompt_chars: int


JUDGE_INSTRUCTIONS = """You are an evaluator of AI tutor responses.

Evaluate the TARGET tutor against both the intended tutoring requirements and the
SOURCE tutor's pedagogical behavior. Do not reward lexical similarity. Two
responses can be behaviorally equivalent while using different words.

Use a 0-4 scale for every dimension:
0 = severe failure, 1 = poor, 2 = partial/mixed, 3 = good, 4 = fully satisfied.

Dimensions:
- instruction_adherence: follows the scenario expectations and prompt policy.
- behavior_preservation: preserves the SOURCE tutor's pedagogical intent/action,
  not its exact wording.
- pedagogical_correctness: educationally appropriate and factually sound.
- providing_guidance: scaffolds the learner instead of merely answering.
- answer_control: reveals or withholds answers at the pedagogically appropriate
  time. A high score does NOT mean always withholding an answer.
- actionability: gives the learner a clear next step.
- coherence: relevant, consistent, and understandable in context.
- tutor_tone: supportive, respectful, and appropriate for a tutor.
- human_likeness: natural conversational tutoring behavior.

Hard constraints are strict. List every clear violation in
hard_constraint_violations. If there is no violation, return an empty list.
Keep feedback concise but diagnostic: identify the most important behavior the
TARGET should preserve or change.
"""


OPTIMIZATION_OBJECTIVE = """Adapt the existing text-LLM tutor system prompt for a
black-box Full-Duplex Realtime model while preserving the source tutor's intended
pedagogical behavior.

Requirements for the optimized prompt:
1. Preserve the original teaching policy and task semantics.
2. Improve instruction following on the Realtime model, not lexical imitation of
   source responses.
3. Prefer Realtime-friendly structure: short labeled sections, concise bullets,
   explicit trigger -> action -> exception rules, and unambiguous priorities.
4. Preserve one-question-at-a-time and answer-revealing/scaffolding behavior when
   required by the source policy.
5. Do not include scenario-specific answers, names, examples, or memorized test
   content.
6. The result must be a reusable system prompt, not commentary about the prompt.
7. Keep the prompt within the application's character limit.
"""


def load_prompt(path: str | None) -> str:
    if not path:
        return DEFAULT_SYSTEM_PROMPT
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict) or not item.get("id"):
                raise ValueError(f"Each scenario needs an id: {path}:{line_no}")
            if not isinstance(item.get("student_message"), str):
                raise ValueError(f"Scenario {item.get('id')} needs student_message")
            item.setdefault("history", [])
            item.setdefault("expectations", [])
            item.setdefault("hard_constraints", {})
            item.setdefault("split", "train")
            rows.append(item)
    if not rows:
        raise ValueError(f"No scenarios found in {path}")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def settings_for_prompt(prompt: str, cfg: ExperimentConfig) -> TurnSettings:
    return validate_turn_settings(
        system_prompt=prompt,
        language_a=cfg.language_a,
        language_b=cfg.language_b,
        voice=cfg.voice,
    )


def scenario_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in example.get("history", []):
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": example["student_message"].strip()})
    return messages


def run_source_tutor(
    client: OpenAI,
    prompt: str,
    example: dict[str, Any],
    cfg: ExperimentConfig,
) -> str:
    settings = settings_for_prompt(prompt, cfg)
    response = client.responses.create(
        model=cfg.source_model,
        instructions=build_instructions(settings, rag_enabled=False),
        input=scenario_messages(example),
    )
    text = response.output_text.strip()
    if not text:
        raise RuntimeError(f"Empty source response for {example['id']}")
    return text


async def run_realtime_tutor_async(
    prompt: str,
    example: dict[str, Any],
    cfg: ExperimentConfig,
) -> str:
    """Run the target through the actual Realtime WebSocket API using text I/O."""

    settings = settings_for_prompt(prompt, cfg)
    client = AsyncOpenAI()
    output_parts: list[str] = []

    async with client.realtime.connect(model=cfg.target_model) as connection:
        await connection.session.update(
            session={
                "type": "realtime",
                "model": cfg.target_model,
                "instructions": build_instructions(settings, rag_enabled=False),
                "output_modalities": ["text"],
                "reasoning": {"effort": cfg.reasoning_effort},
            }
        )

        for message in scenario_messages(example):
            content_type = "output_text" if message["role"] == "assistant" else "input_text"
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": message["role"],
                    "content": [{"type": content_type, "text": message["content"]}],
                }
            )

        await connection.response.create()
        async for event in connection:
            if event.type == "response.output_text.delta":
                output_parts.append(event.delta)
            elif event.type == "response.done":
                status = getattr(event.response, "status", None)
                if status not in {None, "completed"}:
                    raise RuntimeError(
                        f"Realtime response for {example['id']} ended with status={status}"
                    )
                break

    text = "".join(output_parts).strip()
    if not text:
        raise RuntimeError(f"Empty Realtime response for {example['id']}")
    return text


def run_realtime_tutor(
    prompt: str,
    example: dict[str, Any],
    cfg: ExperimentConfig,
) -> str:
    return asyncio.run(run_realtime_tutor_async(prompt, example, cfg))


def deterministic_violations(
    response: str,
    hard_constraints: dict[str, Any],
) -> list[str]:
    violations: list[str] = []

    max_questions = hard_constraints.get("max_questions")
    if isinstance(max_questions, int) and response.count("?") > max_questions:
        violations.append(
            f"max_questions={max_questions} but response contains {response.count('?')} question marks"
        )

    max_chars = hard_constraints.get("max_chars")
    if isinstance(max_chars, int) and len(response) > max_chars:
        violations.append(f"max_chars={max_chars} but response has {len(response)} characters")

    forbidden = hard_constraints.get("forbidden_phrases", [])
    if isinstance(forbidden, list):
        lowered = response.casefold()
        for phrase in forbidden:
            if isinstance(phrase, str) and phrase and phrase.casefold() in lowered:
                violations.append(f"forbidden phrase present: {phrase!r}")

    return violations


def judge_once(
    client: OpenAI,
    source_response: str,
    target_response: str,
    example: dict[str, Any],
    cfg: ExperimentConfig,
) -> TutorJudgeResult:
    payload = {
        "scenario_id": example["id"],
        "conversation_history": example.get("history", []),
        "student_message": example["student_message"],
        "expected_behaviors": example.get("expectations", []),
        "hard_constraints": example.get("hard_constraints", {}),
        "source_tutor_response": source_response,
        "target_full_duplex_response": target_response,
    }
    parsed = client.responses.parse(
        model=cfg.judge_model,
        instructions=JUDGE_INSTRUCTIONS,
        input=json.dumps(payload, ensure_ascii=False, indent=2),
        text_format=TutorJudgeResult,
    )
    for output in parsed.output:
        if output.type != "message":
            continue
        for content in output.content:
            if content.type == "output_text" and content.parsed is not None:
                return content.parsed
    raise RuntimeError(f"Judge did not return a parsed result for {example['id']}")


def aggregate_judgements(judgements: list[TutorJudgeResult]) -> TutorJudgeResult:
    if len(judgements) == 1:
        return judgements[0]

    scalar_fields = [
        "instruction_adherence",
        "behavior_preservation",
        "pedagogical_correctness",
        "providing_guidance",
        "answer_control",
        "actionability",
        "coherence",
        "tutor_tone",
        "human_likeness",
    ]
    values: dict[str, Any] = {}
    for field in scalar_fields:
        values[field] = int(round(statistics.median(getattr(j, field) for j in judgements)))

    all_violations: list[str] = []
    for judgement in judgements:
        all_violations.extend(judgement.hard_constraint_violations)
    values["hard_constraint_violations"] = sorted(set(all_violations))

    totals = [sum(getattr(j, field) for field in scalar_fields) for j in judgements]
    median_total = statistics.median(totals)
    representative = min(
        zip(judgements, totals), key=lambda pair: abs(pair[1] - median_total)
    )[0]
    values["source_teaching_action"] = representative.source_teaching_action
    values["target_teaching_action"] = representative.target_teaching_action
    values["feedback"] = representative.feedback
    return TutorJudgeResult(**values)


def judge_response(
    client: OpenAI,
    source_response: str,
    target_response: str,
    example: dict[str, Any],
    cfg: ExperimentConfig,
    repeats: int,
) -> TutorJudgeResult:
    judgements = [
        judge_once(client, source_response, target_response, example, cfg)
        for _ in range(max(1, repeats))
    ]
    return aggregate_judgements(judgements)


def score_judgement(
    judgement: TutorJudgeResult,
    deterministic: list[str],
) -> tuple[float, dict[str, float], list[str]]:
    hard_violations = sorted(
        set(judgement.hard_constraint_violations + deterministic)
    )

    pedagogy = statistics.mean(
        [
            judgement.pedagogical_correctness,
            judgement.providing_guidance,
            judgement.answer_control,
            judgement.actionability,
            judgement.coherence,
        ]
    ) / 4.0
    style = statistics.mean(
        [judgement.tutor_tone, judgement.human_likeness]
    ) / 4.0
    instruction = judgement.instruction_adherence / 4.0
    preservation = judgement.behavior_preservation / 4.0

    component_scores = {
        "instruction_adherence": instruction,
        "behavior_preservation": preservation,
        "pedagogical_quality": pedagogy,
        "spoken_style": style,
    }

    if hard_violations:
        return 0.0, component_scores, hard_violations

    overall = (
        0.30 * instruction
        + 0.30 * preservation
        + 0.30 * pedagogy
        + 0.10 * style
    )
    return float(overall), component_scores, hard_violations


def evaluate_candidate(
    prompt: str,
    example: dict[str, Any],
    reference_map: dict[str, str],
    cfg: ExperimentConfig,
    judge_repeats: int,
) -> tuple[float, dict[str, Any]]:
    if len(prompt) > cfg.max_prompt_chars:
        return 0.0, {
            "scores": {
                "instruction_adherence": 0.0,
                "behavior_preservation": 0.0,
                "pedagogical_quality": 0.0,
                "spoken_style": 0.0,
            },
            "error": f"Prompt has {len(prompt)} chars; limit is {cfg.max_prompt_chars}.",
        }

    source_response = reference_map[example["id"]]
    target_response = run_realtime_tutor(prompt, example, cfg)
    client = OpenAI()
    judgement = judge_response(
        client,
        source_response,
        target_response,
        example,
        cfg,
        repeats=judge_repeats,
    )
    deterministic = deterministic_violations(
        target_response, example.get("hard_constraints", {})
    )
    score, component_scores, hard_violations = score_judgement(
        judgement, deterministic
    )
    side_info = {
        "scores": component_scores,
        "scenario_id": example["id"],
        "student_message": example["student_message"],
        "expectations": example.get("expectations", []),
        "source_response": source_response,
        "target_response": target_response,
        "source_teaching_action": judgement.source_teaching_action,
        "target_teaching_action": judgement.target_teaching_action,
        "hard_constraint_violations": hard_violations,
        "judge_feedback": judgement.feedback,
        "raw_judge": judgement.model_dump(),
    }
    return score, side_info


def reference_map_from_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        response = row.get("source_response")
        if isinstance(response, str) and response.strip():
            result[row["id"]] = response.strip()
    return result


def make_cfg(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        source_model=args.source_model,
        target_model=args.target_model,
        judge_model=args.judge_model,
        reflection_model=args.reflection_model,
        language_a=args.language_a,
        language_b=args.language_b,
        voice=args.voice,
        reasoning_effort=args.reasoning_effort,
        max_prompt_chars=args.max_prompt_chars,
    )


def cmd_prepare(args: argparse.Namespace) -> None:
    cfg = make_cfg(args)
    source_prompt = load_prompt(args.source_prompt)
    scenarios = load_jsonl(args.scenarios)
    client = OpenAI()
    rows: list[dict[str, Any]] = []
    for index, example in enumerate(scenarios, 1):
        print(f"[reference {index}/{len(scenarios)}] {example['id']}", flush=True)
        response = run_source_tutor(client, source_prompt, example, cfg)
        rows.append({**example, "source_response": response})
    write_jsonl(args.references, rows)
    print(f"Saved {len(rows)} source references to {args.references}")


def evaluate_split(
    prompt: str,
    rows: list[dict[str, Any]],
    cfg: ExperimentConfig,
    judge_repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = reference_map_from_rows(rows)
    outputs: list[dict[str, Any]] = []
    scores: list[float] = []
    components: dict[str, list[float]] = {
        "instruction_adherence": [],
        "behavior_preservation": [],
        "pedagogical_quality": [],
        "spoken_style": [],
    }

    for index, example in enumerate(rows, 1):
        print(f"[evaluate {index}/{len(rows)}] {example['id']}", flush=True)
        score, info = evaluate_candidate(
            prompt,
            example,
            refs,
            cfg,
            judge_repeats=judge_repeats,
        )
        scores.append(score)
        for name, value in info.get("scores", {}).items():
            if name in components:
                components[name].append(float(value))
        outputs.append({"id": example["id"], "score": score, **info})

    summary = {
        "n": len(scores),
        "mean_score": statistics.mean(scores) if scores else 0.0,
        "median_score": statistics.median(scores) if scores else 0.0,
        "component_means": {
            key: statistics.mean(values) if values else 0.0
            for key, values in components.items()
        },
        "hard_violation_rate": (
            sum(bool(row.get("hard_constraint_violations")) for row in outputs)
            / len(outputs)
            if outputs
            else 0.0
        ),
    }
    return outputs, summary


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = make_cfg(args)
    prompt = load_prompt(args.prompt)
    rows = load_jsonl(args.references)
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]
    if not rows:
        raise ValueError(f"No scenarios for split={args.split}")
    outputs, summary = evaluate_split(
        prompt, rows, cfg, judge_repeats=args.judge_repeats
    )
    write_jsonl(args.output, outputs)
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {args.output}")


def make_reflection_lm(model: str):
    client = OpenAI()

    def reflection_lm(prompt_or_messages: Any, **_: Any) -> str:
        response = client.responses.create(
            model=model,
            instructions=(
                "You are optimizing a reusable system prompt. Analyze evaluation "
                "failures carefully and propose precise generalizable improvements."
            ),
            input=prompt_or_messages,
        )
        text = response.output_text.strip()
        if not text:
            raise RuntimeError("Reflection model returned empty output")
        return text

    return reflection_lm


def cmd_optimize(args: argparse.Namespace) -> None:
    try:
        import gepa.optimize_anything as oa
        from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig
    except ImportError as exc:
        raise RuntimeError(
            "GEPA is not installed. Run: pip install -r experiments/auto_prompt/requirements.txt"
        ) from exc

    cfg = make_cfg(args)
    source_prompt = load_prompt(args.source_prompt)
    rows = load_jsonl(args.references)
    refs = reference_map_from_rows(rows)
    missing = [row["id"] for row in rows if row["id"] not in refs]
    if missing:
        raise ValueError(
            "References are missing source_response for: " + ", ".join(missing[:10])
        )

    trainset = [row for row in rows if row.get("split") == "train"]
    valset = [row for row in rows if row.get("split") == "dev"]
    if not trainset:
        raise ValueError("At least one train scenario is required")
    if not valset:
        raise ValueError("At least one dev scenario is required for prompt selection")

    def evaluator(candidate: dict[str, str] | str, example: dict[str, Any]):
        if isinstance(candidate, dict):
            prompt = candidate.get("system_prompt", "")
        else:
            prompt = candidate
        score, side_info = evaluate_candidate(
            prompt,
            example,
            refs,
            cfg,
            judge_repeats=args.judge_repeats,
        )
        oa.log(f"Scenario: {example['id']}")
        oa.log(f"Source response: {side_info.get('source_response', '')}")
        oa.log(f"Target response: {side_info.get('target_response', '')}")
        oa.log(f"Judge feedback: {side_info.get('judge_feedback', side_info.get('error', ''))}")
        oa.log(
            "Hard violations: "
            + json.dumps(side_info.get("hard_constraint_violations", []), ensure_ascii=False)
        )
        return score, side_info

    reflection_lm = make_reflection_lm(cfg.reflection_model)
    objective = OPTIMIZATION_OBJECTIVE + f"\n8. Maximum length: {cfg.max_prompt_chars} characters."

    result = oa.optimize_anything(
        seed_candidate={"system_prompt": source_prompt},
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        objective=objective,
        config=GEPAConfig(
            engine=EngineConfig(
                max_metric_calls=args.max_metric_calls,
                parallel=False,
                cache_evaluation=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                reflection_minibatch_size=args.reflection_minibatch_size,
            ),
        ),
    )

    best_candidate = getattr(result, "best_candidate", None)
    if isinstance(best_candidate, dict):
        best_prompt = best_candidate.get("system_prompt")
    elif isinstance(best_candidate, str):
        best_prompt = best_candidate
    else:
        best_prompt = None
    if not best_prompt:
        raise RuntimeError("GEPA finished without a best system prompt")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_system_prompt.txt"
    best_path.write_text(best_prompt.strip() + "\n", encoding="utf-8")

    test_rows = [row for row in rows if row.get("split") == "test"]
    summary: dict[str, Any] = {
        "source_model": cfg.source_model,
        "target_model": cfg.target_model,
        "judge_model": cfg.judge_model,
        "reflection_model": cfg.reflection_model,
        "max_metric_calls": args.max_metric_calls,
        "source_prompt_chars": len(source_prompt),
        "best_prompt_chars": len(best_prompt),
        "best_system_prompt": str(best_path),
    }
    if test_rows:
        test_outputs, test_summary = evaluate_split(
            best_prompt,
            test_rows,
            cfg,
            judge_repeats=args.final_judge_repeats,
        )
        write_jsonl(output_dir / "test_results.jsonl", test_outputs)
        summary["test"] = test_summary

        baseline_outputs, baseline_summary = evaluate_split(
            source_prompt,
            test_rows,
            cfg,
            judge_repeats=args.final_judge_repeats,
        )
        write_jsonl(output_dir / "direct_transfer_test_results.jsonl", baseline_outputs)
        summary["direct_transfer_test"] = baseline_summary
        summary["test_gain"] = (
            test_summary["mean_score"] - baseline_summary["mean_score"]
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Best Full-Duplex system prompt: {best_path}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-model",
        default=os.getenv("AUTO_PROMPT_SOURCE_MODEL", os.getenv("OPENAI_TEXT_MODEL", "gpt-5.6-luna")),
    )
    parser.add_argument(
        "--target-model",
        default=os.getenv("AUTO_PROMPT_TARGET_MODEL", os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")),
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("AUTO_PROMPT_JUDGE_MODEL", "gpt-5.6-sol"),
    )
    parser.add_argument(
        "--reflection-model",
        default=os.getenv("AUTO_PROMPT_REFLECTION_MODEL", "gpt-5.6-sol"),
    )
    parser.add_argument("--language-a", default="zh-TW")
    parser.add_argument("--language-b", default="en")
    parser.add_argument("--voice", default="marin")
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["minimal", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--max-prompt-chars", type=int, default=12_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-adapt a text tutor prompt to a Full-Duplex Realtime model."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Generate/caches source text-tutor responses")
    add_common_args(prepare)
    prepare.add_argument("--source-prompt")
    prepare.add_argument(
        "--scenarios",
        default="experiments/auto_prompt/scenarios.example.jsonl",
    )
    prepare.add_argument(
        "--references",
        default="experiments/auto_prompt/exp/references.jsonl",
    )
    prepare.set_defaults(func=cmd_prepare)

    evaluate = sub.add_parser("evaluate", help="Evaluate any prompt on the Realtime target")
    add_common_args(evaluate)
    evaluate.add_argument("--prompt", required=True)
    evaluate.add_argument(
        "--references",
        default="experiments/auto_prompt/exp/references.jsonl",
    )
    evaluate.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    evaluate.add_argument("--judge-repeats", type=int, default=3)
    evaluate.add_argument(
        "--output",
        default="experiments/auto_prompt/exp/evaluation.jsonl",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    optimize = sub.add_parser("optimize", help="Optimize the prompt with GEPA")
    add_common_args(optimize)
    optimize.add_argument("--source-prompt")
    optimize.add_argument(
        "--references",
        default="experiments/auto_prompt/exp/references.jsonl",
    )
    optimize.add_argument("--max-metric-calls", type=int, default=60)
    optimize.add_argument("--reflection-minibatch-size", type=int, default=3)
    optimize.add_argument(
        "--judge-repeats",
        type=int,
        default=1,
        help="Judge repeats during search; keep 1 for cost, use final-judge-repeats for reporting.",
    )
    optimize.add_argument("--final-judge-repeats", type=int, default=3)
    optimize.add_argument(
        "--output-dir",
        default="experiments/auto_prompt/exp/gepa",
    )
    optimize.set_defaults(func=cmd_optimize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        parser.error("OPENAI_API_KEY is required")
    args.func(args)


if __name__ == "__main__":
    main()
