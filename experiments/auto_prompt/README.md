# Automatic Text-to-Full-Duplex Prompt Adaptation

This experiment starts from a system prompt that already works well with the Text-LLM tutor pipeline and automatically adapts it for the project's OpenAI Full Duplex model.

The final artifact is:

    experiments/auto_prompt/exp/gepa/best_system_prompt.txt

The production WebRTC UI is intentionally left unchanged. During prompt search, the target is still called through the actual Realtime WebSocket API, but with text input/output. This isolates prompt and pedagogical behavior from microphone, ASR, playback, and network timing. After selection, copy the best prompt into the existing System prompt field and validate it with real speech.

## Method

1. Run the existing Text-LLM tutor with the original prompt and cache reference responses.
2. Directly transfer the same prompt to GPT-Realtime and measure the migration gap on the dev split.
3. Use GEPA to evolve only the system prompt. The original text-tutor prompt is the seed.
4. For every candidate, run the same tutoring scenarios on GPT-Realtime.
5. An independent GPT-5.6 Sol judge scores instruction adherence, source-behavior preservation, pedagogical quality, answer control, guidance, coherence, tone, and human-likeness.
6. Feed both the scalar score and diagnostic feedback back to GEPA.
7. Select on the dev split. Only after prompt selection, compare direct transfer and the adapted prompt on the held-out test split.

The evaluator is inspired by MRBench / Unifying AI Tutor Evaluation (NAACL 2025). It adds behavior_preservation because prompt migration should preserve the source tutor's teaching action, not its exact wording. A deterministic or judge-detected hard-constraint violation forces the optimization score to zero.

## Why GEPA

GEPA is a black-box optimizer for text artifacts. It accepts a candidate prompt, an evaluator score, and Actionable Side Information (diagnostic feedback). It therefore fits this project's API-only setting without model weights or fine-tuning.

## Install

    export BACKEND=openai
    source ./path.sh
    pip install -r experiments/auto_prompt/requirements.txt

## Data

scenarios.example.jsonl is only a smoke-test/example dataset. For a paper or serious comparison, replace it with representative tutoring cases from the application and keep a fixed train/dev/test split.

Each JSONL row contains:

    {
      "id": "wrong_answer_scaffold",
      "split": "train",
      "history": [{"role": "assistant", "content": "..."}],
      "student_message": "...",
      "expectations": ["Do not immediately reveal the answer", "Give a hint"],
      "hard_constraints": {"max_questions": 1}
    }

Supported deterministic hard constraints are max_questions, max_chars, and forbidden_phrases.

## Run

    export OPENAI_API_KEY="sk-..."
    bash experiments/auto_prompt/run.sh \
      --source_model gpt-5.6-luna \
      --target_model gpt-realtime-2 \
      --judge_model gpt-5.6-sol \
      --reflection_model gpt-5.6-sol \
      --max_metric_calls 60

To adapt your actual prompt:

    bash experiments/auto_prompt/run.sh \
      --source_prompt /path/to/your_text_tutor_system_prompt.txt

The wrapper follows the repository's stage/stop_stage convention:

    # Only generate source references
    bash experiments/auto_prompt/run.sh --stage 0 --stop_stage 0

    # Re-run optimization when references already exist
    bash experiments/auto_prompt/run.sh --stage 2 --stop_stage 3

## Stages

- 0: Generate source Text-LLM reference responses.
- 1: Evaluate direct transfer of the source prompt on the dev split.
- 2: Optimize the Realtime prompt with GEPA.
- 3: Compare the direct-transfer baseline and selected prompt on held-out test scenarios.

## Search score

The scalar search score is behavior-oriented rather than lexical:

- 0.30 Instruction adherence
- 0.30 Source behavior preservation
- 0.30 Pedagogical quality
- 0.10 Spoken conversational style

Pedagogical quality averages pedagogical correctness, guidance, answer control, actionability, and coherence. Any hard-constraint violation makes the score zero. BLEU/ROUGE are intentionally not used.

During search, one judge call per example is the default to control API cost. Final held-out evaluation uses three calls and aggregates dimensions by median. For publication-quality results, annotate a subset with human raters and report judge agreement/correlation.

## Recommended comparison

1. Text pipeline + original prompt (source/reference)
2. Realtime + original prompt (direct-transfer baseline)
3. Realtime + manually rewritten Realtime-style prompt
4. Realtime + GEPA-adapted prompt

Report instruction adherence, behavior preservation, pedagogical quality, hard-violation rate, and direct-transfer-to-adapted gain. Evaluate Full-Duplex timing behavior such as interruptions, premature response, and stop latency separately from Realtime event logs because transcript-only judging cannot measure timing.

## References

- GEPA: https://github.com/gepa-ai/gepa
- MRBench / Unifying AI Tutor Evaluation (NAACL 2025): https://aclanthology.org/2025.naacl-long.57/
- OpenAI Realtime prompting guide: https://developers.openai.com/api/docs/guides/realtime-models-prompting
- OpenAI Python Realtime example: https://github.com/openai/openai-python/blob/main/examples/realtime/realtime.py
