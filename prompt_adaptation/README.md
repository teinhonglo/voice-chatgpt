# Automatic Prompt Adaptation: Pipeline -> GPT-Realtime-2

This experiment adapts a validated multi-turn reading-tutor policy for `gpt-realtime-2`.

The experiment is intentionally narrow:

- continuous multi-turn reading tutoring
- fixed learner audio for every condition
- one persistent Realtime session per trajectory
- per-turn AI-tutor evaluation
- no barge-in / interruption / VAD evaluation
- no latency metric
- no separate final-report model

The curated Pipeline trajectories are treated as fixed Text-LLM teacher sequences. The Pipeline itself does **not** need to be rerun.

## Experimental conditions

1. **Fixed Text Teacher**: the stored high-quality Pipeline response at every turn.
2. **Direct transfer**: GPT-Realtime-2 + initial/source tutor prompt.
3. **Auto-adapted**: GPT-Realtime-2 + GEPA-optimized tutor prompt.

The primary question is whether automatic prompt adaptation improves Realtime adherence to the intended pedagogical process across continuous turns.

## Metrics

Each Realtime tutor turn is compared directly with the fixed Text Teacher response:

- **Pedagogical Action Alignment** (0.45): same accept/remediate/advance/complete decision and next objective.
- **Semantic Content Alignment** (0.40): same key facts, hints, corrections, and question intent.
- **Response-Form Alignment** (0.15): similar amount of guidance, number of questions, directness, and spoken style.
- **Process Adherence** is retained as a diagnostic metric but is not mixed into the GEPA reward.

The primary GEPA scalar reward is **Reference Alignment**, the weighted sum of the three teacher-alignment dimensions above. By default, the opening is reported but excluded from the optimization reward.

The final report also contains a per-turn curve so you can inspect whether performance degrades as the conversation gets longer.

## Files

```text
prompt_adaptation/
├── config.json
├── dataset.py
├── settings.py
├── prepare_audio.py
├── realtime_runner.py
├── evaluator.py
├── run.py
├── evaluate.py
├── prompts/
│   ├── source_tutoring_policy.txt
│   └── reconstructed_pipeline_system_prompt.txt
├── private_data/                 # ignored by git
│   ├── trajectories.jsonl
│   └── audio/
└── outputs/                      # ignored by git
```

`reconstructed_pipeline_system_prompt.txt` documents the full Pipeline behavior inferred from the supplied conversation logs, including its internal JSON/state contract.

`source_tutoring_policy.txt` is the **portable initial prompt used by this experiment**. It removes Pipeline-only JSON wrappers and the separate final-report behavior so the optimization focuses on tutoring policy rather than output formatting.


## Recommended entry point

Run the whole experiment from the repository root with the stage-controlled wrapper:

```bash
export OPENAI_API_KEY="sk-..."
bash run_prompt_adaptation.sh
```

The wrapper sources `path.sh`, uses `parse_options.sh`, and supports:

```bash
bash run_prompt_adaptation.sh --stage 1 --stop-stage 1
bash run_prompt_adaptation.sh --stage 2 --stop-stage 2
bash run_prompt_adaptation.sh --stage 3 --stop-stage 3
```

Stages are:

- `-1`: install dependencies
- `0`: prepare/verify trajectories and initial prompt
- `1`: generate fixed learner audio
- `2`: run GEPA reference-distillation prompt adaptation
- `3`: run held-out teacher-vs-Realtime evaluation

By default, stage 0 looks for `prompt_adaptation/private_data/SR_prompt_adaptation_trajectories.jsonl` and copies it to the path configured by `data_path`.

Per-stage terminal logs are saved under `prompt_adaptation/outputs/logs/`. GEPA reference-distillation state is stored under `prompt_adaptation/outputs/gepa_reference_distillation/`, so old fitness caches from previous reward definitions are not reused.

## 1. Checkout

```bash
git fetch origin
git checkout exp/auto-prompt-full-duplex
git pull --ff-only
```

## 2. Install

Use the existing OpenAI environment, then install the experiment-only packages:

```bash
export BACKEND=openai
source ./path.sh

pip install -r requirements.prompt_adaptation.txt
export OPENAI_API_KEY="sk-..."
```

## 3. Prepare trajectories

Create the private data directory:

```bash
mkdir -p prompt_adaptation/private_data
```

Copy the curated file into:

```text
prompt_adaptation/private_data/trajectories.jsonl
```

For the supplied SR dataset, use the curated file generated from the Excel conversation log:

```bash
cp SR_prompt_adaptation_trajectories.jsonl   prompt_adaptation/private_data/trajectories.jsonl
```

The current curated file contains 11 book-disjoint trajectories:

- 5 train
- 2 dev
- 4 test

The test split covers DACC 2, 3, 4, and 10.

The separate post-completion final-report LLM has already been removed from these trajectories.

### Trajectory JSONL format

Each line is one complete reading session:

```json
{
  "trajectory_id": "sr_train_001",
  "split": "train",
  "source_session_id": "...",
  "book": "...",
  "dacc": 4,
  "book_context": "[BOOK_CONTEXT]\n...",
  "opening": {
    "expected_teacher_state": "步驟｜向度3｜題型＝開放題｜是否達標＝否",
    "expected_dimension": 3,
    "expected_mastery": "否",
    "source_tutor_response": "..."
  },
  "turns": [
    {
      "turn_index": 1,
      "student_text": "...",
      "expected_teacher_state": "步驟｜向度3｜題型＝開放題｜是否達標＝是",
      "expected_dimension": 3,
      "expected_mastery": "是",
      "expected_question_type": "開放題",
      "expected_completed": false,
      "expected_ib_dacc": null,
      "source_tutor_response": "..."
    }
  ]
}
```

Do not put child trajectories in the public repository. `private_data/` is gitignored.

## 4. Prepare the initial prompt

Default:

```text
prompt_adaptation/prompts/source_tutoring_policy.txt
```

This is the data-inferred portable tutoring policy.

If you have the actual production Pipeline system prompt and want to use that as the initial prompt, create a private prompt file:

```bash
cp /path/to/your_initial_prompt.txt   prompt_adaptation/private_data/initial_prompt.txt
```

Then change `source_prompt` in `prompt_adaptation/config.json`:

```json
{
  "source_prompt": "prompt_adaptation/private_data/initial_prompt.txt"
}
```

For this experiment, the initial prompt should contain the **interactive tutor policy only**. Remove:

- the separate final-report LLM instructions
- book-specific facts
- a single fixed book title
- memorized answers
- Pipeline-only JSON/state output formatting if those fields are not spoken by Full Duplex

Dynamic book context is injected separately for every trajectory.

## 5. Generate fixed learner audio

The curated trajectories contain learner transcripts but not original audio. Generate the audio once:

```bash
python -m prompt_adaptation.prepare_audio   --config prompt_adaptation/config.json
```

This creates 24-kHz PCM files such as:

```text
prompt_adaptation/private_data/audio/
└── sr_train_001/
    ├── turn_01.pcm
    ├── turn_02.pcm
    └── ...
```

Every experimental condition reuses exactly the same files.

If you later have real learner recordings, you can replace these PCM files with mono 24-kHz PCM using the same filenames.

## 6. Run automatic prompt optimization

```bash
python -m prompt_adaptation.run   --config prompt_adaptation/config.json
```

The runner:

1. evaluates GPT-Realtime-2 + the initial prompt against the fixed Text Teacher trajectories on dev
2. uses the 5 train trajectories for GEPA optimization
3. gives GEPA contrastive per-turn feedback containing the teacher response, Realtime response, mismatch diagnosis, and a generalizable prompt recommendation
4. uses the 2 dev trajectories for GEPA validation on Reference Alignment
5. writes the best Full-Duplex prompt
6. evaluates that prompt again on dev

The student inputs are sent as audio. Each trajectory uses one persistent Realtime WebSocket session, so earlier tutor outputs remain in later context.

Outputs:

```text
prompt_adaptation/outputs/
├── system_prompt.full_duplex.txt
├── optimization_report.json
└── gepa_reference_distillation/
```

The main artifact is:

```text
prompt_adaptation/outputs/system_prompt.full_duplex.txt
```

That is the optimized system prompt to use with Full Duplex.

## 7. Run held-out test evaluation

After optimization:

```bash
python -m prompt_adaptation.evaluate   --config prompt_adaptation/config.json
```

This compares on the held-out `test` books:

- GPT-Realtime-2 + initial prompt
- GPT-Realtime-2 + optimized prompt

By default, the judge is repeated 3 times per turn for the final test evaluation.

Output:

```text
prompt_adaptation/outputs/test_evaluation.json
```

The report contains:

```text
direct_transfer.aggregate
adapted.aggregate
transfer_gain
process_adherence_gain

direct_transfer.learner_turn_curve
adapted.learner_turn_curve
```

The `learner_turn_curve` is suitable for plotting scores against turn number, following the same general per-turn analysis style used in long-horizon audio-dialogue evaluation.

## 8. Main configuration

Edit only `prompt_adaptation/config.json` for normal experiments.

Important fields:

```json
{
  "data_path": "prompt_adaptation/private_data/trajectories.jsonl",
  "audio_dir": "prompt_adaptation/private_data/audio",
  "source_prompt": "prompt_adaptation/prompts/source_tutoring_policy.txt",
  "target_model": "gpt-realtime-2",
  "judge_model": "gpt-5.6-luna",
  "reflection_model": "openai/gpt-5.6-luna",
  "max_metric_calls": 60,
  "judge_repeats_final": 3,
  "include_opening_in_reward": false,
  "alignment_weights": {
    "pedagogical_action_alignment": 0.45,
    "semantic_content_alignment": 0.40,
    "response_form_alignment": 0.15
  }
}
```

For a first smoke test, reduce `max_metric_calls` before running the full experiment.

## Notes on privacy and experimental validity

- Real child transcripts are intentionally excluded from git.
- The final-report model is excluded from prompt reconstruction and evaluation.
- The stored source Pipeline response is the fixed sequence-level teacher target. Semantic and behavioral equivalence is rewarded; exact lexical copying is not required.
- GEPA is explicitly instructed not to copy book titles, story facts, learner utterances, expected answers, or source-response wording into the optimized prompt.
- The current experiment evaluates the semantic/pedagogical behavior of continuous audio dialogue. It does not evaluate interruption timing or latency.
