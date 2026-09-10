from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "data_path",
        "audio_dir",
        "source_prompt",
        "output_dir",
        "target_model",
        "judge_model",
        "reflection_model",
        "tts_model",
        "tts_voice",
        "realtime_voice",
        "reasoning_effort",
        "max_output_tokens",
        "max_metric_calls",
        "judge_repeats_final",
        "include_opening_in_reward",
        "alignment_weights",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(missing)}")

    weights = config["alignment_weights"]
    expected_weights = {
        "pedagogical_action_alignment",
        "semantic_content_alignment",
        "response_form_alignment",
    }
    if set(weights) != expected_weights:
        raise ValueError(
            f"alignment_weights must contain exactly {sorted(expected_weights)}"
        )
    total = sum(float(weights[key]) for key in expected_weights)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"alignment_weights must sum to 1.0, got {total}")
    return config
