from __future__ import annotations

import argparse
import os
from pathlib import Path

from openai import OpenAI

from prompt_adaptation.dataset import audio_path, load_trajectories
from prompt_adaptation.settings import load_config


def _response_bytes(response: object) -> bytes:
    if hasattr(response, "read"):
        content = response.read()
    else:
        content = getattr(response, "content", None)
    if not isinstance(content, bytes):
        raise RuntimeError("Speech API returned unexpected content")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize fixed 24-kHz PCM student audio for prompt-adaptation trajectories."
    )
    parser.add_argument("--config", default="prompt_adaptation/config.json")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate audio files that already exist.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise SystemExit("OPENAI_API_KEY must be configured")

    config = load_config(args.config)
    trajectories = load_trajectories(config["data_path"])
    client = OpenAI()

    created = 0
    reused = 0
    for trajectory in trajectories:
        for turn in trajectory.turns:
            output_path = audio_path(
                config["audio_dir"], trajectory.trajectory_id, turn.turn_index
            )
            if output_path.exists() and not args.overwrite:
                reused += 1
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            response = client.audio.speech.create(
                model=config["tts_model"],
                voice=config["tts_voice"],
                input=turn.student_text,
                instructions=(
                    "Speak the learner utterance naturally in Traditional Chinese. "
                    "Use a child or young learner conversational pace. Do not add or omit words."
                ),
                response_format="pcm",
            )
            output_path.write_bytes(_response_bytes(response))
            created += 1
            print(f"created {output_path}")

    print(f"Done. created={created}, reused={reused}")


if __name__ == "__main__":
    main()
