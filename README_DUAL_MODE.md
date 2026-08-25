# OpenAI dual-mode voice chat

This implementation adds a new browser UI with a one-click switch between two OpenAI-only voice modes:

| Mode | Flow | Best for |
|---|---|---|
| Pipeline | speech-to-text → Responses API → text-to-speech | predictable turns, clear transcripts, easier debugging |
| Full Duplex | browser WebRTC ↔ OpenAI Realtime API | low-latency natural conversation with interruption |

Both modes share the homepage settings for **System prompt**, **Language A** (the user speaks), **Language B** (the assistant answers), and **Voice**. The API key never goes to the browser.

## Run locally

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.dual_mode.txt
export OPENAI_API_KEY="sk-..."
./run_dual_mode.sh
```

Open <http://localhost:7860>. Browsers allow microphone access on localhost. For a remote deployment, serve the app over HTTPS.

The page defaults to Traditional Chinese input, English output, and the `marin` voice. Settings are saved in the browser and may be changed at any time. In an active Full Duplex call, disconnect and reconnect to apply changed settings.

## Optional model overrides

Copy the names from `.env.example` into your deployment environment to change models. Defaults are:

- Text: `gpt-4.1-mini`
- Pipeline transcription: `gpt-transcribe`
- Text-to-speech: `gpt-4o-mini-tts`
- Full Duplex: `gpt-realtime-2.1`
- Full Duplex input transcript: `gpt-4o-mini-transcribe`

## Architecture and security

- `POST /api/pipeline/turn` accepts one browser recording and runs all three Pipeline calls on the server.
- `POST /api/realtime/session` accepts the browser SDP offer and forwards it with the configured Realtime session to OpenAI's unified WebRTC endpoint.
- The browser receives only the SDP answer. The standard server API key remains server-side.
- The server validates language codes, voice names, prompt size, history roles, SDP size, and recording size.
- The UI explicitly discloses that its spoken output is AI-generated.

For production, add authentication, per-user rate limits, request logging with redaction, and your own retention policy. Do not log audio, transcripts, prompts, or API keys by default.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
node --check dual_mode/static/app.js
```

The original Gradio experiments remain untouched, so the new implementation can be evaluated and rolled back independently.
