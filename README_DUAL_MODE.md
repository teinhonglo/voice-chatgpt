# Cloud / Local voice chat

The homepage provides one-click switching between four voice modes. All modes share the same **System prompt**, **Language A** (the user speaks), **Language B** (the assistant answers), **Voice**, transcript UI, and file upload workflow.

| Mode | ASR | LLM + RAG | TTS | Interaction |
|---|---|---|---|---|
| OpenAI Pipeline | OpenAI | OpenAI Responses + hosted vector store | OpenAI | recorded turns |
| OpenAI Full Duplex | OpenAI Realtime | OpenAI Realtime + hosted vector-store tool | OpenAI Realtime | end-to-end duplex |
| Local Pipeline | OpenAI | local OpenAI-compatible LLM + local embeddings + Qdrant | OpenAI | recorded turns |
| Local Full Duplex | OpenAI streaming transcription | local OpenAI-compatible LLM + local embeddings + Qdrant | OpenAI streaming chunks | continuous microphone + barge-in |

Local Full Duplex is a **cascaded duplex** implementation, not a local end-to-end speech-to-speech foundation model. The microphone remains open while output plays. When VAD reports that the user has started speaking, the browser immediately stops queued audio and cancels the active local generation request. It normally has more latency than OpenAI Full Duplex because ASR, retrieval, LLM, and TTS are separate stages.

## Data boundary

In both Local modes, uploaded files are parsed by this server and stored only in the configured local embedding service and Qdrant; the System prompt, retrieved chunks, history, and LLM reasoning go to the configured local LLM endpoint. User audio still goes to OpenAI ASR, and generated answer segments go to OpenAI TTS. Therefore these modes are locally managed for **LLM and vector data**, but they are not fully offline.

The OpenAI API key never goes to the browser.

## Install and run OpenAI-only modes

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.dual_mode.txt
export OPENAI_API_KEY="sk-..."
./run_dual_mode.sh
```

Open <http://localhost:7860>. Browsers allow microphone access on localhost. Remote deployments must use HTTPS.

## Install and run Local modes (Ollama + Qdrant)

Requirements:

- Docker with the Compose plugin
- NVIDIA Container Toolkit for the GPU configuration in `docker-compose.local.yml`
- an OpenAI API key for ASR and TTS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.dual_mode.txt
export OPENAI_API_KEY="sk-..."
./run_local.sh
```

`run_local.sh` starts Ollama and Qdrant, pulls `qwen3:8b` and `bge-m3`, and starts the web app. Model data and Qdrant collections use named Docker volumes, so they survive restarts. Check the three local dependencies at <http://localhost:7860/api/local/health>.

For a CPU-only machine, remove the `deploy.resources.reservations.devices` block from `docker-compose.local.yml`. Generation will be much slower.

To stop the app, press Ctrl-C. To stop local services without deleting data:

```bash
docker compose -f docker-compose.local.yml stop
```

Do not use `docker compose down -v` unless you intend to delete all downloaded models and local Qdrant data.

## Use vLLM or another local server

The app uses standard OpenAI-compatible Chat Completions and Embeddings endpoints. Configure separate endpoints if the LLM and embedding model are served by different processes:

```bash
export LOCAL_LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LOCAL_LLM_API_KEY="local"
export LOCAL_LLM_MODEL="your-chat-model"
export LOCAL_EMBEDDING_BASE_URL="http://127.0.0.1:8001/v1"
export LOCAL_EMBEDDING_API_KEY="local"
export LOCAL_EMBEDDING_MODEL="your-embedding-model"
export QDRANT_URL="http://127.0.0.1:6333"
./run_dual_mode.sh
```

No model names are hard-coded in application logic; `.env.example` lists every override.

## RAG file upload

The knowledge card automatically targets the engine selected on the homepage:

- an OpenAI mode uploads to an OpenAI vector store;
- a Local mode parses and chunks files on the server, obtains embeddings from the local embedding endpoint, and stores vectors in Qdrant.

The browser maintains separate signed tokens for the OpenAI and Local knowledge bases, so switching modes never sends a cloud vector-store token to Qdrant or a local token to OpenAI. The two OpenAI modes share one cloud knowledge base; the two Local modes share one local knowledge base.

Each upload accepts up to 10 files and defaults to 20 MB per file and 50 MB per request. OpenAI RAG supports `.doc` in addition to the formats below. Local RAG supports `.docx`, `.pptx`, `.pdf`, `.txt`, `.md`, `.json`, `.html`, `.c`, `.cpp`, `.cs`, `.css`, `.go`, `.java`, `.js`, `.php`, `.py`, `.rb`, `.sh`, `.tex`, and `.ts`.

Local deletion removes the selected Qdrant collection. Cloud deletion removes the OpenAI vector store and its uploaded OpenAI file objects.

## Optional models and limits

Important defaults are:

- OpenAI text: `gpt-4.1-mini`
- Pipeline transcription: `gpt-transcribe`
- OpenAI TTS: `gpt-4o-mini-tts`
- OpenAI Full Duplex: `gpt-realtime-2.1`
- Local Full Duplex streaming ASR: `gpt-live-transcribe`
- Local LLM: `qwen3:8b`
- Local embedding: `bge-m3`
- Qdrant: `http://127.0.0.1:6333`

Copy `.env.example` values into your environment to change them. Set stable `RAG_TOKEN_SECRET` and `LOCAL_RAG_TOKEN_SECRET` values in production.

## Public HTTPS URL with Cloudflare Tunnel

Install [`cloudflared`](https://developers.cloudflare.com/tunnel/downloads/), start either `run_dual_mode.sh` or `run_local.sh`, and in another terminal run:

```bash
cloudflared tunnel --url http://127.0.0.1:7860
```

Alternatively, `./run_public.sh` starts the app and a Quick Tunnel together. Copy the random `https://...trycloudflare.com` URL printed by `cloudflared`. Quick Tunnels are intended only for testing.

For a stable hostname, set `CLOUDFLARE_TUNNEL_TOKEN` and run `./run_public.sh`. Protect any public hostname with Cloudflare Access and application-level authentication/rate limits. Anyone who can reach an unprotected URL can consume OpenAI quota, local GPU time, and storage.

## API summary

- `POST /api/pipeline/turn`: OpenAI Pipeline turn.
- `POST /api/realtime/session`: OpenAI end-to-end Realtime WebRTC SDP exchange.
- `POST /api/local/pipeline/turn`: OpenAI ASR/TTS with local RAG and LLM.
- `POST /api/local/realtime/session`: transcription-only OpenAI Realtime WebRTC SDP exchange.
- `POST /api/local/realtime/turn`: NDJSON stream containing local LLM text deltas and ordered OpenAI TTS chunks.
- `POST /api/rag/upload`, `/api/rag/delete`, `/api/rag/search`: OpenAI hosted RAG.
- `POST /api/local/rag/upload`, `/api/local/rag/delete`: local parsing, embedding, and Qdrant lifecycle.
- `GET /api/health`, `/api/local/health`: configuration and dependency health.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
node --check dual_mode/static/app.js
```

The original Gradio experiments remain untouched. Do not log audio, transcripts, prompts, uploaded contents, API keys, or retrieved chunks in production.
