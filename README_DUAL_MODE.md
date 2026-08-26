# Cloud / Local voice chat

The project implements four voice architectures, but each launch exposes only the two modes that match `BACKEND`. `BACKEND=openai` shows OpenAI Pipeline and OpenAI Full Duplex. `BACKEND=local` shows Local Pipeline and Local Full Duplex. The two visible modes share the same **System prompt**, **Language A** (the user speaks), **Language B** (the assistant answers), **Voice**, transcript UI, and file upload workflow. The **LLM** selector changes automatically with the selected mode.

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

## Install Cloudflare Tunnel

`run.sh` always creates a public HTTPS URL and therefore requires `cloudflared`. On Ubuntu or Debian, install it from [Cloudflare's official APT repository](https://developers.cloudflare.com/tunnel/downloads/):

```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update
sudo apt-get install cloudflared
cloudflared --version
```

The final command must print the installed `cloudflared` version before continuing. A temporary Quick Tunnel does not require a Cloudflare account. A stable hostname requires a Cloudflare account, a named tunnel, and `CLOUDFLARE_TUNNEL_TOKEN` as described below.

## Install the Conda environments

Miniconda or Anaconda is required. Python 3.11 and all Python packages are declared in `environment.openai.yml` and `environment.local.yml`. Both specifications use `conda-forge` plus `nodefaults`, so unrelated channels configured in a user or system `.condarc` are not queried during installation.

Install both environments:

```bash
./install_conda_envs.sh
```

Install or update only one backend environment:

```bash
BACKEND=openai ./install_conda_envs.sh
BACKEND=local ./install_conda_envs.sh
```

`path.sh` is the environment-selection entry point. It uses the fixed Conda executable at `/share/homes/teinhonglo/anaconda3/bin/conda`. The single startup script, `run.sh`, exports `BACKEND`, sources `path.sh`, and activates one of these environments:

| `BACKEND` | Default Conda environment | Extra startup behavior |
|---|---|---|
| `openai` | `voice-chatgpt-openai` | start the web server and Cloudflare Tunnel |
| `local` | `voice-chatgpt-local` | start Ollama, Qdrant, the web server, and Cloudflare Tunnel |

`BACKEND` selects the server environment, dependency bootstrap, API routes, and the two homepage modes. OpenAI launches do not show or enable Local routes; Local launches do not show or enable OpenAI LLM/RAG routes.

To activate an environment manually:

```bash
export BACKEND=openai
source ./path.sh
# or
export BACKEND=local
source ./path.sh
```

## Run with the OpenAI backend environment

```bash
export OPENAI_API_KEY="sk-..."
BACKEND=openai ./run.sh
```

The script starts the web app and Cloudflare Tunnel together. Copy the public HTTPS URL printed by `cloudflared`. The tunnel uses HTTP/2 over TCP so it also works on servers that block outbound QUIC/UDP.

## Run with the Local backend environment (Ollama + Qdrant)

Requirements:

- Docker with the Compose plugin
- NVIDIA Container Toolkit for the GPU configuration in `docker-compose.local.yml`
- an OpenAI API key for ASR and TTS

```bash
export OPENAI_API_KEY="sk-..."
BACKEND=local ./run.sh
```

Use `BACKEND=local` for the two Local modes. It starts the local dependencies and hides the OpenAI-only modes. Use `BACKEND=openai` for the two OpenAI modes; it neither starts nor probes local dependencies.

Local startup activates `voice-chatgpt-local`, starts Ollama and Qdrant, downloads `qwen3:8b` and `bge-m3` when missing, and starts the web app. Set `SKIP_LOCAL_MODEL_PULL=1` to skip the model check. Model data and Qdrant collections use named Docker volumes, so they survive restarts. Check the three local dependencies at <http://localhost:7860/api/local/health>.

## Select an LLM on the homepage

The browser calls `GET /api/models` when the page opens. The backend uses the configured `OPENAI_API_KEY` with OpenAI's [List models API](https://developers.openai.com/api/reference/python/resources/models/methods/list), filters the returned identifiers into Responses text models and Realtime models, and sends only those identifiers to the browser. The API key remains on the server.

The selector changes by mode:

- OpenAI Pipeline lists accessible Responses text models.
- OpenAI Full Duplex lists accessible Realtime models.
- Both Local modes share the models reported by the local OpenAI-compatible `/v1/models` endpoint.

The Local selector also recommends these popular quantized Ollama models whose published weight sizes fit comfortably within a 24 GB RTX 3090 for this application:

| Model | Ollama size | Intended use |
|---|---:|---|
| [`qwen3:8b`](https://ollama.com/library/qwen3:8b) | 5.2 GB | default, multilingual conversation |
| [`qwen3:14b`](https://ollama.com/library/qwen3:14b) | 9.3 GB | stronger multilingual instruction following |
| [`llama3.1:8b`](https://ollama.com/library/llama3.1:8b) | 4.9 GB | popular general conversation |
| [`gemma3:12b`](https://ollama.com/library/gemma3:12b) | 8.1 GB | multilingual general model |
| [`deepseek-r1:14b`](https://ollama.com/library/deepseek-r1:14b) | 9.0 GB | reasoning model, higher latency |

Only `qwen3:8b` is downloaded automatically, so normal installation remains small. Before choosing another recommended model, install it on the server. For example:

```bash
docker compose -f docker-compose.local.yml exec ollama ollama pull qwen3:14b
```

Refresh the page after the download. The model then appears under **已安裝**. Actual VRAM use also depends on context length and concurrent requests; the application does not select the 19–20 GB Qwen variants by default because their remaining VRAM margin is much smaller.

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
export MANAGE_LOCAL_SERVICES=0
BACKEND=local ./run.sh
```

`MANAGE_LOCAL_SERVICES=0` prevents the startup script from launching Docker or pulling Ollama models. The frontend reads models reported by the configured endpoint; the RTX 3090 recommendation list is used only as a convenience. `.env.example` lists every override.

## RAG file upload

The knowledge card automatically targets the engine selected on the homepage:

- an OpenAI mode uploads to an OpenAI vector store;
- a Local mode parses and chunks files on the server, obtains embeddings from the local embedding endpoint, and stores vectors in Qdrant.

The browser maintains separate signed tokens for the OpenAI and Local knowledge bases across launches. The two OpenAI modes share one cloud knowledge base; the two Local modes share one local knowledge base. A token from one backend is never sent to the other backend.

Each upload accepts up to 10 files and defaults to 20 MB per file and 50 MB per request. OpenAI RAG supports `.doc` in addition to the formats below. Local RAG supports `.docx`, `.pptx`, `.pdf`, `.txt`, `.md`, `.json`, `.html`, `.c`, `.cpp`, `.cs`, `.css`, `.go`, `.java`, `.js`, `.php`, `.py`, `.rb`, `.sh`, `.tex`, and `.ts`.

Local deletion removes the selected Qdrant collection. Cloud deletion removes the OpenAI vector store and its uploaded OpenAI file objects.

## Optional models and limits

Important defaults are:

- OpenAI text fallback: `gpt-5.6-luna` (the homepage normally selects from the API-provided list)
- Pipeline transcription: `gpt-transcribe`
- OpenAI TTS: `gpt-4o-mini-tts`
- OpenAI Full Duplex fallback: `gpt-realtime-2` (the homepage normally selects from the API-provided list)
- Local Full Duplex streaming ASR: `gpt-live-transcribe`
- Local LLM: `qwen3:8b`
- Local embedding: `bge-m3`
- Qdrant: `http://127.0.0.1:6333`

Copy `.env.example` values into your environment to change them. Set stable `RAG_TOKEN_SECRET` and `LOCAL_RAG_TOKEN_SECRET` values in production.

## Public HTTPS URL with Cloudflare Tunnel

After installing `cloudflared` with the commands above, `run.sh` uses `BACKEND` to start the matching Conda environment and always creates a public HTTPS tunnel:

```bash
export OPENAI_API_KEY="sk-..."
BACKEND=openai ./run.sh
# or
BACKEND=local ./run.sh
```

Copy the random `https://...trycloudflare.com` URL printed by `cloudflared`. Quick Tunnels are intended only for testing.

`run.sh` forces Cloudflare Tunnel to use HTTP/2 over TCP. The server firewall must allow outbound TCP port `7844`; no inbound port needs to be opened.

For a stable hostname, create a remotely managed tunnel and configure its public hostname service to point to `http://127.0.0.1:7860` (or your `PUBLIC_BIND_HOST` and `PORT`). Then set `CLOUDFLARE_TUNNEL_TOKEN` and run `./run.sh`. The script passes the token through `TUNNEL_TOKEN`, so it is not exposed as a command-line argument. Protect any public hostname with Cloudflare Access and application-level authentication/rate limits. Anyone who can reach an unprotected URL can consume OpenAI quota, local GPU time, and storage.

## API summary

- `GET /api/models`: mode-compatible OpenAI choices, installed Local models, and RTX 3090 recommendations.
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
export BACKEND=openai
source ./path.sh
python -m unittest discover -s tests -p 'test_*.py'
node --check dual_mode/static/app.js
```

The original Gradio experiments remain untouched. Do not log audio, transcripts, prompts, uploaded contents, API keys, or retrieved chunks in production.
