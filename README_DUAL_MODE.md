# Cloud / Local voice chat

The project implements four voice architectures, but each launch exposes only the two modes that match `BACKEND`. `BACKEND=openai` shows OpenAI Pipeline and OpenAI Full Duplex. `BACKEND=local` shows Local Pipeline and Local Full Duplex. The **LLM** selector changes automatically with the selected mode: the Local Pipeline selector contains multilingual Ollama text models, while Local Full Duplex uses a dedicated speech-to-speech model.

| Mode | ASR | LLM + RAG | TTS | Interaction |
|---|---|---|---|---|
| OpenAI Pipeline | OpenAI | OpenAI Responses + hosted vector store | OpenAI | recorded turns |
| OpenAI Full Duplex | OpenAI Realtime | OpenAI Realtime + hosted vector-store tool | OpenAI Realtime | end-to-end duplex |
| Local Pipeline | OpenAI | local OpenAI-compatible LLM + local embeddings + Qdrant | OpenAI | recorded turns |
| Local Full Duplex | MiniCPM-o 4.5 | MiniCPM-o 4.5 GPTQ | MiniCPM-o 4.5 | native local speech-to-speech duplex |

Local Full Duplex is now a **native end-to-end speech-to-speech** implementation. The browser sends mono 16 kHz PCM16 frames through the app's same-origin WebSocket proxy to a private vLLM-Omni MiniCPM-o runtime, and plays returned 24 kHz PCM16 chunks while the microphone remains active. It no longer reuses the Ollama text model and no longer calls OpenAI ASR or TTS.

MiniCPM-o 4.5 has broader multilingual understanding, but its model card specifically guarantees realtime speech conversation in **Chinese and English**. The Local Full Duplex language selectors therefore expose only Traditional/Simplified Mandarin and English. Local Pipeline retains the full multilingual language selector.

## Data boundary

Local Pipeline parses uploaded files on this server and stores vectors in the configured local embedding service and Qdrant. Its prompt, retrieved chunks, history, and answer generation go to the local Ollama-compatible text endpoint; recorded user audio still goes to OpenAI ASR, and the answer text goes to OpenAI TTS.

Local Full Duplex sends microphone audio, instructions, and generated speech only to the configured private MiniCPM-o endpoint. It does not use the OpenAI API and currently does not support RAG. The OpenAI API key is still required by `run.sh --backend local` because the same deployment also exposes Local Pipeline.

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
| `local` | `voice-chatgpt-local` | start Ollama, Qdrant, MiniCPM-o, the web server, and Cloudflare Tunnel |

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
./run.sh --backend openai --port 7860
```

The script starts the web app and Cloudflare Tunnel together. Copy the public HTTPS URL printed by `cloudflared`. The tunnel uses HTTP/2 over TCP so it also works on servers that block outbound QUIC/UDP.

## Run with the Local backend environment (Ollama + Qdrant + MiniCPM-o)

Requirements:

- Docker with the Compose plugin
- NVIDIA Container Toolkit for the GPU configuration in `docker-compose.local.yml`
- NVIDIA driver 525.60.13 or newer (CUDA 13 / R580 is not required)
- a 24 GB NVIDIA GPU (RTX 3090-class) for the documented quantized defaults
- enough disk for the vLLM-Omni image, Qwen text weights, MiniCPM-o GPTQ weights, and Hugging Face cache
- an OpenAI API key for the Local Pipeline ASR/TTS path

```bash
export OPENAI_API_KEY="sk-..."
./run.sh --backend local --gpuid 0 --port 7860
```

Use `BACKEND=local` for the two Local modes. It starts the local dependencies and hides the OpenAI-only modes. Use `BACKEND=openai` for the two OpenAI modes; it neither starts nor probes local dependencies.

Local startup activates `voice-chatgpt-local`, starts Ollama, Qdrant and vLLM-Omni, downloads `qwen3.5:9b` and `bge-m3` when missing, and loads `openbmb/MiniCPM-o-4_5-GPTQ`. The first launch builds the MiniCPM-o service from the official `vllm/vllm-openai:v0.26.0-cu129` base and the matching vLLM-Omni `v0.26.0` source, then downloads the Hugging Face model. It is large and can take several minutes; later starts reuse Docker's build cache and named model volumes. `LOCAL_DUPLEX_STARTUP_TIMEOUT_SECONDS` defaults to 900 seconds. Set `SKIP_LOCAL_MODEL_PULL=1` to skip only the Ollama model checks. Check all local dependencies at <http://localhost:7860/api/local/health>.

The `-cu129` base is intentional. The unsuffixed vLLM 0.26 image requires CUDA 13 and an R580-or-newer driver. NVIDIA documents CUDA 12.x minor-version compatibility for Linux drivers from 525.60.13 through R579, so `start_local_services.sh` checks the selected GPU's driver before Docker starts and enables compatibility mode only in that range. This supports an RTX 3090 without blindly upgrading the host to CUDA 13. Verify the selected GPU and driver with:

```bash
nvidia-smi --id=2 --query-gpu=name,driver_version --format=csv,noheader
```

The startup script also downloads MiniCPM-o's official reference WAV into the ignored `runtime/ref_audio` directory. That clip defines the speech model's assistant voice. Override `LOCAL_DUPLEX_REF_AUDIO` with another compatible WAV when a different reference voice is required.

The managed Docker Ollama does not reserve host port `11434`. Docker assigns an available loopback port automatically, and `start_local_services.sh` discovers that mapping and configures the Local LLM and embedding URLs. An existing host Ollama can therefore continue listening on `11434` without conflicting with this project. The selected Docker Ollama endpoint is printed during startup.

The managed MiniCPM-o service uses the same pattern for container port `8099`; it receives a random available loopback host port. The browser never connects to that port directly. FastAPI exposes `/api/local/duplex` and proxies a restricted set of PCM16/playback WebSocket events to the private service.

`run.sh` parses `--backend`, `--gpuid`, and `--port` through `parse_options.sh`. `--backend` defaults to `openai`, `--gpuid` defaults to `0`, and `--port` defaults to `7860`. The GPU option selects the NVIDIA device assigned to the Local Ollama container. For example:

```bash
./run.sh --backend local --gpuid 1 --port 8888
```

To run the OpenAI and Local deployments simultaneously, give them different ports. With Quick Tunnels, each process prints its own public URL:

```bash
./run.sh --backend openai --port 7860
./run.sh --backend local --gpuid 1 --port 8888
```

## Select an LLM on the homepage

The browser calls `GET /api/models` when the page opens. The backend uses the configured `OPENAI_API_KEY` with OpenAI's [List models API](https://developers.openai.com/api/reference/python/resources/models/methods/list), filters the returned identifiers into Responses text models and Realtime models, and sends only those identifiers to the browser. The API key remains on the server.

The selector changes by mode:

- OpenAI Pipeline lists accessible Responses text models.
- OpenAI Full Duplex lists accessible Realtime models.
- Local Pipeline lists installed and recommended multilingual Ollama text models.
- Local Full Duplex uses the separately loaded `openbmb/MiniCPM-o-4_5-GPTQ` speech model.

The Local selector also recommends these popular quantized Ollama models whose published weight sizes fit comfortably within a 24 GB RTX 3090 for this application:

| Model | Ollama size | Intended use |
|---|---:|---|
| [`qwen3.5:9b`](https://ollama.com/library/qwen3.5:9b) | 6.6 GB | default; 201 languages and dialects |
| [`qwen3:14b`](https://ollama.com/library/qwen3:14b) | 9.3 GB | mature multilingual instruction following |
| [`gemma3:12b`](https://ollama.com/library/gemma3:12b) | 8.1 GB | general model supporting 140+ languages |

The default text choice is based on current published artifacts rather than a hard-coded assumption that every popular model is equally multilingual. Ollama reports `qwen3.5:9b` as a 6.6 GB Q4_K_M artifact with 18.6M downloads and 201 supported languages/dialects. For native speech, the [MiniCPM-o 4.5 GPTQ model card](https://huggingface.co/openbmb/MiniCPM-o-4_5-GPTQ) reports 9B parameters, true concurrent listening/speaking, Chinese/English realtime speech, and about 11 GB of single-GPU INT4 memory use. The larger Qwen3-Omni INT4 result in the same published comparison is about 20.3 GB, leaving too little 24 GB headroom for this combined deployment. The server follows vLLM-Omni's official [MiniCPM-o native-duplex serving recipe](https://docs.vllm.ai/projects/vllm-omni/en/v0.26.0/user_guide/examples/online_serving/minicpmo/).

Only `qwen3.5:9b` is downloaded during startup for text generation. In Local Pipeline, select a recommended model and click **設定模型**. The page then:

1. streams the Ollama download progress;
2. refreshes the installed-model catalog;
3. sends an empty Ollama generation request to preload the selected model;
4. keeps the model ready for 5 minutes by default.

The browser setup endpoint accepts only the configured default, the models already installed on the server, and the three 24 GB recommendations listed above. It cannot be used to pull an arbitrary model name. The button is intentionally hidden for Local Full Duplex because its GPTQ speech model is loaded by vLLM-Omni during service startup, not by Ollama.

### Save the System prompt

Edit the System prompt and click **Save Prompt** to store the current text as the default for the current browser and site origin. This does not write a server-wide file and does not let one visitor change another visitor's default. Language and RAG/tool policies continue to be added separately by the application. The Voice selector controls OpenAI TTS/Realtime; Local Full Duplex disables it because MiniCPM-o uses the configured reference WAV instead.

To retain the manual workflow, or when browser model setup is disabled, run:

```bash
docker compose -f docker-compose.local.yml exec ollama ollama pull qwen3:14b
```

Refresh the page after a manual download. The model then appears under **已安裝**. Actual VRAM use also depends on context length, audio-session length and concurrent requests. The Compose defaults cap Ollama at a 4096-token context with one loaded model/one parallel request, and configure the three MiniCPM-o stages for one Full Duplex session with conservative GPU-memory budgets. These limits are deliberate so the 6.6 GB text weights and approximately 11 GB INT4 speech runtime have a practical chance to coexist on a 24 GB card.

If the server has another NVIDIA GPU, isolate the speech model from Ollama for more headroom:

```bash
export LOCAL_DUPLEX_GPU_ID=1
./run.sh --backend local --gpuid 0 --port 7860
```

`--gpuid` selects Ollama's GPU. `LOCAL_DUPLEX_GPU_ID` overrides only MiniCPM-o; when omitted it uses the same GPU. Local Full Duplex requires CUDA and is not supported by the documented CPU-only setup.

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
export LOCAL_DUPLEX_MODEL="openbmb/MiniCPM-o-4_5-GPTQ"
export LOCAL_DUPLEX_WS_URL="ws://127.0.0.1:8099/v1/realtime"
export LOCAL_DUPLEX_REF_AUDIO="/absolute/path/to/reference.wav"
export MANAGE_LOCAL_SERVICES=0
./run.sh --backend local --gpuid 0 --port 7860
```

`MANAGE_LOCAL_SERVICES=0` prevents the startup script from launching Docker or pulling Ollama models and disables the browser **設定模型** button by default. In this mode, the administrator must provide both the OpenAI-compatible text/embedding endpoints and a vLLM-Omni native-duplex WebSocket endpoint. Set `ENABLE_LOCAL_MODEL_SETUP=1` only when the text endpoint is Ollama and its native API is reachable. `.env.example` lists every override.

## RAG file upload

The knowledge card automatically targets the engine selected on the homepage:

- an OpenAI mode uploads to an OpenAI vector store;
- Local Pipeline parses and chunks files on the server, obtains embeddings from the local embedding endpoint, and stores vectors in Qdrant; Local Full Duplex hides this card.

The browser maintains separate signed tokens for the OpenAI and Local knowledge bases across launches. The two OpenAI modes share one cloud knowledge base. The Local knowledge base is used by Local Pipeline; native Local Full Duplex hides the RAG controls because that runtime currently has no compatible retrieval tool protocol. A token from one backend is never sent to the other backend.

Each upload accepts up to 10 files and defaults to 20 MB per file and 50 MB per request. OpenAI RAG supports `.doc` in addition to the formats below. Local RAG supports `.docx`, `.pptx`, `.pdf`, `.txt`, `.md`, `.json`, `.html`, `.c`, `.cpp`, `.cs`, `.css`, `.go`, `.java`, `.js`, `.php`, `.py`, `.rb`, `.sh`, `.tex`, and `.ts`.

Local deletion removes the selected Qdrant collection. Cloud deletion removes the OpenAI vector store and its uploaded OpenAI file objects.

## Optional models and limits

Important defaults are:

- OpenAI text fallback: `gpt-5.6-luna` (the homepage normally selects from the API-provided list)
- Pipeline transcription: `gpt-transcribe`
- OpenAI TTS: `gpt-4o-mini-tts`
- OpenAI Full Duplex fallback: `gpt-realtime-2` (the homepage normally selects from the API-provided list)
- Local Pipeline text LLM: `qwen3.5:9b`
- Local Full Duplex speech LLM: `openbmb/MiniCPM-o-4_5-GPTQ`
- Local embedding: `bge-m3`
- Qdrant: `http://127.0.0.1:6333`

Copy `.env.example` values into your environment to change them. Set stable `RAG_TOKEN_SECRET` and `LOCAL_RAG_TOKEN_SECRET` values in production.

## Public HTTPS URL with Cloudflare Tunnel

After installing `cloudflared` with the commands above, `run.sh` uses `BACKEND` to start the matching Conda environment and always creates a public HTTPS tunnel:

```bash
export OPENAI_API_KEY="sk-..."
./run.sh --backend openai --port 7860
# or
./run.sh --backend local --gpuid 0 --port 7860
```

Copy the random `https://...trycloudflare.com` URL printed by `cloudflared`. Quick Tunnels are intended only for testing.

`run.sh` forces Cloudflare Tunnel to use HTTP/2 over TCP. The server firewall must allow outbound TCP port `7844`; no inbound port needs to be opened.

For a stable hostname, create a remotely managed tunnel and configure its public hostname service to point to `http://127.0.0.1:7860` (or your `PUBLIC_BIND_HOST` and `PORT`). Then set `CLOUDFLARE_TUNNEL_TOKEN` and run `./run.sh`. The script passes the token through `TUNNEL_TOKEN`, so it is not exposed as a command-line argument. Protect any public hostname with Cloudflare Access and application-level authentication/rate limits. Anyone who can reach an unprotected URL can consume OpenAI quota, local GPU time, and storage.

## API summary

- `GET /api/models`: mode-compatible OpenAI choices, installed Local models, and RTX 3090 recommendations.
- `POST /api/local/models/setup`: stream an approved Ollama model download and preload it for Local inference.
- `POST /api/pipeline/turn`: OpenAI Pipeline turn.
- `POST /api/realtime/session`: OpenAI end-to-end Realtime WebRTC SDP exchange.
- `POST /api/local/pipeline/turn`: OpenAI ASR/TTS with local RAG and LLM.
- `WS /api/local/duplex`: restricted same-origin PCM16 proxy to native MiniCPM-o Full Duplex.
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
