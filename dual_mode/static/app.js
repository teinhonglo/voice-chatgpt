const LANGUAGES = [
  ["zh-TW", "繁體中文（台灣）"],
  ["zh-CN", "简体中文（中国）"],
  ["zh-HK", "粵語（香港）"],
  ["en", "English"],
  ["ja", "日本語"],
  ["ko", "한국어"],
  ["es", "Español"],
  ["fr", "Français"],
  ["de", "Deutsch"],
  ["it", "Italiano"],
  ["pt", "Português"],
  ["vi", "Tiếng Việt"],
  ["th", "ไทย"],
  ["id", "Bahasa Indonesia"],
  ["hi", "हिन्दी"],
  ["ar", "العربية"],
  ["ru", "Русский"],
];

const VOICES = ["marin", "cedar", "coral", "alloy", "ash", "ballad", "echo", "sage", "shimmer", "verse"];
const STORAGE_KEY = "voice-chatgpt-dual-mode-settings";
const CLOUD_RAG_STORAGE_KEY = "voice-chatgpt-dual-mode-rag";
const LOCAL_RAG_STORAGE_KEY = "voice-chatgpt-local-rag";
const RAG_TOOL_NAME = "search_knowledge_base";
const LEGACY_DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant. Answer accurately, naturally, and concisely.";

const el = {
  appTitle: document.querySelector("#app-title"),
  modeSwitch: document.querySelector("#mode-switch"),
  modeButtons: [...document.querySelectorAll(".mode-button")],
  systemPrompt: document.querySelector("#system-prompt"),
  llmModel: document.querySelector("#llm-model"),
  modelHelp: document.querySelector("#model-help"),
  languageA: document.querySelector("#language-a"),
  languageB: document.querySelector("#language-b"),
  voice: document.querySelector("#voice"),
  savedIndicator: document.querySelector("#saved-indicator"),
  ragFiles: document.querySelector("#rag-files"),
  ragUploadButton: document.querySelector("#rag-upload-button"),
  ragDeleteButton: document.querySelector("#rag-delete-button"),
  ragStatus: document.querySelector("#rag-status"),
  ragFileList: document.querySelector("#rag-file-list"),
  ragHelp: document.querySelector("#rag-help"),
  privacyNote: document.querySelector("#privacy-note"),
  status: document.querySelector("#status-text"),
  stateDot: document.querySelector(".state-dot"),
  modeDescription: document.querySelector("#mode-description"),
  chatLog: document.querySelector("#chat-log"),
  emptyState: document.querySelector("#empty-state"),
  liveCaption: document.querySelector("#live-caption"),
  pipelineControls: document.querySelector("#pipeline-controls"),
  realtimeControls: document.querySelector("#realtime-controls"),
  recordButton: document.querySelector("#record-button"),
  recordButtonLabel: document.querySelector("#record-button-label"),
  callButton: document.querySelector("#call-button"),
  callButtonLabel: document.querySelector("#call-button-label"),
  muteButton: document.querySelector("#mute-button"),
  clearButton: document.querySelector("#clear-button"),
  clearRealtimeButton: document.querySelector("#clear-realtime-button"),
  remoteAudio: document.querySelector("#remote-audio"),
  pipelineAudio: document.querySelector("#pipeline-audio"),
};

const state = {
  backend: "openai",
  mode: "pipeline",
  recorder: null,
  recorderStream: null,
  chunks: [],
  discardRecording: false,
  pipelineHistory: [],
  pc: null,
  dataChannel: null,
  micStream: null,
  muted: false,
  inputTranscript: "",
  outputTranscript: "",
  modelCatalog: null,
  modelSelections: { text: "", realtime: "", local: "" },
  knowledge: {
    cloud: { token: "", files: [] },
    local: { token: "", files: [] },
  },
  localTurnController: null,
  localAudioQueue: [],
  localAudioObjectUrl: "",
};

function isLocalMode() {
  return state.mode.startsWith("local-");
}

function isRealtimeMode() {
  return state.mode === "realtime" || state.mode === "local-realtime";
}

function activeKnowledge() {
  return state.knowledge[isLocalMode() ? "local" : "cloud"];
}

function activeModelKind() {
  if (isLocalMode()) return "local";
  return isRealtimeMode() ? "realtime" : "text";
}

function populateSelects() {
  for (const [code, label] of LANGUAGES) {
    el.languageA.add(new Option(`${label} · ${code}`, code));
    el.languageB.add(new Option(`${label} · ${code}`, code));
  }
  for (const voice of VOICES) {
    const label = voice.charAt(0).toUpperCase() + voice.slice(1);
    el.voice.add(new Option(label, voice));
  }
}

function loadSettings() {
  const defaults = {
    systemPrompt: el.systemPrompt.value.trim(),
    languageA: "zh-TW",
    languageB: "en",
    voice: "marin",
    models: { text: "", realtime: "", local: "" },
  };
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    if (saved.systemPrompt === LEGACY_DEFAULT_SYSTEM_PROMPT) delete saved.systemPrompt;
    const savedModels = saved.models && typeof saved.models === "object" ? saved.models : {};
    return { ...defaults, ...saved, models: { ...defaults.models, ...savedModels } };
  } catch {
    return defaults;
  }
}

function currentSettings() {
  return {
    systemPrompt: el.systemPrompt.value.trim(),
    languageA: el.languageA.value,
    languageB: el.languageB.value,
    voice: el.voice.value,
    models: { ...state.modelSelections },
  };
}

function localRecommendation(modelId) {
  return state.modelCatalog?.local?.recommended?.find((item) => item.id === modelId) || null;
}

function renderModelHelp() {
  if (!state.modelCatalog) return;
  const kind = activeModelKind();
  const selected = el.llmModel.value;
  if (kind !== "local") {
    const warning = state.modelCatalog.warnings?.find((message) => message.startsWith("OpenAI"));
    el.modelHelp.textContent = warning
      ? "無法自動取得 OpenAI 清單，目前使用後端預設模型。"
      : kind === "realtime"
        ? "已依 OPENAI_API_KEY 自動載入 Realtime 模型；變更後請重新建立 Full Duplex 連線。"
        : "已依 OPENAI_API_KEY 自動載入可用的 Responses 文字模型。";
    return;
  }

  const installed = new Set(state.modelCatalog.local?.installed || []);
  const recommendation = localRecommendation(selected);
  if (installed.has(selected)) {
    el.modelHelp.textContent = recommendation
      ? `已安裝 · ${recommendation.label} · ${recommendation.size} · ${recommendation.note}`
      : "此模型已安裝在目前的 Local LLM 服務。";
    return;
  }
  el.modelHelp.textContent = `尚未安裝。請先在伺服器執行：docker compose -f docker-compose.local.yml exec ollama ollama pull ${selected}`;
}

function renderModelSelect() {
  if (!state.modelCatalog) return;
  const kind = activeModelKind();
  const defaults = state.modelCatalog.defaults || {};
  el.llmModel.replaceChildren();

  if (kind === "local") {
    const installed = [...new Set(state.modelCatalog.local?.installed || [])];
    const recommendations = state.modelCatalog.local?.recommended || [];
    const installedGroup = document.createElement("optgroup");
    installedGroup.label = "已安裝";
    for (const modelId of installed) {
      const recommendation = recommendations.find((item) => item.id === modelId);
      installedGroup.append(new Option(
        recommendation ? `${recommendation.label} · ${modelId}` : modelId,
        modelId,
      ));
    }
    if (installed.length) el.llmModel.append(installedGroup);

    const recommendedGroup = document.createElement("optgroup");
    recommendedGroup.label = "RTX 3090 推薦（需先下載）";
    for (const item of recommendations) {
      if (installed.includes(item.id)) continue;
      recommendedGroup.append(new Option(`${item.label} · ${item.size} · ${item.note}`, item.id));
    }
    if (recommendedGroup.children.length) el.llmModel.append(recommendedGroup);
  } else {
    const models = state.modelCatalog.openai?.[kind] || [];
    for (const modelId of models) el.llmModel.add(new Option(modelId, modelId));
  }

  const desired = state.modelSelections[kind] || defaults[kind] || "";
  const available = [...el.llmModel.options].map((option) => option.value);
  if (desired && !available.includes(desired)) {
    const configuredGroup = document.createElement("optgroup");
    configuredGroup.label = "目前設定";
    configuredGroup.append(new Option(desired, desired));
    el.llmModel.prepend(configuredGroup);
  }
  el.llmModel.value = desired || el.llmModel.options[0]?.value || "";
  state.modelSelections[kind] = el.llmModel.value;
  el.llmModel.disabled = !el.llmModel.value;
  renderModelHelp();
}

async function loadModelCatalog() {
  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(errorDetail(payload, `HTTP ${response.status}`));
    state.modelCatalog = payload;
  } catch (error) {
    state.modelCatalog = {
      backend: "openai",
      modes: ["pipeline", "realtime"],
      defaults: { text: "gpt-5.6-luna", realtime: "gpt-realtime-2", local: "qwen3:8b" },
      openai: { text: ["gpt-5.6-luna"], realtime: ["gpt-realtime-2"] },
      local: { installed: [], recommended: [{ id: "qwen3:8b", label: "Qwen 3 8B", size: "5.2 GB", note: "預設" }] },
      warnings: ["OpenAI model list could not be loaded; using the configured defaults."],
    };
    el.modelHelp.textContent = `模型清單載入失敗：${error.message}`;
  }
  applyDeploymentBackend(state.modelCatalog.backend, state.modelCatalog.modes);
  renderModelSelect();
}

function applyDeploymentBackend(backend, modes) {
  state.backend = backend === "local" ? "local" : "openai";
  const allowedModes = new Set(Array.isArray(modes) && modes.length
    ? modes
    : state.backend === "local" ? ["local-pipeline", "local-realtime"] : ["pipeline", "realtime"]);
  for (const button of el.modeButtons) {
    button.classList.toggle("hidden", !allowedModes.has(button.dataset.mode));
  }
  el.modeSwitch.classList.remove("hidden");
  el.appTitle.textContent = state.backend === "local" ? "Local LLM + RAG 語音聊天" : "OpenAI 語音聊天";
  document.title = state.backend === "local" ? "Local LLM + RAG 語音聊天" : "OpenAI 語音聊天";
  void setMode(state.backend === "local" ? "local-pipeline" : "pipeline", true);
}

function loadKnowledgeBase() {
  for (const [engine, storageKey] of [["cloud", CLOUD_RAG_STORAGE_KEY], ["local", LOCAL_RAG_STORAGE_KEY]]) {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
      if (typeof saved.token !== "string" || !Array.isArray(saved.files)) continue;
      state.knowledge[engine].token = saved.token;
      state.knowledge[engine].files = saved.files.filter((name) => typeof name === "string").slice(0, 100);
    } catch {
      localStorage.removeItem(storageKey);
    }
  }
}

function saveKnowledgeBase(engine = isLocalMode() ? "local" : "cloud") {
  const storageKey = engine === "local" ? LOCAL_RAG_STORAGE_KEY : CLOUD_RAG_STORAGE_KEY;
  const knowledge = state.knowledge[engine];
  if (!knowledge.token) {
    localStorage.removeItem(storageKey);
    return;
  }
  localStorage.setItem(
    storageKey,
    JSON.stringify({ token: knowledge.token, files: knowledge.files }),
  );
}

function setRagStatus(message, kind = "idle") {
  el.ragStatus.textContent = message;
  el.ragStatus.className = `knowledge-state ${kind}`;
}

function renderKnowledgeBase() {
  const local = isLocalMode();
  const knowledge = activeKnowledge();
  el.ragFileList.replaceChildren();
  for (const filename of knowledge.files) {
    const chip = document.createElement("span");
    chip.className = "file-chip";
    chip.textContent = filename;
    chip.title = filename;
    el.ragFileList.append(chip);
  }
  el.ragFileList.classList.toggle("hidden", knowledge.files.length === 0);
  el.ragDeleteButton.disabled = !knowledge.token;
  setRagStatus(
    knowledge.token
      ? `${local ? "Local / Qdrant" : "OpenAI"} · 已啟用 · ${knowledge.files.length} 個檔案`
      : `${local ? "Local / Qdrant" : "OpenAI"} · 尚未上傳檔案`,
    knowledge.token ? "ready" : "idle",
  );
  el.ragHelp.textContent = local
    ? "目前為地端知識庫：檔案在本機解析，embedding 與 Qdrant 都走地端；兩個 Local 模式共用。支援 PDF、DOCX、PPTX、文字與常見程式碼格式。"
    : "目前為 OpenAI 知識庫：完成索引後，兩個 OpenAI 模式會共用相同檔案。";
}

function setRagControlsBusy(busy) {
  el.modeButtons.forEach((button) => { button.disabled = busy; });
  el.ragFiles.disabled = busy;
  el.ragUploadButton.disabled = busy;
  el.ragDeleteButton.disabled = busy || !activeKnowledge().token;
  el.recordButton.disabled = busy;
  el.callButton.disabled = busy;
}

let saveTimer;
function saveSettings() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(currentSettings()));
  el.savedIndicator.textContent = "已儲存";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    el.savedIndicator.textContent = "自動儲存";
  }, 1200);
}

function setStatus(message, kind = "idle") {
  el.status.textContent = message;
  el.stateDot.className = `state-dot ${kind}`;
}

function addMessage(role, text) {
  el.emptyState?.remove();
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "YOU" : "AI";
  const content = document.createElement("p");
  content.textContent = text;
  wrapper.append(label, content);
  el.chatLog.append(wrapper);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
}

function clearConversation() {
  stopLocalTurn();
  state.pipelineHistory = [];
  el.chatLog.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.id = "empty-state";
  empty.innerHTML = '<span class="wave-icon">⌁</span><p>你的對話會顯示在這裡</p>';
  el.chatLog.append(empty);
  el.emptyState = empty;
  el.liveCaption.classList.add("hidden");
}

function appendSettings(formData) {
  const settings = currentSettings();
  formData.append("system_prompt", settings.systemPrompt);
  formData.append("language_a", settings.languageA);
  formData.append("language_b", settings.languageB);
  formData.append("voice", settings.voice);
  if (el.llmModel.value) formData.append("llm_model", el.llmModel.value);
  const knowledge = activeKnowledge();
  if (knowledge.token) formData.append("rag_token", knowledge.token);
}

function errorDetail(payload, fallback) {
  if (payload && typeof payload.detail === "string") return payload.detail;
  return fallback;
}

async function setMode(mode, force = false) {
  if (state.modelCatalog?.modes?.length && !state.modelCatalog.modes.includes(mode)) return;
  if (mode === state.mode && !force) return;
  if (state.pc) stopRealtime();
  stopLocalTurn();
  if (state.recorder && state.recorder.state === "recording") {
    state.discardRecording = true;
    state.recorder.stop();
  }

  state.mode = mode;
  el.modeButtons.forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  const realtime = isRealtimeMode();
  const local = isLocalMode();
  el.pipelineControls.classList.toggle("hidden", realtime);
  el.realtimeControls.classList.toggle("hidden", !realtime);
  const descriptions = {
    pipeline: "OpenAI 依序完成語音辨識、Responses 回覆與語音合成，適合保留清楚的逐輪紀錄。",
    realtime: "OpenAI Realtime 端到端雙向串流，可自然插話並立即打斷 AI。",
    "local-pipeline": "OpenAI 負責 ASR/TTS；RAG、embedding、Qdrant 與 LLM 回覆都在地端執行。",
    "local-realtime": "OpenAI 串流 ASR/TTS + 地端 RAG/LLM。麥克風持續開啟，開口即可中止播放與尚未完成的生成。",
  };
  el.modeDescription.textContent = realtime
    ? `${descriptions[mode]} 連線期間修改設定時，請重新連線套用。`
    : descriptions[mode];
  el.privacyNote.textContent = local
    ? "語音由 AI 生成。檔案、RAG、embedding 與 LLM 留在地端；麥克風音訊會送至 OpenAI ASR，回答文字會送至 OpenAI TTS。"
    : "語音由 AI 生成。OpenAI 模式會將上傳檔案交由 OpenAI 建立檢索索引；請勿上傳未經授權的機密資料。";
  renderModelSelect();
  renderKnowledgeBase();
  el.callButtonLabel.textContent = local ? "開始 Local Full Duplex" : "開始 Full Duplex";
  setStatus(`${local ? "Local " : ""}${realtime ? "Full Duplex" : "Pipeline"} 待命`);
}

async function uploadKnowledgeFiles() {
  const files = [...el.ragFiles.files];
  if (!files.length) {
    setRagStatus("請先選擇檔案", "error");
    return;
  }
  if (state.pc) stopRealtime();
  if (state.recorder?.state === "recording") {
    state.discardRecording = true;
    state.recorder.stop();
  }
  setRagControlsBusy(true);
  setRagStatus("正在上傳並建立索引…");

  const engine = isLocalMode() ? "local" : "cloud";
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file, file.name));
  const knowledge = state.knowledge[engine];
  if (knowledge.token) formData.append("rag_token", knowledge.token);

  try {
    const endpoint = engine === "local" ? "/api/local/rag/upload" : "/api/rag/upload";
    const response = await fetch(endpoint, { method: "POST", body: formData });
    let payload;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) throw new Error(errorDetail(payload, `HTTP ${response.status}`));

    knowledge.token = payload.rag_token;
    knowledge.files = [...knowledge.files, ...payload.files];
    saveKnowledgeBase(engine);
    el.ragFiles.value = "";
    renderKnowledgeBase();
  } catch (error) {
    setRagStatus(`上傳失敗：${error.message}`, "error");
  } finally {
    setRagControlsBusy(false);
  }
}

async function deleteKnowledgeBase() {
  const engine = isLocalMode() ? "local" : "cloud";
  const knowledge = state.knowledge[engine];
  if (!knowledge.token) return;
  const target = engine === "local" ? "本機 Qdrant 知識庫" : "這個知識庫及其 OpenAI 檔案";
  if (!window.confirm(`要刪除${target}嗎？`)) return;
  if (state.pc) stopRealtime();
  if (state.recorder?.state === "recording") {
    state.discardRecording = true;
    state.recorder.stop();
  }
  setRagControlsBusy(true);
  setRagStatus("正在刪除知識庫…");

  const formData = new FormData();
  formData.append("rag_token", knowledge.token);
  try {
    const endpoint = engine === "local" ? "/api/local/rag/delete" : "/api/rag/delete";
    const response = await fetch(endpoint, { method: "POST", body: formData });
    let payload;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) throw new Error(errorDetail(payload, `HTTP ${response.status}`));
    knowledge.token = "";
    knowledge.files = [];
    saveKnowledgeBase(engine);
    renderKnowledgeBase();
  } catch (error) {
    setRagStatus(`刪除失敗：${error.message}`, "error");
  } finally {
    setRagControlsBusy(false);
  }
}

function preferredMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

async function startRecording() {
  try {
    state.recorderStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = preferredMimeType();
    state.recorder = new MediaRecorder(state.recorderStream, mimeType ? { mimeType } : undefined);
    state.chunks = [];
    state.discardRecording = false;
    state.recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) state.chunks.push(event.data);
    });
    state.recorder.addEventListener("stop", async () => {
      state.recorderStream?.getTracks().forEach((track) => track.stop());
      state.recorderStream = null;
      el.recordButton.classList.remove("recording");
      el.recordButtonLabel.textContent = "開始錄音";
      if (state.discardRecording) return;
      const blob = new Blob(state.chunks, { type: state.recorder.mimeType || "audio/webm" });
      if (blob.size) await sendPipelineTurn(blob);
    });
    state.recorder.start();
    el.recordButton.classList.add("recording");
    el.recordButtonLabel.textContent = "停止並送出";
    setStatus("正在錄音…", "live");
  } catch (error) {
    setStatus(`無法使用麥克風：${error.message}`, "error");
  }
}

function stopRecording() {
  if (state.recorder?.state === "recording") state.recorder.stop();
}

async function sendPipelineTurn(blob) {
  el.modeButtons.forEach((button) => { button.disabled = true; });
  el.recordButton.disabled = true;
  el.ragFiles.disabled = true;
  el.ragUploadButton.disabled = true;
  el.ragDeleteButton.disabled = true;
  const local = isLocalMode();
  setStatus(local ? "地端 RAG / LLM 正在處理…" : "OpenAI 正在處理…", "busy");
  const extension = blob.type.includes("mp4") ? "m4a" : "webm";
  const formData = new FormData();
  formData.append("audio", blob, `recording.${extension}`);
  formData.append("history_json", JSON.stringify(state.pipelineHistory));
  appendSettings(formData);

  try {
    const endpoint = local ? "/api/local/pipeline/turn" : "/api/pipeline/turn";
    const response = await fetch(endpoint, { method: "POST", body: formData });
    let payload;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) throw new Error(errorDetail(payload, `HTTP ${response.status}`));

    addMessage("user", payload.transcript);
    addMessage("assistant", payload.reply);
    state.pipelineHistory.push(
      { role: "user", content: payload.transcript },
      { role: "assistant", content: payload.reply },
    );
    state.pipelineHistory = state.pipelineHistory.slice(-20);
    el.pipelineAudio.src = `data:${payload.audio_mime};base64,${payload.audio_base64}`;
    await el.pipelineAudio.play().catch(() => {});
    setStatus(local ? "Local Pipeline 待命" : "Pipeline 待命", "ready");
  } catch (error) {
    setStatus(`處理失敗：${error.message}`, "error");
  } finally {
    el.modeButtons.forEach((button) => { button.disabled = false; });
    el.recordButton.disabled = false;
    el.ragFiles.disabled = false;
    el.ragUploadButton.disabled = false;
    el.ragDeleteButton.disabled = !activeKnowledge().token;
  }
}

function showLiveCaption(role, text) {
  if (!text) {
    el.liveCaption.classList.add("hidden");
    return;
  }
  el.liveCaption.textContent = `${role === "user" ? "YOU" : "AI"} · ${text}`;
  el.liveCaption.classList.remove("hidden");
}

async function runRealtimeKnowledgeTools(toolCalls) {
  const knowledge = state.knowledge.cloud;
  setStatus("正在查詢知識庫…", "busy");
  for (const call of toolCalls) {
    let output;
    try {
      const args = JSON.parse(call.arguments || "{}");
      const query = typeof args.query === "string" ? args.query.trim() : "";
      if (!query) throw new Error("Realtime 未提供有效的檢索問題");
      if (!knowledge.token) throw new Error("知識庫已停用，請重新連線");

      const formData = new FormData();
      formData.append("query", query);
      formData.append("rag_token", knowledge.token);
      const response = await fetch("/api/rag/search", { method: "POST", body: formData });
      let payload;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }
      if (!response.ok) throw new Error(errorDetail(payload, `HTTP ${response.status}`));
      output = JSON.stringify(payload);
    } catch (error) {
      output = JSON.stringify({ error: error.message, results: [] });
      setRagStatus(`檢索失敗：${error.message}`, "error");
    }

    if (state.dataChannel?.readyState !== "open") return;
    state.dataChannel.send(JSON.stringify({
      type: "conversation.item.create",
      item: {
        type: "function_call_output",
        call_id: call.call_id,
        output,
      },
    }));
  }

  if (state.dataChannel?.readyState === "open") {
    state.dataChannel.send(JSON.stringify({ type: "response.create" }));
    if (knowledge.token) {
      setRagStatus(`OpenAI · 已啟用 · ${knowledge.files.length} 個檔案`, "ready");
    }
  }
}

function stopLocalTurn() {
  state.localTurnController?.abort();
  state.localTurnController = null;
  state.localAudioQueue = [];
  el.pipelineAudio.pause();
  el.pipelineAudio.removeAttribute("src");
  el.pipelineAudio.load();
  if (state.localAudioObjectUrl) URL.revokeObjectURL(state.localAudioObjectUrl);
  state.localAudioObjectUrl = "";
  state.outputTranscript = "";
  showLiveCaption("assistant", "");
}

function playNextLocalAudio() {
  if (state.localAudioObjectUrl || !state.localAudioQueue.length) return;
  const blob = state.localAudioQueue.shift();
  state.localAudioObjectUrl = URL.createObjectURL(blob);
  el.pipelineAudio.src = state.localAudioObjectUrl;
  el.pipelineAudio.onended = () => {
    URL.revokeObjectURL(state.localAudioObjectUrl);
    state.localAudioObjectUrl = "";
    playNextLocalAudio();
  };
  void el.pipelineAudio.play().catch(() => {});
}

function queueLocalAudio(audioBase64, mimeType) {
  const binary = atob(audioBase64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  state.localAudioQueue.push(new Blob([bytes], { type: mimeType || "audio/mpeg" }));
  playNextLocalAudio();
}

async function runLocalDuplexTurn(transcript) {
  stopLocalTurn();
  const controller = new AbortController();
  state.localTurnController = controller;
  state.outputTranscript = "";
  setStatus("地端 RAG / LLM 正在回答…", "busy");

  const formData = new FormData();
  formData.append("transcript", transcript);
  formData.append("history_json", JSON.stringify(state.pipelineHistory));
  appendSettings(formData);

  try {
    const response = await fetch("/api/local/realtime/turn", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    if (!response.ok) {
      let payload;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }
      throw new Error(errorDetail(payload, `HTTP ${response.status}`));
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalReply = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "text_delta") {
          state.outputTranscript += event.delta || "";
          showLiveCaption("assistant", state.outputTranscript);
        } else if (event.type === "audio") {
          queueLocalAudio(event.audio_base64, event.audio_mime);
        } else if (event.type === "done") {
          finalReply = (event.reply || state.outputTranscript).trim();
        } else if (event.type === "error") {
          throw new Error(event.message || "Local Full Duplex failed");
        }
      }
      if (done) break;
    }
    if (!finalReply) finalReply = state.outputTranscript.trim();
    if (finalReply) {
      addMessage("assistant", finalReply);
      state.pipelineHistory.push(
        { role: "user", content: transcript },
        { role: "assistant", content: finalReply },
      );
      state.pipelineHistory = state.pipelineHistory.slice(-20);
    }
    state.outputTranscript = "";
    showLiveCaption("assistant", "");
    setStatus("Local Full Duplex 已連線，可直接說話", "ready");
  } catch (error) {
    if (error.name !== "AbortError") setStatus(`地端回答失敗：${error.message}`, "error");
  } finally {
    if (state.localTurnController === controller) state.localTurnController = null;
  }
}

async function handleRealtimeEvent(event) {
  const type = event.type || "";
  if (type === "conversation.item.input_audio_transcription.delta") {
    state.inputTranscript += event.delta || "";
    showLiveCaption("user", state.inputTranscript);
    return;
  }
  if (type === "conversation.item.input_audio_transcription.completed") {
    const transcript = (event.transcript || state.inputTranscript).trim();
    if (transcript) {
      addMessage("user", transcript);
      if (state.mode === "local-realtime") void runLocalDuplexTurn(transcript);
    }
    state.inputTranscript = "";
    showLiveCaption("user", "");
    return;
  }
  if (type === "response.output_audio_transcript.delta" || type === "response.audio_transcript.delta") {
    state.outputTranscript += event.delta || "";
    showLiveCaption("assistant", state.outputTranscript);
    return;
  }
  if (type === "response.output_audio_transcript.done" || type === "response.audio_transcript.done") {
    const transcript = (event.transcript || state.outputTranscript).trim();
    if (transcript) addMessage("assistant", transcript);
    state.outputTranscript = "";
    showLiveCaption("assistant", "");
    return;
  }
  if (type === "input_audio_buffer.speech_started") {
    if (state.mode === "local-realtime") stopLocalTurn();
    setStatus("正在聆聽…", "live");
    return;
  }
  if (type === "response.created") {
    setStatus("AI 正在回答…", "busy");
    return;
  }
  if (type === "response.done") {
    const toolCalls = (event.response?.output || []).filter(
      (item) => item.type === "function_call" && item.name === RAG_TOOL_NAME,
    );
    if (toolCalls.length) {
      await runRealtimeKnowledgeTools(toolCalls);
      return;
    }
    setStatus("已連線，可直接說話", "ready");
    return;
  }
  if (type === "error") {
    setStatus(`Realtime 錯誤：${event.error?.message || "未知錯誤"}`, "error");
  }
}

async function startRealtime() {
  const local = state.mode === "local-realtime";
  el.callButton.disabled = true;
  el.modeButtons.forEach((button) => { button.disabled = true; });
  setStatus("正在建立安全連線…", "busy");
  try {
    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    state.pc = new RTCPeerConnection();
    state.pc.addEventListener("track", (event) => {
      el.remoteAudio.srcObject = event.streams[0] || new MediaStream([event.track]);
    });
    state.pc.addEventListener("connectionstatechange", () => {
      if (state.pc?.connectionState === "failed") {
        setStatus("Realtime 連線失敗", "error");
        stopRealtime();
      }
    });
    state.micStream.getTracks().forEach((track) => state.pc.addTrack(track, state.micStream));

    state.dataChannel = state.pc.createDataChannel("oai-events");
    state.dataChannel.addEventListener("open", () => setStatus(
      local ? "Local Full Duplex 已連線，可直接說話" : "已連線，可直接說話",
      "ready",
    ));
    state.dataChannel.addEventListener("message", (message) => {
      try {
        void handleRealtimeEvent(JSON.parse(message.data));
      } catch {
        // Ignore non-JSON diagnostic messages.
      }
    });

    const offer = await state.pc.createOffer();
    await state.pc.setLocalDescription(offer);
    const formData = new FormData();
    formData.append("sdp", offer.sdp);
    appendSettings(formData);
    const endpoint = local ? "/api/local/realtime/session" : "/api/realtime/session";
    const response = await fetch(endpoint, { method: "POST", body: formData });
    if (!response.ok) {
      let payload;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }
      throw new Error(errorDetail(payload, `HTTP ${response.status}`));
    }
    await state.pc.setRemoteDescription({ type: "answer", sdp: await response.text() });

    el.callButton.classList.add("connected");
    el.callButtonLabel.textContent = local ? "結束 Local Full Duplex" : "結束 Full Duplex";
    el.muteButton.disabled = false;
  } catch (error) {
    stopRealtime(false);
    setStatus(`連線失敗：${error.message}`, "error");
  } finally {
    el.callButton.disabled = false;
    el.modeButtons.forEach((button) => { button.disabled = false; });
  }
}

function stopRealtime(updateStatus = true) {
  stopLocalTurn();
  state.dataChannel?.close();
  state.pc?.close();
  state.micStream?.getTracks().forEach((track) => track.stop());
  state.dataChannel = null;
  state.pc = null;
  state.micStream = null;
  state.muted = false;
  state.inputTranscript = "";
  state.outputTranscript = "";
  el.remoteAudio.srcObject = null;
  el.callButton.classList.remove("connected");
  el.callButtonLabel.textContent = state.mode === "local-realtime" ? "開始 Local Full Duplex" : "開始 Full Duplex";
  el.muteButton.textContent = "靜音";
  el.muteButton.disabled = true;
  el.liveCaption.classList.add("hidden");
  if (updateStatus) setStatus(state.mode === "local-realtime" ? "Local Full Duplex 待命" : "Full Duplex 待命");
}

function toggleMute() {
  state.muted = !state.muted;
  state.micStream?.getAudioTracks().forEach((track) => {
    track.enabled = !state.muted;
  });
  el.muteButton.textContent = state.muted ? "取消靜音" : "靜音";
  const ready = state.mode === "local-realtime" ? "Local Full Duplex 已連線，可直接說話" : "已連線，可直接說話";
  setStatus(state.muted ? "麥克風已靜音" : ready, state.muted ? "idle" : "ready");
}

function initialize() {
  populateSelects();
  const settings = loadSettings();
  el.systemPrompt.value = settings.systemPrompt || el.systemPrompt.value;
  el.languageA.value = settings.languageA;
  el.languageB.value = settings.languageB;
  el.voice.value = settings.voice;
  state.modelSelections = { ...state.modelSelections, ...settings.models };
  void loadModelCatalog();
  loadKnowledgeBase();
  renderKnowledgeBase();

  el.modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  [el.systemPrompt, el.languageA, el.languageB, el.voice].forEach((input) => input.addEventListener("change", saveSettings));
  el.llmModel.addEventListener("change", () => {
    state.modelSelections[activeModelKind()] = el.llmModel.value;
    renderModelHelp();
    saveSettings();
  });
  el.systemPrompt.addEventListener("input", saveSettings);
  el.recordButton.addEventListener("click", () => {
    if (state.recorder?.state === "recording") stopRecording();
    else startRecording();
  });
  el.callButton.addEventListener("click", () => {
    if (state.pc) stopRealtime();
    else startRealtime();
  });
  el.muteButton.addEventListener("click", toggleMute);
  el.ragUploadButton.addEventListener("click", uploadKnowledgeFiles);
  el.ragDeleteButton.addEventListener("click", deleteKnowledgeBase);
  el.ragFiles.addEventListener("change", () => {
    const count = el.ragFiles.files.length;
    if (count) setRagStatus(`已選擇 ${count} 個檔案，尚未上傳`);
    else renderKnowledgeBase();
  });
  el.clearButton.addEventListener("click", clearConversation);
  el.clearRealtimeButton.addEventListener("click", clearConversation);
  window.addEventListener("beforeunload", () => {
    state.discardRecording = true;
    if (state.recorder?.state === "recording") state.recorder.stop();
    stopRealtime(false);
  });
}

initialize();
