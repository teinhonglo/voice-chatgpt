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
const PROMPT_DEFAULT_STORAGE_KEY = "voice-chatgpt-default-system-prompt";
const CLOUD_RAG_STORAGE_KEY = "voice-chatgpt-dual-mode-rag";
const LOCAL_RAG_STORAGE_KEY = "voice-chatgpt-local-rag";
const RAG_TOOL_NAME = "search_knowledge_base";
const LOCAL_DUPLEX_LANGUAGES = new Set(["zh-TW", "zh-CN", "en"]);
const LOCAL_DUPLEX_INPUT_RATE = 16000;
const LOCAL_DUPLEX_OUTPUT_RATE = 24000;
const LEGACY_DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant. Answer accurately, naturally, and concisely.";

const el = {
  appTitle: document.querySelector("#app-title"),
  modeSwitch: document.querySelector("#mode-switch"),
  modeButtons: [...document.querySelectorAll(".mode-button")],
  systemPrompt: document.querySelector("#system-prompt"),
  promptSaveButton: document.querySelector("#prompt-save-button"),
  promptHelp: document.querySelector("#prompt-help"),
  llmModel: document.querySelector("#llm-model"),
  modelSetupButton: document.querySelector("#model-setup-button"),
  modelProgress: document.querySelector("#model-progress"),
  modelProgressBar: document.querySelector("#model-progress-bar"),
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
  knowledgeCard: document.querySelector("#knowledge-card"),
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
  modelSelections: { text: "", realtime: "", local_text: "", local_duplex: "" },
  modelSetupInProgress: false,
  promptDirty: false,
  savedPrompt: "",
  knowledge: {
    cloud: { token: "", files: [] },
    local: { token: "", files: [] },
  },
  localDuplexSocket: null,
  localDuplexInputContext: null,
  localDuplexOutputContext: null,
  localDuplexSource: null,
  localDuplexProcessor: null,
  localDuplexGain: null,
  localDuplexInputSamples: [],
  localDuplexAudioEndMs: 0,
  localDuplexNextPlaybackTime: 0,
  localDuplexPlaybackSources: new Set(),
  localDuplexCancelledSources: new Set(),
  localDuplexPlayedMs: new Map(),
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
  if (state.mode === "local-pipeline") return "local_text";
  if (state.mode === "local-realtime") return "local_duplex";
  return isRealtimeMode() ? "realtime" : "text";
}

function realtimeActive() {
  return Boolean(state.pc || state.localDuplexSocket);
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
    models: { text: "", realtime: "", local_text: "", local_duplex: "" },
  };
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    if (saved.systemPrompt === LEGACY_DEFAULT_SYSTEM_PROMPT) delete saved.systemPrompt;
    let savedPrompt = (localStorage.getItem(PROMPT_DEFAULT_STORAGE_KEY) || "").trim();
    if (!savedPrompt && typeof saved.systemPrompt === "string" && saved.systemPrompt.trim()) {
      savedPrompt = saved.systemPrompt.trim();
      localStorage.setItem(PROMPT_DEFAULT_STORAGE_KEY, savedPrompt);
    }
    delete saved.systemPrompt;
    const savedModels = saved.models && typeof saved.models === "object" ? saved.models : {};
    if (savedModels.local && !savedModels.local_text) savedModels.local_text = savedModels.local;
    delete savedModels.local;
    return {
      ...defaults,
      ...saved,
      systemPrompt: savedPrompt || defaults.systemPrompt,
      models: { ...defaults.models, ...savedModels },
    };
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

function setPromptHelp(message, kind = "idle") {
  el.promptHelp.textContent = message;
  el.promptHelp.className = `prompt-help ${kind}`;
}

function markPromptDirty(message = "Prompt 已修改但尚未存成預設值。") {
  state.promptDirty = true;
  clearTimeout(saveTimer);
  el.savedIndicator.textContent = "Prompt 尚未儲存";
  setPromptHelp(message);
}

function savePromptDefault() {
  const prompt = el.systemPrompt.value.trim();
  if (!prompt) {
    setPromptHelp("System Prompt 不可為空白。", "error");
    return;
  }
  if (prompt.length > 12_000) {
    setPromptHelp("System Prompt 不可超過 12,000 個字元。", "error");
    return;
  }

  el.systemPrompt.value = prompt;
  state.savedPrompt = prompt;
  state.promptDirty = false;
  localStorage.setItem(PROMPT_DEFAULT_STORAGE_KEY, prompt);
  saveSettings();
  el.savedIndicator.textContent = "Prompt 已設為預設";
  setPromptHelp("目前內容已存成這個瀏覽器的預設 Prompt。", "success");
}

function localRecommendation(modelId) {
  return state.modelCatalog?.local?.recommended?.find((item) => item.id === modelId) || null;
}

function renderModelSetupControls() {
  const visible = activeModelKind() === "local_text" && Boolean(state.modelCatalog?.local?.setup_enabled);
  el.modelSetupButton.classList.toggle("hidden", !visible);
  el.modelSetupButton.disabled = !visible || !el.llmModel.value || state.modelSetupInProgress;
  if (!state.modelSetupInProgress) {
    el.modelProgress.classList.add("hidden");
    el.modelProgressBar.style.width = "0%";
    el.modelProgress.setAttribute("aria-valuenow", "0");
  }
}

function renderModelHelp() {
  if (!state.modelCatalog) return;
  const kind = activeModelKind();
  const selected = el.llmModel.value;
  if (kind === "local_duplex") {
    renderModelSetupControls();
    const duplex = state.modelCatalog.local?.duplex || {};
    el.modelHelp.textContent = duplex.ready
      ? `${duplex.model} 已載入 · ${duplex.note}`
      : `${duplex.model || selected} 尚未就緒；請查看伺服器的 MiniCPM-o 啟動 log。`;
    return;
  }
  if (kind !== "local_text") {
    renderModelSetupControls();
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
  const setupEnabled = Boolean(state.modelCatalog.local?.setup_enabled);
  renderModelSetupControls();
  if (installed.has(selected)) {
    const description = recommendation
      ? `已安裝 · ${recommendation.label} · ${recommendation.size} · ${recommendation.note}`
      : "此模型已安裝在目前的 Local LLM 服務。";
    el.modelHelp.textContent = setupEnabled
      ? `${description} · 按「設定模型」可立即載入。`
      : description;
    return;
  }
  el.modelHelp.textContent = setupEnabled
    ? "尚未安裝。按「設定模型」後會自動下載並載入；下載期間請勿關閉頁面。"
    : `尚未安裝。請先在伺服器執行：docker compose -f docker-compose.local.yml exec ollama ollama pull ${selected}`;
}

function renderModelSelect() {
  if (!state.modelCatalog) return;
  const kind = activeModelKind();
  const defaults = state.modelCatalog.defaults || {};
  el.llmModel.replaceChildren();

  if (kind === "local_text") {
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
    recommendedGroup.label = "24GB GPU 多語文字模型（需先下載）";
    for (const item of recommendations) {
      if (installed.includes(item.id)) continue;
      recommendedGroup.append(new Option(`${item.label} · ${item.size} · ${item.note}`, item.id));
    }
    if (recommendedGroup.children.length) el.llmModel.append(recommendedGroup);
  } else if (kind === "local_duplex") {
    const duplex = state.modelCatalog.local?.duplex || {};
    const modelId = duplex.model || defaults.local_duplex;
    if (modelId) el.llmModel.add(new Option("MiniCPM-o 4.5 GPTQ · 原生中英文 Full Duplex", modelId));
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

async function loadModelCatalog(applyBackend = true) {
  try {
    const response = await fetch("/api/models", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(errorDetail(payload, `HTTP ${response.status}`));
    state.modelCatalog = payload;
  } catch (error) {
    if (!applyBackend && state.modelCatalog) {
      el.modelHelp.textContent = `模型已設定，但清單重新整理失敗：${error.message}`;
      return;
    }
    state.modelCatalog = {
      backend: "openai",
      modes: ["pipeline", "realtime"],
      defaults: { text: "gpt-5.6-luna", realtime: "gpt-realtime-2", local_text: "qwen3.5:9b", local_duplex: "openbmb/MiniCPM-o-4_5-GPTQ" },
      openai: { text: ["gpt-5.6-luna"], realtime: ["gpt-realtime-2"] },
      local: {
        installed: [],
        recommended: [{ id: "qwen3.5:9b", label: "Qwen 3.5 9B", size: "6.6 GB", note: "201 種語言／方言" }],
        setup_enabled: false,
        duplex: { model: "openbmb/MiniCPM-o-4_5-GPTQ", ready: false, languages: ["en", "zh-CN", "zh-TW"], note: "原生中英文 Speech-to-Speech" },
      },
      warnings: ["OpenAI model list could not be loaded; using the configured defaults."],
    };
    el.modelHelp.textContent = `模型清單載入失敗：${error.message}`;
  }
  if (applyBackend) applyDeploymentBackend(state.modelCatalog.backend, state.modelCatalog.modes);
  renderModelSelect();
}

function setModelProgress(percent, message) {
  const normalized = Math.min(Math.max(Number(percent) || 0, 0), 100);
  el.modelProgress.classList.remove("hidden");
  el.modelProgressBar.style.width = `${normalized}%`;
  el.modelProgress.setAttribute("aria-valuenow", String(Math.round(normalized)));
  if (message) el.modelHelp.textContent = message;
}

function setModelSetupBusy(busy) {
  state.modelSetupInProgress = busy;
  el.llmModel.disabled = busy || !el.llmModel.value;
  el.modelSetupButton.disabled = busy || !el.llmModel.value;
  el.promptSaveButton.disabled = busy;
  el.modeButtons.forEach((button) => { button.disabled = busy; });
  el.recordButton.disabled = busy;
  el.callButton.disabled = busy;
  if (busy) {
    el.modelSetupButton.textContent = "設定中…";
    setModelProgress(2, "正在連線到 Ollama…");
  } else {
    el.modelSetupButton.textContent = "設定模型";
    renderModelSetupControls();
  }
}

async function setupLocalModel() {
  if (activeModelKind() !== "local_text" || !el.llmModel.value) return;
  if (realtimeActive() || state.recorder?.state === "recording") {
    el.modelHelp.textContent = "請先結束目前的錄音或 Full Duplex 連線，再設定模型。";
    return;
  }

  const model = el.llmModel.value;
  state.modelSelections.local_text = model;
  saveSettings();
  setModelSetupBusy(true);
  const formData = new FormData();
  formData.append("model", model);
  let readyMessage = "";

  try {
    const response = await fetch("/api/local/models/setup", { method: "POST", body: formData });
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
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "phase" && event.phase === "download") {
          setModelProgress(2, event.message || `正在下載 ${model}…`);
        } else if (event.type === "progress") {
          const percent = Number.isFinite(event.percent) ? event.percent : 5;
          const suffix = Number.isFinite(event.percent) ? ` ${event.percent.toFixed(1)}%` : "";
          setModelProgress(percent, `${event.status || "正在下載模型…"}${suffix}`);
        } else if (event.type === "phase" && event.phase === "load") {
          setModelProgress(100, event.message || `正在載入 ${model}…`);
        } else if (event.type === "ready") {
          readyMessage = event.message || `${model} 已載入`;
        } else if (event.type === "error") {
          throw new Error(event.message || "模型設定失敗");
        }
      }
      if (done) break;
    }
    if (!readyMessage) throw new Error("Ollama 未回報模型設定完成");

    state.modelSetupInProgress = false;
    await loadModelCatalog(false);
    setStatus(isRealtimeMode() ? "Local Full Duplex 待命" : "Local Pipeline 待命", "ready");
    el.modelHelp.textContent = `${readyMessage}，現在可以開始對話。`;
  } catch (error) {
    el.modelHelp.textContent = `模型設定失敗：${error.message}`;
  } finally {
    setModelSetupBusy(false);
  }
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
    ? "目前為 Local Pipeline 知識庫：檔案在本機解析，embedding 與 Qdrant 都走地端。原生 Local Full Duplex 暫不支援 RAG。支援 PDF、DOCX、PPTX、文字與常見程式碼格式。"
    : "目前為 OpenAI 知識庫：完成索引後，兩個 OpenAI 模式會共用相同檔案。";
}

function setRagControlsBusy(busy) {
  el.modeButtons.forEach((button) => { button.disabled = busy; });
  el.ragFiles.disabled = busy;
  el.ragUploadButton.disabled = busy;
  el.ragDeleteButton.disabled = busy || !activeKnowledge().token;
  el.recordButton.disabled = busy;
  el.callButton.disabled = busy;
  el.promptSaveButton.disabled = busy;
}

let saveTimer;
function saveSettings() {
  const settings = currentSettings();
  settings.systemPrompt = state.savedPrompt || settings.systemPrompt;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  el.savedIndicator.textContent = state.promptDirty ? "Prompt 尚未儲存" : "已儲存";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    el.savedIndicator.textContent = state.promptDirty ? "Prompt 尚未儲存" : "自動儲存";
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
  if (realtimeActive()) stopRealtime();
  if (state.recorder && state.recorder.state === "recording") {
    state.discardRecording = true;
    state.recorder.stop();
  }

  state.mode = mode;
  el.modeButtons.forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  const realtime = isRealtimeMode();
  const local = isLocalMode();
  const nativeLocalDuplex = mode === "local-realtime";
  el.pipelineControls.classList.toggle("hidden", realtime);
  el.realtimeControls.classList.toggle("hidden", !realtime);
  const descriptions = {
    pipeline: "OpenAI 依序完成語音辨識、Responses 回覆與語音合成，適合保留清楚的逐輪紀錄。",
    realtime: "OpenAI Realtime 端到端雙向串流，可自然插話並立即打斷 AI。",
    "local-pipeline": "OpenAI 負責 ASR/TTS；RAG、embedding、Qdrant 與 LLM 回覆都在地端執行。",
    "local-realtime": "MiniCPM-o 4.5 在地端直接接收音訊並同步產生語音；支援中英文連續對話與插話，不經 OpenAI ASR/TTS。",
  };
  el.modeDescription.textContent = realtime
    ? `${descriptions[mode]} 連線期間修改設定時，請重新連線套用。`
    : descriptions[mode];
  el.privacyNote.textContent = nativeLocalDuplex
    ? "語音由 AI 生成。此模式的麥克風音訊、理解與語音輸出均由地端 MiniCPM-o 處理，不會送至 OpenAI；此模式目前不支援 RAG。"
    : local
      ? "語音由 AI 生成。RAG、embedding 與文字 LLM 留在地端；Pipeline 的麥克風音訊會送至 OpenAI ASR，回答文字會送至 OpenAI TTS。"
    : "語音由 AI 生成。OpenAI 模式會將上傳檔案交由 OpenAI 建立檢索索引；請勿上傳未經授權的機密資料。";
  el.knowledgeCard.classList.toggle("hidden", nativeLocalDuplex);
  for (const option of [...el.languageA.options, ...el.languageB.options]) {
    option.disabled = nativeLocalDuplex && !LOCAL_DUPLEX_LANGUAGES.has(option.value);
  }
  if (nativeLocalDuplex && !LOCAL_DUPLEX_LANGUAGES.has(el.languageA.value)) el.languageA.value = "zh-TW";
  if (nativeLocalDuplex && !LOCAL_DUPLEX_LANGUAGES.has(el.languageB.value)) el.languageB.value = "en";
  el.voice.disabled = nativeLocalDuplex;
  el.voice.title = nativeLocalDuplex ? "Local Full Duplex 使用 MiniCPM-o 參考音訊的固定聲線" : "";
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
  if (realtimeActive()) stopRealtime();
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
  if (realtimeActive()) stopRealtime();
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
  el.promptSaveButton.disabled = true;
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
    el.promptSaveButton.disabled = false;
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

function bytesToBase64(bytes) {
  let binary = "";
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
  }
  return btoa(binary);
}

function floatToPcm16Base64(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  samples.forEach((sample, index) => {
    const limited = Math.max(-1, Math.min(1, sample));
    view.setInt16(index * 2, limited < 0 ? limited * 0x8000 : limited * 0x7fff, true);
  });
  return bytesToBase64(new Uint8Array(buffer));
}

function resampleMono(input, sourceRate, targetRate) {
  if (sourceRate === targetRate) return Float32Array.from(input);
  const outputLength = Math.max(1, Math.round(input.length * targetRate / sourceRate));
  const output = new Float32Array(outputLength);
  const ratio = sourceRate / targetRate;
  for (let index = 0; index < outputLength; index += 1) {
    const position = index * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const mix = position - left;
    output[index] = input[left] * (1 - mix) + input[right] * mix;
  }
  return output;
}

function sendLocalDuplexEvent(event) {
  if (state.localDuplexSocket?.readyState === WebSocket.OPEN) {
    state.localDuplexSocket.send(JSON.stringify(event));
  }
}

function flushLocalDuplexInput(final = false) {
  const frameSamples = 3200;
  while (state.localDuplexInputSamples.length >= frameSamples || (final && state.localDuplexInputSamples.length)) {
    const count = final ? Math.min(frameSamples, state.localDuplexInputSamples.length) : frameSamples;
    const frame = state.localDuplexInputSamples.splice(0, count);
    const durationMs = frame.length * 1000 / LOCAL_DUPLEX_INPUT_RATE;
    state.localDuplexAudioEndMs += durationMs;
    sendLocalDuplexEvent({
      type: "input_audio_buffer.append",
      audio: floatToPcm16Base64(frame),
      input_audio_format: "pcm16",
      sample_rate_hz: LOCAL_DUPLEX_INPUT_RATE,
      duration_ms: durationMs,
      audio_end_ms: state.localDuplexAudioEndMs,
    });
  }
}

function stopLocalDuplexPlayback() {
  for (const source of state.localDuplexPlaybackSources) {
    state.localDuplexCancelledSources.add(source);
    try { source.stop(); } catch { /* already stopped */ }
  }
  state.localDuplexPlaybackSources.clear();
  state.localDuplexNextPlaybackTime = 0;
}

function queueLocalDuplexPcm(audioBase64, sampleRate, responseId, itemId) {
  const binary = atob(audioBase64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  const sampleCount = Math.floor(bytes.length / 2);
  if (!sampleCount) return;

  const context = state.localDuplexOutputContext;
  if (!context) return;
  const audioBuffer = context.createBuffer(1, sampleCount, sampleRate || LOCAL_DUPLEX_OUTPUT_RATE);
  const channel = audioBuffer.getChannelData(0);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < sampleCount; index += 1) {
    channel[index] = view.getInt16(index * 2, true) / 0x8000;
  }

  const source = context.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(context.destination);
  const startAt = Math.max(context.currentTime + 0.02, state.localDuplexNextPlaybackTime || 0);
  state.localDuplexNextPlaybackTime = startAt + audioBuffer.duration;
  state.localDuplexPlaybackSources.add(source);
  source.addEventListener("ended", () => {
    state.localDuplexPlaybackSources.delete(source);
    if (state.localDuplexCancelledSources.delete(source)) return;
    if (!responseId) return;
    const playedMs = (state.localDuplexPlayedMs.get(responseId) || 0) + audioBuffer.duration * 1000;
    state.localDuplexPlayedMs.set(responseId, playedMs);
    sendLocalDuplexEvent({
      type: "playback.ack",
      response_id: responseId,
      item_id: itemId || `item_${responseId}`,
      played_ms: playedMs,
      committed_ms: playedMs,
    });
  });
  source.start(startAt);
}

function handleLocalDuplexEvent(event) {
  const type = event.type || "";
  if (type === "session.created" || type === "session.updated") {
    setStatus("Local Full Duplex 已連線，可直接說話", "ready");
    return;
  }
  if (type.includes("input_audio_transcription") && type.endsWith(".delta")) {
    state.inputTranscript += event.delta || "";
    showLiveCaption("user", state.inputTranscript);
    return;
  }
  if (type.includes("input_audio_transcription") && (type.endsWith(".completed") || type.endsWith(".done"))) {
    const transcript = (event.transcript || state.inputTranscript).trim();
    if (transcript) addMessage("user", transcript);
    state.inputTranscript = "";
    showLiveCaption("user", "");
    return;
  }
  if (type === "response.audio.delta") {
    queueLocalDuplexPcm(
      event.delta || event.audio || "",
      event.sample_rate_hz || LOCAL_DUPLEX_OUTPUT_RATE,
      event.response_id || event.response?.id || "",
      event.item_id || "",
    );
    return;
  }
  if (type === "response.audio_transcript.delta" || type === "response.output_text.delta") {
    state.outputTranscript += event.delta || "";
    showLiveCaption("assistant", state.outputTranscript);
    return;
  }
  if (type === "response.listen") {
    stopLocalDuplexPlayback();
    setStatus("正在聆聽…", "live");
    return;
  }
  if (type === "response.created") {
    state.outputTranscript = "";
    setStatus("MiniCPM-o 正在回答…", "busy");
    return;
  }
  if (type === "response.done") {
    const transcript = (event.transcript || event.response?.output_text || state.outputTranscript).trim();
    if (transcript) addMessage("assistant", transcript);
    state.outputTranscript = "";
    showLiveCaption("assistant", "");
    setStatus("Local Full Duplex 已連線，可直接說話", "ready");
    return;
  }
  if (type === "error") {
    const message = typeof event.error === "string"
      ? event.error
      : event.error?.message || event.message || "未知錯誤";
    setStatus(`Local Full Duplex 錯誤：${message}`, "error");
  }
}

async function startLocalDuplex() {
  el.callButton.disabled = true;
  el.promptSaveButton.disabled = true;
  el.modeButtons.forEach((button) => { button.disabled = true; });
  setStatus("正在連接地端 MiniCPM-o…", "busy");
  try {
    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    state.localDuplexInputContext = new AudioContextClass();
    state.localDuplexOutputContext = new AudioContextClass();
    await Promise.all([state.localDuplexInputContext.resume(), state.localDuplexOutputContext.resume()]);

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${location.host}/api/local/duplex`);
    state.localDuplexSocket = socket;
    await new Promise((resolve, reject) => {
      socket.addEventListener("open", resolve, { once: true });
      socket.addEventListener("error", () => reject(new Error("無法連接 Local Full Duplex WebSocket")), { once: true });
    });
    const sessionReady = new Promise((resolve, reject) => {
      const timeout = window.setTimeout(() => reject(new Error("MiniCPM-o session 建立逾時")), 30000);
      socket.addEventListener("message", (message) => {
        try {
          const event = JSON.parse(message.data);
          handleLocalDuplexEvent(event);
          if (event.type === "session.created") {
            window.clearTimeout(timeout);
            resolve();
          } else if (event.type === "error") {
            window.clearTimeout(timeout);
            const message = typeof event.error === "string"
              ? event.error
              : event.error?.message || event.message || "MiniCPM-o session 建立失敗";
            reject(new Error(message));
          }
        } catch { /* diagnostic event */ }
      });
    });
    socket.addEventListener("close", () => {
      if (state.localDuplexSocket === socket) {
        stopRealtime(false);
        setStatus("Local Full Duplex 連線已結束", "error");
      }
    });
    const settings = currentSettings();
    socket.send(JSON.stringify({
      type: "voice_chat.configure",
      system_prompt: settings.systemPrompt,
      language_a: settings.languageA,
      language_b: settings.languageB,
      voice: settings.voice,
      llm_model: el.llmModel.value,
    }));
    await sessionReady;

    state.localDuplexSource = state.localDuplexInputContext.createMediaStreamSource(state.micStream);
    state.localDuplexProcessor = state.localDuplexInputContext.createScriptProcessor(4096, 1, 1);
    state.localDuplexGain = state.localDuplexInputContext.createGain();
    state.localDuplexGain.gain.value = 0;
    state.localDuplexProcessor.addEventListener("audioprocess", (audioEvent) => {
      if (state.muted || socket.readyState !== WebSocket.OPEN) return;
      const mono = audioEvent.inputBuffer.getChannelData(0);
      const resampled = resampleMono(mono, state.localDuplexInputContext.sampleRate, LOCAL_DUPLEX_INPUT_RATE);
      state.localDuplexInputSamples.push(...resampled);
      flushLocalDuplexInput();
    });
    state.localDuplexSource.connect(state.localDuplexProcessor);
    state.localDuplexProcessor.connect(state.localDuplexGain);
    state.localDuplexGain.connect(state.localDuplexInputContext.destination);
    state.localDuplexAudioEndMs = 0;
    state.localDuplexNextPlaybackTime = 0;
    el.callButton.classList.add("connected");
    el.callButtonLabel.textContent = "結束 Local Full Duplex";
    el.muteButton.disabled = false;
  } catch (error) {
    stopRealtime(false);
    setStatus(`連線失敗：${error.message}`, "error");
  } finally {
    el.callButton.disabled = false;
    el.promptSaveButton.disabled = false;
    el.modeButtons.forEach((button) => { button.disabled = false; });
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
    if (transcript) addMessage("user", transcript);
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
  if (state.mode === "local-realtime") return startLocalDuplex();
  el.callButton.disabled = true;
  el.promptSaveButton.disabled = true;
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
    state.dataChannel.addEventListener("open", () => setStatus("已連線，可直接說話", "ready"));
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
    const response = await fetch("/api/realtime/session", { method: "POST", body: formData });
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
    el.callButtonLabel.textContent = "結束 Full Duplex";
    el.muteButton.disabled = false;
  } catch (error) {
    stopRealtime(false);
    setStatus(`連線失敗：${error.message}`, "error");
  } finally {
    el.callButton.disabled = false;
    el.promptSaveButton.disabled = false;
    el.modeButtons.forEach((button) => { button.disabled = false; });
  }
}

function stopRealtime(updateStatus = true) {
  if (state.localDuplexSocket?.readyState === WebSocket.OPEN) {
    flushLocalDuplexInput(true);
    sendLocalDuplexEvent({ type: "input_audio_buffer.commit", final: true });
    sendLocalDuplexEvent({ type: "session.close" });
  }
  const localSocket = state.localDuplexSocket;
  state.localDuplexSocket = null;
  localSocket?.close();
  state.localDuplexProcessor?.disconnect();
  state.localDuplexSource?.disconnect();
  state.localDuplexGain?.disconnect();
  stopLocalDuplexPlayback();
  void state.localDuplexInputContext?.close();
  void state.localDuplexOutputContext?.close();
  state.localDuplexProcessor = null;
  state.localDuplexSource = null;
  state.localDuplexGain = null;
  state.localDuplexInputContext = null;
  state.localDuplexOutputContext = null;
  state.localDuplexInputSamples = [];
  state.localDuplexAudioEndMs = 0;
  state.localDuplexPlayedMs.clear();
  state.localDuplexCancelledSources.clear();
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
  state.savedPrompt = el.systemPrompt.value.trim();
  el.languageA.value = settings.languageA;
  el.languageB.value = settings.languageB;
  el.voice.value = settings.voice;
  state.modelSelections = { ...state.modelSelections, ...settings.models };
  void loadModelCatalog();
  loadKnowledgeBase();
  renderKnowledgeBase();

  el.modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  [el.languageA, el.languageB, el.voice].forEach((input) => input.addEventListener("change", saveSettings));
  el.llmModel.addEventListener("change", () => {
    state.modelSelections[activeModelKind()] = el.llmModel.value;
    renderModelHelp();
    saveSettings();
  });
  el.modelSetupButton.addEventListener("click", setupLocalModel);
  el.promptSaveButton.addEventListener("click", savePromptDefault);
  el.systemPrompt.addEventListener("input", () => markPromptDirty());
  el.recordButton.addEventListener("click", () => {
    if (state.recorder?.state === "recording") stopRecording();
    else startRecording();
  });
  el.callButton.addEventListener("click", () => {
    if (realtimeActive()) stopRealtime();
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
