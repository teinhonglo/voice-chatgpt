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
const RAG_STORAGE_KEY = "voice-chatgpt-dual-mode-rag";
const RAG_TOOL_NAME = "search_knowledge_base";

const el = {
  modeButtons: [...document.querySelectorAll(".mode-button")],
  systemPrompt: document.querySelector("#system-prompt"),
  languageA: document.querySelector("#language-a"),
  languageB: document.querySelector("#language-b"),
  voice: document.querySelector("#voice"),
  savedIndicator: document.querySelector("#saved-indicator"),
  ragFiles: document.querySelector("#rag-files"),
  ragUploadButton: document.querySelector("#rag-upload-button"),
  ragDeleteButton: document.querySelector("#rag-delete-button"),
  ragStatus: document.querySelector("#rag-status"),
  ragFileList: document.querySelector("#rag-file-list"),
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
  ragToken: "",
  ragFileNames: [],
};

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
  const defaults = { languageA: "zh-TW", languageB: "en", voice: "marin" };
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return { ...defaults, ...saved };
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
  };
}

function loadKnowledgeBase() {
  try {
    const saved = JSON.parse(localStorage.getItem(RAG_STORAGE_KEY) || "{}");
    if (typeof saved.token !== "string" || !Array.isArray(saved.files)) return;
    state.ragToken = saved.token;
    state.ragFileNames = saved.files.filter((name) => typeof name === "string").slice(0, 100);
  } catch {
    localStorage.removeItem(RAG_STORAGE_KEY);
  }
}

function saveKnowledgeBase() {
  if (!state.ragToken) {
    localStorage.removeItem(RAG_STORAGE_KEY);
    return;
  }
  localStorage.setItem(
    RAG_STORAGE_KEY,
    JSON.stringify({ token: state.ragToken, files: state.ragFileNames }),
  );
}

function setRagStatus(message, kind = "idle") {
  el.ragStatus.textContent = message;
  el.ragStatus.className = `knowledge-state ${kind}`;
}

function renderKnowledgeBase() {
  el.ragFileList.replaceChildren();
  for (const filename of state.ragFileNames) {
    const chip = document.createElement("span");
    chip.className = "file-chip";
    chip.textContent = filename;
    chip.title = filename;
    el.ragFileList.append(chip);
  }
  el.ragFileList.classList.toggle("hidden", state.ragFileNames.length === 0);
  el.ragDeleteButton.disabled = !state.ragToken;
  setRagStatus(
    state.ragToken ? `已啟用 · ${state.ragFileNames.length} 個檔案` : "尚未上傳檔案",
    state.ragToken ? "ready" : "idle",
  );
}

function setRagControlsBusy(busy) {
  el.ragFiles.disabled = busy;
  el.ragUploadButton.disabled = busy;
  el.ragDeleteButton.disabled = busy || !state.ragToken;
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
  if (state.ragToken) formData.append("rag_token", state.ragToken);
}

function errorDetail(payload, fallback) {
  if (payload && typeof payload.detail === "string") return payload.detail;
  return fallback;
}

async function setMode(mode) {
  if (mode === state.mode) return;
  if (state.pc) stopRealtime();
  if (state.recorder && state.recorder.state === "recording") {
    state.discardRecording = true;
    state.recorder.stop();
  }

  state.mode = mode;
  el.modeButtons.forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  const realtime = mode === "realtime";
  el.pipelineControls.classList.toggle("hidden", realtime);
  el.realtimeControls.classList.toggle("hidden", !realtime);
  el.modeDescription.textContent = realtime
    ? "持續雙向串流，可自然插話並立即打斷 AI。連線期間修改設定不會生效，請重新連線套用。"
    : "錄一段話後，依序完成語音辨識、文字回覆與語音合成。適合保留清楚的逐輪紀錄。";
  setStatus(realtime ? "Full Duplex 待命" : "Pipeline 待命");
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

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file, file.name));
  if (state.ragToken) formData.append("rag_token", state.ragToken);

  try {
    const response = await fetch("/api/rag/upload", { method: "POST", body: formData });
    let payload;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) throw new Error(errorDetail(payload, `HTTP ${response.status}`));

    state.ragToken = payload.rag_token;
    state.ragFileNames = [...state.ragFileNames, ...payload.files];
    saveKnowledgeBase();
    el.ragFiles.value = "";
    renderKnowledgeBase();
  } catch (error) {
    setRagStatus(`上傳失敗：${error.message}`, "error");
  } finally {
    setRagControlsBusy(false);
  }
}

async function deleteKnowledgeBase() {
  if (!state.ragToken) return;
  if (!window.confirm("要刪除這個知識庫及其 OpenAI 檔案嗎？")) return;
  if (state.pc) stopRealtime();
  if (state.recorder?.state === "recording") {
    state.discardRecording = true;
    state.recorder.stop();
  }
  setRagControlsBusy(true);
  setRagStatus("正在刪除知識庫…");

  const formData = new FormData();
  formData.append("rag_token", state.ragToken);
  try {
    const response = await fetch("/api/rag/delete", { method: "POST", body: formData });
    let payload;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) throw new Error(errorDetail(payload, `HTTP ${response.status}`));
    state.ragToken = "";
    state.ragFileNames = [];
    saveKnowledgeBase();
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
  el.recordButton.disabled = true;
  el.ragFiles.disabled = true;
  el.ragUploadButton.disabled = true;
  el.ragDeleteButton.disabled = true;
  setStatus("OpenAI 正在處理…", "busy");
  const extension = blob.type.includes("mp4") ? "m4a" : "webm";
  const formData = new FormData();
  formData.append("audio", blob, `recording.${extension}`);
  formData.append("history_json", JSON.stringify(state.pipelineHistory));
  appendSettings(formData);

  try {
    const response = await fetch("/api/pipeline/turn", { method: "POST", body: formData });
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
    setStatus("Pipeline 待命", "ready");
  } catch (error) {
    setStatus(`處理失敗：${error.message}`, "error");
  } finally {
    el.recordButton.disabled = false;
    el.ragFiles.disabled = false;
    el.ragUploadButton.disabled = false;
    el.ragDeleteButton.disabled = !state.ragToken;
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
  setStatus("正在查詢知識庫…", "busy");
  for (const call of toolCalls) {
    let output;
    try {
      const args = JSON.parse(call.arguments || "{}");
      const query = typeof args.query === "string" ? args.query.trim() : "";
      if (!query) throw new Error("Realtime 未提供有效的檢索問題");
      if (!state.ragToken) throw new Error("知識庫已停用，請重新連線");

      const formData = new FormData();
      formData.append("query", query);
      formData.append("rag_token", state.ragToken);
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
    if (state.ragToken) {
      setRagStatus(`已啟用 · ${state.ragFileNames.length} 個檔案`, "ready");
    }
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
  el.callButton.disabled = true;
  setStatus("正在建立安全連線…", "busy");
  try {
    state.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
  }
}

function stopRealtime(updateStatus = true) {
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
  el.callButtonLabel.textContent = "開始 Full Duplex";
  el.muteButton.textContent = "靜音";
  el.muteButton.disabled = true;
  el.liveCaption.classList.add("hidden");
  if (updateStatus) setStatus("Full Duplex 待命");
}

function toggleMute() {
  state.muted = !state.muted;
  state.micStream?.getAudioTracks().forEach((track) => {
    track.enabled = !state.muted;
  });
  el.muteButton.textContent = state.muted ? "取消靜音" : "靜音";
  setStatus(state.muted ? "麥克風已靜音" : "已連線，可直接說話", state.muted ? "idle" : "ready");
}

function initialize() {
  populateSelects();
  const settings = loadSettings();
  el.systemPrompt.value = settings.systemPrompt || el.systemPrompt.value;
  el.languageA.value = settings.languageA;
  el.languageB.value = settings.languageB;
  el.voice.value = settings.voice;
  loadKnowledgeBase();
  renderKnowledgeBase();

  el.modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  [el.systemPrompt, el.languageA, el.languageB, el.voice].forEach((input) => input.addEventListener("change", saveSettings));
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
