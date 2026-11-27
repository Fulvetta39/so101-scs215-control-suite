const camera = document.querySelector("#camera");
const micButton = document.querySelector("#mic-button");
const micLabel = document.querySelector("#mic-label");
const micHint = document.querySelector("#mic-hint");
const textForm = document.querySelector("#text-form");
const textInput = document.querySelector("#text-input");
const sendButton = document.querySelector("#send-button");
const chatLog = document.querySelector("#chat-log");
const thinking = document.querySelector("#thinking");
const thinkingLabel = document.querySelector("#thinking-label");
const modelPill = document.querySelector("#model-pill");
const armPill = document.querySelector("#arm-pill");
const cameraMeta = document.querySelector("#camera-meta");
const cameraError = document.querySelector("#camera-error");
const backendName = document.querySelector("#backend-name");
const safetyState = document.querySelector("#safety-state");
const poseList = document.querySelector("#pose-list");
const poseName = document.querySelector("#pose-name");
const savePoseButton = document.querySelector("#save-pose");
const stopButton = document.querySelector("#stop-button");
const resumeButton = document.querySelector("#resume-button");
const resetButton = document.querySelector("#reset-button");
const toast = document.querySelector("#toast");
const waveform = document.querySelector("#waveform");
const clock = document.querySelector("#clock");

let recorder = null;
let busy = false;
let toastTimer = null;
let voiceCache = [];
let cameraObjectUrl = null;
let cameraTimer = null;

for (let index = 0; index < 27; index += 1) {
  const bar = document.createElement("i");
  bar.style.animationDelay = `${(index % 9) * 0.055}s`;
  waveform.appendChild(bar);
}

function setPill(element, text, state) {
  element.classList.remove("ok", "bad");
  if (state) element.classList.add(state);
  element.querySelector("span").textContent = text;
}

function showToast(message, isError = false) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("show");
  toastTimer = setTimeout(() => toast.classList.remove("show"), 5200);
}

function setBusy(value, label = "正在观察并规划动作") {
  busy = value;
  thinking.hidden = !value;
  thinkingLabel.textContent = label;
  sendButton.disabled = value;
  micButton.disabled = value;
}

function addMessage(role, text, actions = []) {
  const article = document.createElement("div");
  article.className = `message ${role}`;
  const author = document.createElement("div");
  author.className = "message-author";
  const name = document.createElement("span");
  name.textContent = role === "assistant" ? "小臂" : "你";
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });
  author.append(name, time);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  article.append(author, bubble);

  if (actions.length) {
    const trace = document.createElement("div");
    trace.className = "action-trace";
    actions.forEach((action) => {
      const chip = document.createElement("span");
      const argumentValues = Object.values(action.arguments || {}).join(" ");
      chip.textContent = `${action.tool}${argumentValues ? ` · ${argumentValues}` : ""}`;
      if (!action.ok) chip.classList.add("failed");
      chip.title = action.message || "";
      trace.appendChild(chip);
    });
    article.appendChild(trace);
  }

  chatLog.appendChild(article);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function speak(text) {
  if (!("speechSynthesis" in window) || !text) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "zh-CN";
  utterance.rate = 1.03;
  utterance.pitch = 0.96;
  if (!voiceCache.length) {
    voiceCache = window.speechSynthesis.getVoices();
  }
  const voice = voiceCache.find((item) => /^zh/i.test(item.lang));
  if (voice) utterance.voice = voice;
  window.speechSynthesis.speak(utterance);
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const status = await response.json();
    if (status.agent_ready) {
      setPill(modelPill, status.model, "ok");
    } else {
      setPill(modelPill, "需要 API Key", "bad");
      if (status.agent_error) showToast(status.agent_error, true);
    }
    const robot = status.robot || {};
    if (robot.connected) {
      setPill(armPill, `${robot.backend.toUpperCase()} · SO101`, "ok");
    } else {
      setPill(armPill, "机械臂未连接", "bad");
    }
    backendName.textContent = robot.backend || "--";
    safetyState.textContent = robot.stopped ? "STOPPED" : "ENABLED";
    safetyState.classList.toggle("safe", !robot.stopped);
    resumeButton.disabled = !robot.stopped;
    cameraMeta.textContent =
      `CAM ${status.camera?.mode?.toUpperCase() || "--"} · ` +
      `${status.camera?.width || "--"}×${status.camera?.height || "--"}`;
    renderPoses(robot.poses || []);
  } catch (error) {
    setPill(modelPill, "服务未连接", "bad");
    setPill(armPill, "服务未连接", "bad");
  }
}

function renderPoses(poses) {
  poseList.replaceChildren();
  if (!poses.length) {
    const empty = document.createElement("span");
    empty.className = "empty";
    empty.textContent = "还没有命名姿态";
    poseList.appendChild(empty);
    return;
  }
  poses.forEach((pose) => {
    const chip = document.createElement("span");
    chip.textContent = pose;
    poseList.appendChild(chip);
  });
}

function mergeFloatChunks(chunks) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function downsample(samples, inputRate, outputRate = 16000) {
  if (inputRate === outputRate) return samples;
  const ratio = inputRate / outputRate;
  const outputLength = Math.max(1, Math.round(samples.length / ratio));
  const output = new Float32Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(samples.length, Math.floor((index + 1) * ratio));
    let sum = 0;
    for (let source = start; source < end; source += 1) sum += samples[source];
    output[index] = sum / Math.max(1, end - start);
  }
  return output;
}

function encodeWav(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

class WavRecorder {
  constructor() {
    this.stream = null;
    this.context = null;
    this.source = null;
    this.processor = null;
    this.silentGain = null;
    this.chunks = [];
    this.startedAt = 0;
    this.timer = null;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    this.context = new AudioContext();
    await this.context.resume();
    this.source = this.context.createMediaStreamSource(this.stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.silentGain = this.context.createGain();
    this.silentGain.gain.value = 0;
    this.chunks = [];
    this.processor.onaudioprocess = (event) => {
      this.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    this.source.connect(this.processor);
    this.processor.connect(this.silentGain);
    this.silentGain.connect(this.context.destination);
    this.startedAt = performance.now();
  }

  async stop() {
    if (!this.context) return null;
    const duration = performance.now() - this.startedAt;
    if (this.processor) this.processor.onaudioprocess = null;
    if (this.source) this.source.disconnect();
    if (this.processor) this.processor.disconnect();
    if (this.silentGain) this.silentGain.disconnect();
    if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
    const sampleRate = this.context.sampleRate;
    await this.context.close();
    this.context = null;
    this.stream = null;
    if (duration < 500 || !this.chunks.length) return null;
    const merged = mergeFloatChunks(this.chunks);
    const downsampled = downsample(merged, sampleRate, 16000);
    return encodeWav(downsampled);
  }
}

async function sendInput(text, audioBlob = null) {
  if (busy) return;
  if (!text.trim() && !audioBlob) {
    showToast("没有说话内容，请重试。", true);
    return;
  }
  addMessage("user", text.trim() || "🎙 语音指令");
  setBusy(true, audioBlob ? "正在理解语音与画面" : "正在理解画面");
  try {
    const form = new FormData();
    form.append("text", text);
    if (audioBlob) form.append("audio", audioBlob, "voice.wav");
    const response = await fetch("/api/chat", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "请求失败");
    addMessage("assistant", result.reply, result.actions || []);
    speak(result.reply);
    await loadStatus();
  } catch (error) {
    addMessage("assistant", `这次没有完成：${error.message}`);
    showToast(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function toggleRecording() {
  if (busy) return;
  if (recorder) {
    const active = recorder;
    recorder = null;
    micButton.classList.remove("recording");
    waveform.classList.remove("active");
    micLabel.textContent = "正在整理语音";
    micHint.textContent = "请稍候";
    const blob = await active.stop();
    if (!blob) {
      micLabel.textContent = "录音太短";
      micHint.textContent = "按住说话约一秒以上再停止";
      return;
    }
    micLabel.textContent = "点击说话";
    micHint.textContent = "语音已采集，正在交给模型";
    await sendInput("", blob);
    micHint.textContent = "录音结束后，语音与画面会一起交给 Qwen-Omni";
    return;
  }

  try {
    recorder = new WavRecorder();
    await recorder.start();
    micButton.classList.add("recording");
    waveform.classList.add("active");
    micLabel.textContent = "正在聆听";
    micHint.textContent = "再次点击结束 · 最长 20 秒";
    window.setTimeout(() => {
      if (recorder) toggleRecording();
    }, 20000);
  } catch (error) {
    recorder = null;
    showToast(`无法使用麦克风：${error.message}`, true);
  }
}

micButton.addEventListener("click", toggleRecording);

document.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName);
  if (event.code === "Space" && !typing && !event.repeat) {
    event.preventDefault();
    toggleRecording();
  }
});

textForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = textInput.value;
  textInput.value = "";
  await sendInput(text);
});

textInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    textForm.requestSubmit();
  }
});

stopButton.addEventListener("click", async () => {
  try {
    const response = await fetch("/api/stop", { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "急停失败");
    showToast(result.message);
    safetyState.textContent = "STOPPED";
    safetyState.classList.remove("safe");
    resumeButton.disabled = false;
  } catch (error) {
    showToast(error.message, true);
  }
});

resumeButton.addEventListener("click", async () => {
  if (!window.confirm("请确认机械臂周围清空、急停可触达。是否恢复力矩并继续？")) return;
  resumeButton.disabled = true;
  try {
    const response = await fetch("/api/resume", { method: "POST" });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.detail || "恢复失败");
    showToast(result.message);
    safetyState.textContent = "ENABLED";
    safetyState.classList.add("safe");
  } catch (error) {
    resumeButton.disabled = false;
    showToast(error.message, true);
  }
});

savePoseButton.addEventListener("click", async () => {
  const name = poseName.value.trim();
  if (!name) {
    showToast("请先填写姿态名称。", true);
    return;
  }
  savePoseButton.disabled = true;
  try {
    const response = await fetch(`/api/poses/${encodeURIComponent(name)}`, {
      method: "POST",
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "记录失败");
    poseName.value = "";
    showToast(`已记录姿态：${name}`);
    await loadStatus();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    savePoseButton.disabled = false;
  }
});

resetButton.addEventListener("click", async () => {
  try {
    await fetch("/api/history/reset", { method: "POST" });
    chatLog.replaceChildren();
    addMessage(
      "assistant",
      "对话已清空。当前相机画面与机械臂状态不受影响。",
    );
  } catch {
    showToast("清空对话失败。", true);
  }
});

async function refreshCamera() {
  try {
    const response = await fetch(`/api/frame?t=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error("camera unavailable");
    const blob = await response.blob();
    const nextUrl = URL.createObjectURL(blob);
    camera.src = nextUrl;
    if (cameraObjectUrl) URL.revokeObjectURL(cameraObjectUrl);
    cameraObjectUrl = nextUrl;
    cameraError.hidden = true;
  } catch {
    cameraError.hidden = false;
    cameraError.textContent = "相机画面暂时中断，正在重连...";
  } finally {
    cameraTimer = window.setTimeout(refreshCamera, 250);
  }
}

window.setInterval(() => {
  clock.textContent = new Date().toLocaleTimeString("zh-CN", {
    hour12: false,
  });
}, 1000);

window.setInterval(loadStatus, 15000);
if ("speechSynthesis" in window) {
  window.speechSynthesis.onvoiceschanged = () => {
    voiceCache = window.speechSynthesis.getVoices();
  };
}

window.addEventListener("beforeunload", () => {
  window.clearTimeout(cameraTimer);
  if (cameraObjectUrl) URL.revokeObjectURL(cameraObjectUrl);
});
refreshCamera();
clock.textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
loadStatus();

