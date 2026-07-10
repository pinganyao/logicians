const form = document.getElementById("settings-form");
const midiInput = document.getElementById("midi-input");
const midiOutput = document.getElementById("midi-output");
const drumOutput = document.getElementById("drum-output");
const bassOutput = document.getElementById("bass-output");
const drumStyle = document.getElementById("drum-style");

const btnStart = document.getElementById("btn-start");
const btnImprovise = document.getElementById("btn-improvise");
const btnStop = document.getElementById("btn-stop");
const btnReset = document.getElementById("btn-reset");

const statusDot = document.getElementById("status-dot");
const statusState = document.getElementById("status-state");
const statusPosition = document.getElementById("status-position");
const posLoop = document.getElementById("pos-loop");
const posBar = document.getElementById("pos-bar");
const posBeat = document.getElementById("pos-beat");
const beatTrack = document.getElementById("beat-track");
const beatTicks = beatTrack ? [...beatTrack.children] : [];

const statusKey = document.getElementById("status-key");
const keyLabel = document.getElementById("key-label");
const statusChords = document.getElementById("status-chords");
const statusMessage = document.getElementById("status-message");
const statusError = document.getElementById("status-error");
const statusWarning = document.getElementById("status-warning");

let pollTimer = null;

function formatState(state) {
  if (!state) return "Idle";
  return state.charAt(0).toUpperCase() + state.slice(1).replace(/-/g, " ");
}

function setSettingsDisabled(disabled) {
  form.querySelectorAll("input, select").forEach((el) => {
    el.disabled = disabled;
  });
}

function fillSelect(select, options, preferred) {
  select.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "(disabled)";
  select.appendChild(empty);
  (options || []).forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    select.appendChild(opt);
  });
  if (preferred && options.includes(preferred)) {
    select.value = preferred;
  } else if (options && options.length > 0) {
    select.selectedIndex = options.length > 1 ? 1 : 0;
  }
}

async function loadPorts() {
  try {
    const res = await fetch("/api/ports");
    if (!res.ok) throw new Error(`ports ${res.status}`);
    const data = await res.json();
    fillSelect(midiInput, data.inputs, "IAC Driver Bus 1");
    fillSelect(midiOutput, data.outputs, "IAC Driver Bus 2");
    fillSelect(drumOutput, data.outputs, "IAC Driver Bus 2");
    fillSelect(bassOutput, data.outputs, "IAC Driver Bus 2");
  } catch (err) {
    statusError.hidden = false;
    statusError.textContent = `Cannot reach server: ${err.message}`;
  }
}

async function loadDrumStyles() {
  try {
    const res = await fetch("/api/drum-styles");
    if (!res.ok) return;
    const data = await res.json();
    (data.styles || []).forEach((style) => {
      const opt = document.createElement("option");
      opt.value = style;
      opt.textContent = style;
      drumStyle.appendChild(opt);
    });
  } catch (_) {}
}

function getConfig() {
  const fd = new FormData(form);
  const seedVal = fd.get("seed");
  const keyVal = fd.get("key");
  return {
    midi_input: fd.get("midi_input"),
    midi_output: fd.get("midi_output"),
    drum_output: fd.get("drum_output") || null,
    drum_channel: Number(fd.get("drum_channel")),
    drum_variation: Number(fd.get("drum_variation")),
    drum_style: fd.get("drum_style") || null,
    bass_output: fd.get("bass_output") || null,
    bass_channel: Number(fd.get("bass_channel")),
    tempo: Number(fd.get("tempo")),
    bars: Number(fd.get("bars")),
    time_signature: fd.get("time_signature"),
    key: keyVal ? String(keyVal).trim() || null : null,
    density: fd.get("density"),
    seed: seedVal ? Number(seedVal) : null,
    continuous: document.getElementById("continuous").checked,
    playback_loop: Number(fd.get("playback_loop")),
    count_in: Number(fd.get("count_in")),
  };
}

function updateButtons(state) {
  btnStart.disabled = state !== "idle" && state !== "stopped";
  btnImprovise.disabled = state !== "synced";
  btnStop.disabled = state === "idle" || state === "stopped";
  btnReset.disabled = state === "idle";
  setSettingsDisabled(state !== "idle" && state !== "stopped");
}

function beatsPerBar(timeSignature) {
  const select = document.getElementById("time-signature");
  const ts = timeSignature || select?.value || "4/4";
  if (ts === "6/8") return 2;
  const num = Number(ts.split("/")[0]);
  return Number.isFinite(num) && num > 0 ? num : 4;
}

function updateBeatTrack(beat, beatsInBar, visible) {
  if (!beatTrack) return;
  beatTrack.classList.toggle("visible", visible);
  const count = Math.min(beatTicks.length, beatsInBar);
  beatTicks.forEach((tick, i) => {
    tick.classList.toggle("active", visible && i < count && i + 1 === beat);
    tick.style.display = i < count ? "" : "none";
  });
}

function updateStatus(data) {
  if (!data) return;
  const state = data.state || "idle";
  statusState.textContent = formatState(state);
  statusDot.dataset.state = state;
  updateButtons(state);

  if (data.loop > 0) {
    statusPosition.hidden = false;
    posLoop.textContent = data.loop;
    posBar.textContent = data.bar;
    posBeat.textContent = data.beat;
    updateBeatTrack(data.beat, beatsPerBar(data.time_signature), true);
  } else {
    statusPosition.hidden = true;
    updateBeatTrack(0, 4, false);
  }

  if (data.key_label && data.state !== "capturing" && data.state !== "arming") {
    statusKey.hidden = false;
    keyLabel.textContent = data.key_label;
  } else {
    statusKey.hidden = true;
  }

  if (data.chord_progression?.length && data.state !== "capturing" && data.state !== "arming") {
    statusChords.hidden = false;
    statusChords.textContent = data.chord_progression.join("  ·  ");
  } else {
    statusChords.hidden = true;
    statusChords.textContent = "";
  }

  statusMessage.hidden = !data.message;
  if (data.message) statusMessage.textContent = data.message;

  statusError.hidden = !data.error;
  if (data.error) statusError.textContent = data.error;

  statusWarning.hidden = !data.capture_warning;
  if (data.capture_warning) statusWarning.textContent = data.capture_warning;
}

async function fetchStatus() {
  try {
    const res = await fetch("/api/session/status");
    if (!res.ok) return;
    updateStatus(await res.json());
  } catch (_) {}
}

async function postJson(url, body) {
  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    throw new Error(`Server unreachable — is logicians serve running? (${err.message})`);
  }
  let data;
  try {
    data = await res.json();
  } catch {
    throw new Error(`Bad response from server (${res.status})`);
  }
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    throw new Error(detail || res.statusText);
  }
  updateStatus(data);
  return data;
}

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(fetchStatus, 250);
}

btnStart.addEventListener("click", async () => {
  try {
    statusError.hidden = true;
    await postJson("/api/session/start", getConfig());
  } catch (err) {
    statusError.hidden = false;
    statusError.textContent = err.message;
  }
});

btnImprovise.addEventListener("click", async () => {
  try {
    statusError.hidden = true;
    await postJson("/api/session/improvise");
  } catch (err) {
    statusError.hidden = false;
    statusError.textContent = err.message;
  }
});

btnStop.addEventListener("click", async () => {
  try {
    await postJson("/api/session/stop");
  } catch (err) {
    statusError.hidden = false;
    statusError.textContent = err.message;
  }
});

btnReset.addEventListener("click", async () => {
  try {
    statusError.hidden = true;
    await postJson("/api/session/reset");
  } catch (err) {
    statusError.hidden = false;
    statusError.textContent = err.message;
  }
});

loadPorts();
loadDrumStyles();
fetchStatus();
startPolling();
