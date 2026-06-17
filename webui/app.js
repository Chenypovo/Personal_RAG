const thread = document.getElementById("thread");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusEl = document.querySelector(".status");
const statusText = document.getElementById("status-text");
const memList = document.getElementById("mem-list");
const memCount = document.getElementById("mem-count");

let knownMemIds = new Set();

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function scrollDown() {
  thread.scrollTop = thread.scrollHeight;
}

function addMessage(role, innerHtml) {
  const wrap = document.createElement("div");
  wrap.className = `msg msg--${role}`;
  const avatar = role === "agent" ? "記" : "你";
  wrap.innerHTML = `<div class="avatar">${avatar}</div><div class="bubble">${innerHtml}</div>`;
  thread.appendChild(wrap);
  scrollDown();
  return wrap;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg msg--agent typing";
  wrap.id = "typing";
  wrap.innerHTML = `<div class="avatar">記</div><div class="bubble"><span></span><span></span><span></span></div>`;
  thread.appendChild(wrap);
  scrollDown();
  return wrap;
}

function removeTyping() {
  const t = document.getElementById("typing");
  if (t) t.remove();
}

function buildMeta(data) {
  const parts = [];

  if (Array.isArray(data.recalled_memories) && data.recalled_memories.length) {
    const chips = data.recalled_memories
      .map((m) => `<span class="chip">${escapeHtml(m.fact_content)}</span>`)
      .join("");
    parts.push(`<div class="meta-row recall"><span class="tag">想起关于你</span><div class="chips">${chips}</div></div>`);
  }

  if (Array.isArray(data.sources) && data.sources.length) {
    const chips = data.sources
      .map((s) => `<span class="chip">${escapeHtml(s.citation || s.source || "source")}</span>`)
      .join("");
    parts.push(`<div class="meta-row"><span class="tag">依据</span><div class="chips">${chips}</div></div>`);
  }

  const applied = (data.memory_ops || []).filter((o) => o.applied);
  if (applied.length) {
    const txt = applied
      .map((o) => `${o.type === "add" ? "记住" : o.type === "update" ? "更新" : "遗忘"}：${escapeHtml(o.fact_content || o.id)}`)
      .join("；");
    parts.push(`<div class="meta-row"><span class="tag">记忆变更</span><span>${txt}</span></div>`);
  }

  return parts.length ? `<div class="meta">${parts.join("")}</div>` : "";
}

function renderMemories(memories) {
  memCount.textContent = memories.length;
  if (!memories.length) {
    memList.innerHTML = `<li class="mem-empty">还没有记住任何事。<br/>说点关于你自己的吧。</li>`;
    return;
  }
  memList.innerHTML = memories
    .map((m) => {
      const fresh = !knownMemIds.has(m.id) && knownMemIds.size > 0 ? " fresh" : "";
      const obj = m.fact_object ? `<div class="mem-obj">${escapeHtml(m.fact_object)}</div>` : "";
      return `<li class="mem-item${fresh}">${obj}<div class="mem-content">${escapeHtml(m.fact_content)}</div></li>`;
    })
    .join("");
  knownMemIds = new Set(memories.map((m) => m.id));
}

async function loadMemories() {
  try {
    const r = await fetch("/api/memories");
    const data = await r.json();
    renderMemories(data.memories || []);
  } catch (e) {
    /* ignore on first load */
  }
}

function setBusy(busy) {
  sendBtn.disabled = busy;
  statusEl.classList.toggle("busy", busy);
  statusText.textContent = busy ? "思考中" : "就绪";
}

async function send(text) {
  addMessage("user", `<p>${escapeHtml(text)}</p>`);
  setBusy(true);
  addTyping();

  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();

    removeTyping();
    const answer = escapeHtml(data.answer || "").replace(/\n/g, "<br/>");
    addMessage("agent", `<p>${answer}</p>${buildMeta(data)}`);

    await loadMemories();
  } catch (e) {
    removeTyping();
    addMessage("agent", `<p class="muted">出错了：${escapeHtml(e.message)}。检查后端是否在运行、.env 是否配置。</p>`);
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.style.height = "auto";
  send(text);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
});

loadMemories();
input.focus();
