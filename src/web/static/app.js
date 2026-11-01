"use strict";
// InsurFlow interface: Chat (main), Chats (history) and Documents. All data comes from /api/*; nothing is hard-coded.
const $ = (s, r = document) => r.querySelector(s);
const view = $("#view");
const TYPE = {contract: "Contract", amendment: "Amendment", claim_form: "Claim", invoice: "Invoice", unknown: "Unknown"};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const money = (v) => `${Number(v).toLocaleString("en-US", {minimumFractionDigits: 2})} MAD`;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const store = {get: (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } }};

/* ---------- icons (thin outline) ---------- */
const PATHS = {
  plus: "M12 5v14 M5 12h14", chat: "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",
  history: "M3 12a9 9 0 1 0 3-6.7 M3 4v5h5 M12 7v5l3 2",
  folder: "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z",
  file: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6 M16 13H8 M16 17H8",
  image: "M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z M8.5 10a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3z M21 15l-5-5L5 21",
  clip: "M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48",
  up: "M12 19V5 M5 12l7-7 7 7", x: "M18 6L6 18 M6 6l12 12", chev: "M6 9l6 6 6-6",
  trash: "M3 6h18 M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6 M10 11v6 M14 11v6 M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2",
  eye: "M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
  search: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16z M21 21l-4.35-4.35",
  check: "M20 6L9 17l-5-5", alert: "M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z M12 9v4 M12 17h.01",
  spark: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z M19 16l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z",
  menu: "M3 12h18 M3 6h18 M3 18h18", side: "M3 5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z M9 3v18",
  pin: "M12 17v5 M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1z",
  pencil: "M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5z M15 5l4 4",
  doc: "M6 2h8l6 6v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z M14 2v6h6",
};
const ic = (n, size) => `<svg class="ic" viewBox="0 0 24 24" ${size ? `style="width:${size}px;height:${size}px"` : ""} aria-hidden="true">${PATHS[n].split(" M").map((d, i) => `<path d="${i ? "M" : ""}${d}"/>`).join("")}</svg>`;
const help = (tip, cls = "") => `<button type="button" class="help ${cls}" data-tip="${esc(tip)}" aria-label="${esc(tip)}">?</button>`;

/* ---------- api + feedback ---------- */
let csrf = "";  // per-session token, sent on every unsafe request
async function api(path, opts = {}) {
  const headers = {...(opts.headers || {})};
  if ((opts.method || "GET").toUpperCase() !== "GET") headers["X-CSRF-Token"] = csrf;
  const res = await fetch(path, {...opts, headers, credentials: "same-origin"});
  const body = await res.json().catch(() => ({}));
  if (res.status === 401) { authScreen("login"); throw new Error("Please sign in."); }
  if (!res.ok) throw new Error(body.error || "Something went wrong. Please try again.");
  return body;
}
const sendJson = (method) => (p, data) => api(p, {method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(data || {})});
const post = sendJson("POST"), patch = sendJson("PATCH"), del = (p) => api(p, {method: "DELETE"});
function toast(msg, type = "ok") {
  const t = document.createElement("div");
  t.className = `toast ${type}`; t.innerHTML = `${ic(type === "err" ? "alert" : "check", 16)}<span>${esc(msg)}</span>`;
  $("#toasts").append(t); setTimeout(() => t.remove(), 4200);
}
function confirmDialog(title, text, label) {
  return new Promise((resolve) => {
    const d = document.createElement("dialog");
    d.innerHTML = `<h2>${esc(title)}</h2><p class="muted">${esc(text)}</p><div class="dlg-actions"><button value="no">Cancel</button><button class="danger-fill" value="yes">${esc(label)}</button></div>`;
    let settled = false;  // answer straight from the click: do not depend on the dialog's close event
    const answer = (yes) => { if (settled) return; settled = true; resolve(yes); d.remove(); };
    d.addEventListener("close", () => answer(false));  // Escape closes the dialog without a choice
    d.querySelectorAll("button").forEach((b) => (b.onclick = () => answer(b.value === "yes")));
    document.body.append(d); d.showModal();
  });
}
function when(iso) {
  const d = new Date(iso), now = new Date(), days = Math.floor((new Date(now.toDateString()) - new Date(d.toDateString())) / 864e5);
  if (days === 0) return `Today, ${d.toLocaleTimeString("en-GB", {hour: "2-digit", minute: "2-digit"})}`;
  if (days === 1) return "Yesterday";
  return d.toLocaleDateString("en-GB", {day: "numeric", month: "short", year: d.getFullYear() === now.getFullYear() ? undefined : "numeric"});
}

/* ---------- document vocabulary ---------- */
const statusOf = (s) => ({pending: "reading", processing: "reading", processed: "ready", needs_review: "review", error: "failed"}[s] || "reading");
const STATUS_LABEL = {reading: "Reading…", ready: "Ready", review: "Needs review", failed: "Failed"};
const statusHtml = (d) => { const k = statusOf(d.status); return `<span class="status ${k}">${k === "reading" ? '<span class="spinner"></span>' : "<i></i>"}${STATUS_LABEL[k]}</span>`; };
const fileIcon = (d) => `<span class="file-ic">${ic(d.kind === "image" ? "image" : "file")}</span>`;
function detailsOf(d) {
  const m = d.meta || {}, bits = [];
  if (d.doc_type === "contract") bits.push(m.policy_number, m.end_date && `expires ${m.end_date}`);
  else if (d.doc_type === "amendment") bits.push(m.policy_number, m.annual_ceiling && `ceiling ${money(m.annual_ceiling)}`);
  else if (d.doc_type === "claim_form" || d.doc_type === "invoice") bits.push(m.reference, m.amount && money(m.amount));
  return bits.filter(Boolean).map(esc).join(" · ") || '<span class="muted">-</span>';
}
const FIELDS = {
  contract: [["Contract no.", "policy_number"], ["Effective date", "start_date"], ["Expiry date", "end_date"], ["Annual premium", "amount", 1], ["Annual ceiling", "annual_ceiling", 1]],
  amendment: [["Amendment no.", "reference"], ["Contract no.", "policy_number"], ["Effective date", "start_date"], ["New annual ceiling", "annual_ceiling", 1], ["Premium adjustment", "amount", 1]],
  claim_form: [["Claim no.", "reference"], ["Contract no.", "policy_number"], ["Date of care", "document_date"], ["Provider", "provider"], ["Claimed amount", "amount", 1]],
  invoice: [["Invoice no.", "reference"], ["Contract no.", "policy_number"], ["Invoice date", "document_date"], ["Provider", "provider"], ["Total due", "amount", 1]],
};
function fieldsHtml(d) {
  const spec = FIELDS[d.doc_type] || [], f = d.fields || {};
  if (!spec.length) return `<p class="muted">This document type could not be recognised, so no values were extracted.</p>`;
  const ai = new Set(d.ai_fields || []);  // values the rules could not read and the language model supplied
  const aiTag = (key) => (ai.has(key) ? ` <span class="tag ai" title="Read by the language model: please check it">AI</span>` : "");
  return `<dl class="kv">${spec.map(([label, key, isMoney]) => `<dt>${label}</dt><dd>${f[key] ? esc(isMoney ? money(f[key]) : f[key]) + aiTag(key) : '<span class="muted">Not found</span>'}</dd>`).join("")}</dl>
    ${(f.guarantees || []).length ? `<div class="section">Guarantees</div><div class="chips">${f.guarantees.map((g) => `<span class="tag accent">${esc(g)}</span>`).join("")}</div>` : ""}
    ${(f.exclusions || []).length ? `<div class="section">Exclusions</div><div class="chips">${f.exclusions.map((g) => `<span class="tag red">${esc(g)}</span>`).join("")}</div>` : ""}`;
}
const previewHtml = (d, page) => d.kind === "image" || /\.(png|jpe?g)$/i.test(d.file_name)
  ? `<img class="preview" alt="${esc(d.file_name)}" src="/api/documents/${d.id}/file">`
  : `<iframe class="preview" style="height:420px" title="${esc(d.file_name)}" src="/api/documents/${d.id}/file#${page ? `page=${page}&` : ""}toolbar=0"></iframe>`;

/* ---------- drawer ---------- */
const drawer = $("#drawer");
function openDrawer(title, body, foot = "") {
  drawer.innerHTML = `<div class="drawer-head"><h2>${esc(title)}</h2><button class="icon-btn" id="dclose" aria-label="Close">${ic("x")}</button></div><div class="drawer-body">${body}</div>${foot ? `<div class="drawer-foot">${foot}</div>` : ""}`;
  drawer.classList.add("open"); drawer.setAttribute("aria-hidden", "false");
  $("#dclose").onclick = closeDrawer;
}
function closeDrawer() {
  drawer.classList.remove("open"); drawer.setAttribute("aria-hidden", "true");
  if (location.hash.startsWith("#/documents/")) history.replaceState(null, "", "#/documents");
}
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

/* ---------- chat state ---------- */
const chat = {id: null, title: "", scope: null, messages: [], sending: false, attach: null, skipHash: false};
function resetChat(scope = null) { Object.assign(chat, {id: null, title: "", scope, messages: [], attach: null}); }

function md(text, i) {
  const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\[(\d+)\]/g, (_, n) => `<button class="cite" data-i="${i}" data-n="${n}" aria-label="Source ${n}">${n}</button>`);
  let out = "", list = false;
  for (const line of esc(text).split("\n")) {
    const li = /^- (.*)$/.exec(line);
    if (li) { if (!list) { out += "<ul>"; list = true; } out += `<li>${inline(li[1])}</li>`; }
    else { if (list) { out += "</ul>"; list = false; } if (line.trim()) out += `<p>${inline(line)}</p>`; }
  }
  return out + (list ? "</ul>" : "");
}
const mark = `<span class="mark">${ic("spark", 22)}</span>`;
function messageHtml(m, i) {
  if (m.role === "user") return `<div class="msg user"><div class="bubble">${esc(m.content)}</div></div>`;
  if (m.refused) return `<div class="msg">${mark}<div class="reply none"><p>I could not find this in your documents.</p><p class="hint">Try naming a contract or claim number, or choose a document below.</p></div></div>`;
  const cited = [...m.content.matchAll(/\[(\d+)\]/g)].map((x) => +x[1]);  // show the passages the answer relies on
  const seen = new Set();  // one chip per document page
  const shown = (cited.length ? m.sources.filter((s) => cited.includes(s.n)) : m.sources.slice(0, 3))
    .filter((s) => { const k = `${s.document_id}:${s.page}`; return seen.has(k) ? false : (seen.add(k), true); });
  const sources = shown.length ? `<details class="sources"><summary>${shown.length === 1 ? "Source" : `Sources (${shown.length})`}</summary><div class="chips">${shown.map((s) =>
    `<button class="src" data-i="${i}" data-n="${s.n}" title="${esc(s.file_name)}, page ${s.page}">${ic("file", 13)}<span>${esc(s.file_name)}</span><small>p.${s.page}</small></button>`).join("")}</div></details>` : "";
  return `<div class="msg">${mark}<div class="reply">${md(m.content, i)}${sources}</div></div>`;
}

/* ---------- chat page ---------- */
async function chatPage(id) {
  if (id && Number(id) !== chat.id) {
    try {
      const c = await api(`/api/conversations/${id}`);
      Object.assign(chat, {id: c.id, title: c.title, scope: c.document, messages: c.messages, attach: null});
    } catch (e) { toast("This conversation no longer exists.", "err"); resetChat(); history.replaceState(null, "", "#/chat"); }
  } else if (!id && chat.id) resetChat();
  view.innerHTML = `<section class="chat" id="chatroot">
    <div class="chat-top"><h1 id="ctitle"></h1></div>
    <div class="thread" id="thread"><div class="thread-inner" id="msgs"></div></div>
    <div class="dock">
      <div class="greet"><span aria-hidden="true">${ic("spark")}</span><h2>How can I help today?</h2></div>
      <form class="composer" id="form">
        <div class="files" id="attach"></div>
        <textarea id="q" rows="1" placeholder="Ask about your insurance documents" aria-label="Your question"></textarea>
        <div class="bar"><input type="file" id="file" accept="application/pdf,image/png,image/jpeg" hidden>
          <button type="button" class="tool" id="clip" aria-label="Import a PDF or an image" title="Import a PDF or an image">${ic("clip")}</button>
          <div class="scope" id="scope"></div><span class="grow"></span>
          <button class="send" id="sendbtn" aria-label="Send" disabled>${ic("up", 18)}</button></div>
      </form>
      <div class="pills">${["When does my contract expire?", "What is excluded from my coverage?", "What amount was claimed?"].map((t) => `<button type="button" data-q="${esc(t)}">${esc(t)}</button>`).join("")}</div>
      <p class="foot">Answers come only from your documents. ${help("Import a PDF or an image with the paperclip. Scans and photos are read automatically.", "up")}</p>
    </div></section>`;
  renderHead(); renderScope(); renderMessages(); renderAttach(); loadRecents();
  const q = $("#q"), btn = $("#sendbtn");
  const grow = () => { q.style.height = "auto"; q.style.height = `${Math.min(q.scrollHeight, 200)}px`; btn.disabled = !q.value.trim() || chat.sending; };
  const go = () => { const t = q.value; q.value = ""; grow(); submit(t); };
  q.addEventListener("input", grow);
  q.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (q.value.trim()) go(); } });
  $("#form").onsubmit = (e) => { e.preventDefault(); if (q.value.trim()) go(); };
  $("#clip").onclick = () => $("#file").click();
  $("#file").onchange = (e) => { if (e.target.files[0]) attachFile(e.target.files[0]); e.target.value = ""; };
  $("#chatroot").querySelectorAll(".pills button").forEach((b) => (b.onclick = () => submit(b.dataset.q)));
  q.focus();
}
function renderHead() { const t = $("#ctitle"); if (t) t.textContent = chat.messages.length ? chat.title : ""; }
function renderMessages() {
  const box = $("#msgs"); if (!box) return;
  box.innerHTML = chat.messages.map(messageHtml).join("");
  $("#chatroot").classList.toggle("empty", !chat.messages.length);
  bindSources(box);
  const t = $("#thread"); t.scrollTop = t.scrollHeight;
}
function bindSources(root) { root.querySelectorAll(".src, .cite").forEach((b) => (b.onclick = () => openSource(+b.dataset.i, +b.dataset.n))); }
function addMessage(m) {
  chat.messages.push(m);
  $("#chatroot").classList.remove("empty");
  $("#msgs").insertAdjacentHTML("beforeend", messageHtml(m, chat.messages.length - 1));
  bindSources($("#msgs").lastElementChild);
  const t = $("#thread"); t.scrollTop = t.scrollHeight;
}
const notice = (html, cls = "") => { $("#msgs").insertAdjacentHTML("beforeend", `<div class="notice ${cls}">${html}</div>`); const t = $("#thread"); t.scrollTop = t.scrollHeight; };

async function ensureConversation() {
  if (chat.id) return;
  const c = await post("/api/conversations", {document_id: chat.scope ? chat.scope.id : null});
  chat.id = c.id; chat.skipHash = true; history.replaceState(null, "", `#/chat/${c.id}`);
}
async function submit(text, retry = false) {
  text = text.trim();
  if (!text || chat.sending) return;
  if (chat.attach && ["uploading", "reading"].includes(chat.attach.phase)) return toast("Your document is still being read.", "err");
  chat.sending = true; $("#sendbtn").disabled = true; clearAttachIfDone();
  try { await ensureConversation(); } catch (e) { chat.sending = false; return toast(e.message, "err"); }
  if (!retry) addMessage({role: "user", content: text});
  $("#msgs").insertAdjacentHTML("beforeend", `<div class="msg" id="thinking">${mark}<div class="reply"><span class="thinking" aria-label="Thinking"><i></i><i></i><i></i></span></div></div>`);
  $("#thread").scrollTop = $("#thread").scrollHeight;
  try {
    const r = await post(`/api/conversations/${chat.id}/messages`, {question: text});
    $("#thinking")?.remove(); chat.title = r.title; renderHead(); addMessage(r.message); loadRecents();
    if (r.warning) toast(r.warning, "err");
  } catch (e) {
    $("#thinking")?.remove();
    notice(`${ic("alert", 16)}<span>${esc(e.message)}</span><button class="quiet" id="retry">Try again</button>`, "err");
    $("#retry").onclick = () => { $("#retry").closest(".notice").remove(); submit(text, true); };
  } finally { chat.sending = false; $("#q")?.focus(); }
}

/* sources */
async function openSource(i, n) {
  const m = chat.messages[i], s = m && m.sources[n - 1]; if (!s) return;
  const pdf = !/\.(png|jpe?g)$/i.test(s.file_name), url = `/api/documents/${s.document_id}/file`;
  const alive = (await fetch(url, {method: "HEAD"}).catch(() => ({ok: false}))).ok;
  openDrawer(s.file_name, `<div class="chips" style="margin-bottom:4px"><span class="tag">${esc(TYPE[s.doc_type] || s.doc_type)}</span>${s.page ? `<span class="tag">Page ${s.page}</span>` : ""}${s.contract ? `<span class="tag accent mono">${esc(s.contract)}</span>` : ""}</div>
    <div class="section">Passage used</div><div class="quote">${esc(s.text)}</div><div class="section">Original document</div>
    ${alive ? previewHtml({id: s.document_id, file_name: s.file_name, kind: pdf ? "pdf" : "image"}, s.page) : '<div class="alert">This document is no longer in your library.</div>'}`,
    alive ? `<a class="btn" href="#/documents/${s.document_id}">Open in Documents</a>` : "");
}

/* scope: the document the conversation reads */
function renderScope() {
  const el = $("#scope"); if (!el) return;
  const s = chat.scope;
  el.innerHTML = `<button type="button" class="chip ${s ? "active" : ""}" id="scopebtn" aria-haspopup="listbox">${ic(s ? "file" : "folder", 15)}<span class="name">${s ? esc(s.file_name) : "All documents"}</span>${ic("chev", 13)}</button>
    ${s ? `<button type="button" class="quiet chip-x" id="scopex" aria-label="Use all documents">${ic("x", 13)}</button>` : ""}
    ${help("The assistant reads only this document. Choose All documents to search your whole library.", "up left")}<div class="popover" id="pop" hidden></div>`;
  $("#scopebtn").onclick = togglePopover;
  const x = $("#scopex"); if (x) x.onclick = () => setScope(null);
}
async function togglePopover() {
  const pop = $("#pop"); if (!pop.hidden) { pop.hidden = true; return; }
  const docs = (await api("/api/documents")).filter((d) => ["processed", "needs_review"].includes(d.status));
  const draw = (q = "") => {
    const list = docs.filter((d) => d.file_name.toLowerCase().includes(q.toLowerCase()));
    $("#plist").innerHTML = `<li><button type="button" data-id="" class="${chat.scope ? "" : "on"}">${ic("folder", 16)}All documents</button></li>` + list.map((d) =>
      `<li><button type="button" data-id="${d.id}" class="${chat.scope && chat.scope.id === d.id ? "on" : ""}">${fileIcon(d)}<span style="min-width:0"><b style="display:block;font-weight:550;overflow:hidden;text-overflow:ellipsis">${esc(d.file_name)}</b><span class="sub">${esc(TYPE[d.doc_type])}${d.contract ? " · " + esc(d.contract) : ""}</span></span></button></li>`).join("");
    $("#plist").querySelectorAll("button").forEach((b) => (b.onclick = () => { pop.hidden = true; setScope(b.dataset.id ? docs.find((d) => d.id === +b.dataset.id) : null); }));
  };
  pop.innerHTML = `<label class="search">${ic("search")}<input type="search" id="psearch" placeholder="Find a document" aria-label="Find a document"></label><ul id="plist"></ul>`;
  pop.hidden = false; draw(); $("#psearch").oninput = (e) => draw(e.target.value); $("#psearch").focus();
  setTimeout(() => document.addEventListener("click", function off(e) { if (!pop.contains(e.target) && !$("#scopebtn")?.contains(e.target)) pop.hidden = true; if (pop.hidden) document.removeEventListener("click", off); }), 0);
}
async function setScope(doc) {
  chat.scope = doc ? {id: doc.id, file_name: doc.file_name, doc_type: doc.doc_type, status: doc.status} : null;
  if (chat.id) { try { await patch(`/api/conversations/${chat.id}`, {document_id: doc ? doc.id : null}); } catch (e) { toast(e.message, "err"); } }
  renderScope();
  if (chat.messages.length) notice(doc ? `Now using ${esc(doc.file_name)}` : "Now using all documents");
}

/* import from the chat */
function renderAttach() {
  const el = $("#attach"); if (!el) return; const a = chat.attach;
  if (!a) { el.innerHTML = ""; return; }
  const lead = {uploading: '<span class="spinner"></span>', reading: '<span class="spinner"></span>', ready: ic("check", 16).replace('class="ic"', 'class="ic ok"'), failed: ic("alert", 16)}[a.phase];
  const line = {uploading: "Uploading…", reading: "Reading the document…", ready: a.duplicate ? "Already in your library · ready" : "Ready", failed: a.message}[a.phase];
  el.innerHTML = `<div class="file-chip ${a.phase}">${lead}<div><b>${esc(a.name)}</b><small>${esc(line)}</small></div>${["ready", "failed"].includes(a.phase) ? `<button type="button" class="quiet icon-btn" id="adismiss" aria-label="Dismiss" style="width:26px;height:26px">${ic("x", 14)}</button>` : ""}</div>`;
  const d = $("#adismiss"); if (d) d.onclick = () => { chat.attach = null; renderAttach(); };
}
function clearAttachIfDone() { if (chat.attach && ["ready", "failed"].includes(chat.attach.phase)) { chat.attach = null; renderAttach(); } }
async function waitUntilRead(id) {
  for (let i = 0; i < 300; i++) {
    const d = await api(`/api/documents/${id}`);
    if (!["pending", "processing"].includes(d.status)) return d;
    await wait(1200);
  }
  throw new Error("Reading is taking too long.");
}
async function attachFile(file) {
  chat.attach = {phase: "uploading", name: file.name}; renderAttach();
  try {
    const fd = new FormData(); fd.append("files", file);
    const [r] = await api("/api/documents", {method: "POST", body: fd});
    if (r.error) throw new Error(r.error.startsWith("unsupported") ? "Not supported. Import a PDF, PNG or JPEG." : r.error);
    chat.attach = {phase: "reading", name: file.name}; renderAttach();
    const d = await waitUntilRead(r.id);
    if (d.status === "error") throw new Error("Could not be read. Try a clearer scan.");
    await setScope(d);
    chat.attach = {phase: "ready", name: d.file_name, duplicate: !r.is_new}; renderAttach(); loadRecents();
  } catch (e) { chat.attach = {phase: "failed", name: file.name, message: e.message}; renderAttach(); }
}

/* ---------- chat actions shared by the sidebar and the Chats page ---------- */
const refreshChats = () => loadRecents();
const pinIcon = (on) => (on ? ic("pin", 16).replace('class="ic"', 'class="ic fill"') : ic("pin", 16));
const chatActionsHtml = (c) => `<button class="quiet icon-btn ${c.pinned ? "on" : ""}" data-pin="${c.id}" data-pinned="${c.pinned ? 1 : 0}" aria-pressed="${c.pinned}" aria-label="${c.pinned ? "Unpin" : "Pin"} chat" title="${c.pinned ? "Unpin" : "Pin to the top"}">${pinIcon(c.pinned)}</button>
  <button class="quiet icon-btn" data-rename="${c.id}" aria-label="Rename chat" title="Rename">${ic("pencil", 16)}</button>
  <button class="quiet icon-btn" data-del="${c.id}" aria-label="Delete chat" title="Delete">${ic("trash", 16)}</button>`;

async function pinChat(id, pin) {
  try { await patch(`/api/conversations/${id}`, {pinned: pin}); toast(pin ? "Pinned to the top" : "Unpinned"); refreshChats(); }
  catch (err) { toast(err.message, "err"); }
}
async function deleteChat(id) {
  if (!(await confirmDialog("Delete this chat?", "The messages will be removed. Your documents are not affected.", "Delete"))) return;
  try {
    await del(`/api/conversations/${id}`);
    const wasOpen = chat.id === id;
    if (wasOpen) resetChat();
    toast("Chat deleted"); refreshChats();
    if (wasOpen && location.hash.startsWith("#/chat/")) location.hash = "#/chat";
  } catch (err) { toast(err.message, "err"); }
}
/* Replace a title with an input: Enter saves, Escape cancels, leaving the field saves a changed title. */
function startRename(titleEl, id) {
  const current = titleEl.textContent.trim(), input = document.createElement("input");
  input.className = "rename"; input.value = current; input.maxLength = 120; input.setAttribute("aria-label", "Chat title");
  titleEl.replaceWith(input); input.focus(); input.select();
  let done = false;
  const finish = async (save) => {
    if (done) return; done = true;
    const name = input.value.trim();
    if (save && name && name !== current) {
      try { await patch(`/api/conversations/${id}`, {title: name}); if (chat.id === id) chat.title = name; toast("Chat renamed"); }
      catch (err) { toast(err.message, "err"); }
    }
    refreshChats();
  };
  input.onclick = (e) => { e.preventDefault(); e.stopPropagation(); };
  input.onkeydown = (e) => { e.stopPropagation(); if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); };
  input.onblur = () => finish(true);
}
function bindChatActions(root) {
  root.querySelectorAll("[data-pin]").forEach((b) => (b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); pinChat(+b.dataset.pin, b.dataset.pinned !== "1"); }));
  root.querySelectorAll("[data-rename]").forEach((b) => (b.onclick = (e) => {
    e.preventDefault(); e.stopPropagation();
    const row = b.closest("[data-id]");
    startRename(row.querySelector(".title"), +b.dataset.rename);
  }));
  root.querySelectorAll("[data-del]").forEach((b) => (b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); deleteChat(+b.dataset.del); }));
}

/* ---------- chats (history) page ---------- */
/* ---------- documents page ---------- */
const filters = {q: "", type: ""};
let docTimer = null, flashId = null;
async function documentsPage(arg) {
  view.innerHTML = `<div class="page" id="docpage"><div class="page-head"><div><h1>Documents</h1><p>Everything you imported, read automatically.</p></div>
    <div class="tools"><label class="search">${ic("search")}<input type="search" id="dq" placeholder="Search documents" value="${esc(filters.q)}" aria-label="Search documents"></label>
      <select id="dtype" aria-label="Type"><option value="">All types</option>${["contract", "amendment", "claim_form", "invoice"].map((t) => `<option value="${t}" ${filters.type === t ? "selected" : ""}>${TYPE[t]}</option>`).join("")}</select>
      <button class="btn primary" id="import">${ic("plus", 16)} Import</button><input type="file" id="dfile" accept="application/pdf,image/png,image/jpeg" multiple hidden></div></div>
    <table><thead><tr><th>Name</th><th class="hide-sm">Details ${help("Key values read from the document: contract number, dates, amounts.")}</th><th class="hide-md">Type</th><th class="hide-md">Imported</th><th class="hide-md">By</th>
      <th>Status ${help("Needs review means a value looks inconsistent, for example a date outside the contract period.", "right")}</th><th></th></tr></thead><tbody id="dbody">
      ${'<tr class="skeleton"><td><div style="width:70%"></div></td><td><div></div></td><td class="hide-md"><div></div></td><td class="hide-md"><div></div></td><td class="hide-md"><div></div></td><td><div></div></td><td></td></tr>'.repeat(4)}</tbody></table></div>`;
  let t; $("#dq").oninput = (e) => { clearTimeout(t); t = setTimeout(() => { filters.q = e.target.value; loadDocs(); }, 250); };
  $("#dtype").onchange = (e) => { filters.type = e.target.value; loadDocs(); };
  $("#import").onclick = () => $("#dfile").click();
  $("#dfile").onchange = (e) => { importFiles([...e.target.files]); e.target.value = ""; };
  await loadDocs();
  if (arg) openDocument(arg);
}
async function loadDocs() {
  clearTimeout(docTimer);
  if (!$("#dbody")) return;
  const qs = new URLSearchParams(Object.entries(filters).filter(([, v]) => v));
  let docs;
  try { docs = await api(`/api/documents?${qs}`); } catch (e) { $("#dbody").innerHTML = `<tr><td colspan="7"><div class="empty"><b>Could not load documents</b>${esc(e.message)}</div></td></tr>`; return; }
  const filtered = filters.q || filters.type;
  $("#dbody").innerHTML = docs.length ? docs.map((d) => {
    const k = statusOf(d.status);
    return `<tr class="row ${d.id === flashId ? "flash" : ""}" data-id="${d.id}"><td><div class="cell-file">${fileIcon(d)}<div><b>${esc(d.file_name)}</b><small class="only-sm">${detailsOf(d)}</small>${d.scanned ? `<small>Scanned or photo ${help("This file is an image. Its text was recognised automatically.", "left")}</small>` : ""}</div></div></td>
    <td class="nw hide-sm">${detailsOf(d)}</td><td class="hide-md type">${esc(TYPE[d.doc_type] || d.doc_type)}</td><td class="hide-md muted nw">${when(d.uploaded_at)}</td><td class="hide-md muted">${esc(d.owner)}</td><td>${statusHtml(d)}</td>
    <td><div class="row-actions"><button class="icon-btn" data-ask="${d.id}" aria-label="Ask about ${esc(d.file_name)}" title="Ask about this document" ${k === "reading" || k === "failed" ? "disabled" : ""}>${ic("chat", 16)}</button>
      <button class="icon-btn" data-open="${d.id}" aria-label="View ${esc(d.file_name)}" title="View">${ic("eye", 16)}</button>
      <button class="icon-btn" data-del="${d.id}" aria-label="Delete ${esc(d.file_name)}" title="Delete">${ic("trash", 16)}</button></div></td></tr>`;
  }).join("")
    : `<tr><td colspan="7"><div class="empty">${ic("folder")}<b>${filtered ? "No document matches" : "No documents yet"}</b>${filtered ? "Try another search." : "Import a PDF, a scan or a photo to get started."}
       ${filtered ? "" : '<br><br><button class="btn" id="samples">Load sample documents</button>'}</div></td></tr>`;
  flashId = null;
  const body = $("#dbody");
  body.querySelectorAll("tr.row").forEach((tr) => (tr.onclick = () => (location.hash = `#/documents/${tr.dataset.id}`)));
  body.querySelectorAll("[data-open]").forEach((b) => (b.onclick = (e) => { e.stopPropagation(); location.hash = `#/documents/${b.dataset.open}`; }));
  body.querySelectorAll("[data-ask]").forEach((b) => (b.onclick = (e) => { e.stopPropagation(); askAbout(docs.find((d) => d.id === +b.dataset.ask)); }));
  body.querySelectorAll("[data-del]").forEach((b) => (b.onclick = (e) => { e.stopPropagation(); removeDocument(docs.find((d) => d.id === +b.dataset.del)); }));
  const samples = $("#samples"); if (samples) samples.onclick = async () => { await post("/api/documents/samples"); toast("Sample documents are being read"); loadDocs(); };
  if (docs.some((d) => statusOf(d.status) === "reading")) docTimer = setTimeout(loadDocs, 2000);
}
async function importFiles(files) {
  if (!files.length) return;
  const fd = new FormData(); files.forEach((f) => fd.append("files", f));
  try {
    const results = await api("/api/documents", {method: "POST", body: fd});
    results.forEach((r) => {
      if (r.error) toast(`${r.file_name}: ${r.error.startsWith("unsupported") ? "not supported. Import a PDF, PNG or JPEG." : r.error}`, "err");
      else if (!r.is_new) { toast(`${r.file_name} is already in your library.`); flashId = r.id; }
      else toast(`${r.file_name} imported. Reading it now…`);
    });
  } catch (e) { toast(e.message, "err"); }
  loadDocs();
}
function askAbout(doc) {
  resetChat({id: doc.id, file_name: doc.file_name, doc_type: doc.doc_type, status: doc.status});
  const same = location.hash === "#/chat";
  location.hash = "#/chat";
  if (same) route();
}
async function removeDocument(doc) {
  if (!(await confirmDialog(`Delete ${doc.file_name}?`, "The file and the information read from it will be removed. Chats that cited it keep their answers.", "Delete"))) return;
  try { await del(`/api/documents/${doc.id}`); toast("Document deleted"); closeDrawer(); if (chat.scope && chat.scope.id === doc.id) chat.scope = null; loadDocs(); }
  catch (e) { toast(e.message, "err"); }
}
async function openDocument(id) {
  try {
    const d = await api(`/api/documents/${id}`), k = statusOf(d.status);
    const alerts = d.status === "error" ? [d.error || "This document could not be read."] : d.alerts;
    openDrawer(d.file_name, `<div class="chips" style="margin-bottom:16px;align-items:center"><span class="tag">${esc(TYPE[d.doc_type] || d.doc_type)}</span>${statusHtml(d)}<span class="muted small">Imported ${esc(when(d.uploaded_at))} by ${esc(d.owner)}</span></div>
      ${previewHtml(d)}
      ${alerts.length ? `<div class="section">To check</div>${alerts.map((a) => `<div class="alert ${d.status === "error" ? "bad" : ""}">${ic("alert", 16)}<span>${esc(a)}</span></div>`).join("")}` : k === "ready" ? `<div class="section">Checks</div><div class="ok-note">${ic("check", 16)}<span>Everything looks consistent.</span></div>` : ""}
      ${d.status === "error" ? "" : `<div class="section">Information read</div>${fieldsHtml(d)}`}
      ${d.related.length ? `<div class="section">Same contract (${esc(d.contract)})</div>${d.related.map((r) => `<div style="margin:4px 0"><a href="#/documents/${r.id}">${esc(r.file_name)}</a> <span class="muted small">${esc(TYPE[r.doc_type])}</span></div>`).join("")}` : ""}`,
      `<button class="btn primary" id="dask" ${k === "reading" || k === "failed" ? "disabled" : ""}>${ic("chat", 16)} Ask about this document</button><button class="btn danger" id="ddel">${ic("trash", 16)} Delete</button>`);
    $("#dask").onclick = () => { closeDrawer(); askAbout(d); };
    $("#ddel").onclick = () => removeDocument(d);
  } catch (e) { toast("This document no longer exists.", "err"); history.replaceState(null, "", "#/documents"); }
}

/* ---------- shell + router ---------- */
const brandHtml = `${ic("doc", 20)}InsurFlow`;
$("#brand").innerHTML = brandHtml; $("#mbrand").innerHTML = brandHtml;
$("#nav").innerHTML = `<button class="item new" id="new-chat"><span class="plus">${ic("plus", 14)}</span>New chat</button>
  <a href="#/documents" data-page="documents">${ic("folder")}<span>Documents</span></a>`;
$("#menu").innerHTML = ic("menu"); $("#collapse").innerHTML = ic("side"); $("#reopen").innerHTML = ic("side");
$("#new-chat").onclick = () => { resetChat(); if (location.hash === "#/chat") route(); else location.hash = "#/chat"; };
const mobile = () => matchMedia("(max-width:900px)").matches;
const closeMenu = () => { $("#sidebar").classList.remove("open"); $("#scrim").classList.remove("show"); };
$("#menu").onclick = () => { $("#sidebar").classList.add("open"); $("#scrim").classList.add("show"); };
$("#scrim").onclick = closeMenu;
const setCollapsed = (on) => { document.body.classList.toggle("collapsed", on); store.set("sidebar", on ? "closed" : "open"); };
$("#collapse").onclick = () => (mobile() ? closeMenu() : setCollapsed(true));
$("#reopen").onclick = () => setCollapsed(false);
if (store.get("sidebar") === "closed") document.body.classList.add("collapsed");

/* ---------- history (sidebar): search + pin / rename / delete ---------- */
let recentsQuery = "";
function mountHistory() {
  $("#recents").innerHTML = `<h3>History ${help("Every conversation is saved. Open one to continue it; the assistant remembers what you discussed. Hover a chat to pin, rename or delete it.", "left")}</h3>
    <label class="search side-search">${ic("search")}<input type="search" id="rq" placeholder="Search history" aria-label="Search history"></label><div class="rlist" id="rlist"></div>`;
  let t; $("#rq").oninput = (e) => { clearTimeout(t); t = setTimeout(() => { recentsQuery = e.target.value; loadRecents(); }, 200); };
}
async function loadRecents() {
  const box = $("#rlist"); if (!box) return;
  try {
    const items = await api(`/api/conversations?q=${encodeURIComponent(recentsQuery)}`);
    box.innerHTML = items.length ? items.map((c) => `<div class="recent ${c.id === chat.id ? "active" : ""}" data-id="${c.id}">
      <a href="#/chat/${c.id}" title="${esc(c.title)} · ${esc(when(c.updated_at))}">${c.pinned ? ic("pin", 12).replace('class="ic"', 'class="ic fill"') : ""}<span class="title">${esc(c.title)}</span></a>
      <div class="r-actions">${chatActionsHtml(c)}</div></div>`).join("")
      : `<p class="rempty">${recentsQuery ? "No chat matches" : "No chats yet"}</p>`;
    bindChatActions(box);
  } catch (e) { /* the history list is optional */ }
}

async function route() {
  const [, page = "chat", arg] = location.hash.split("/");
  const name = ["chat", "documents"].includes(page) ? page : "chat";
  clearTimeout(docTimer); closeMenu();
  if (name !== "documents") drawer.classList.remove("open");
  document.title = `${{chat: "Chat", documents: "Documents"}[name]} · InsurFlow`;
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === name));
  try {
    if (name === "chat") await chatPage(arg);
    else if ($("#docpage") && !arg) { closeDrawer(); await loadDocs(); }
    else if ($("#docpage") && arg) await openDocument(arg);
    else await documentsPage(arg);
  } catch (e) { view.innerHTML = `<div class="page"><div class="empty"><b>Something went wrong</b>${esc(e.message)}</div></div>`; }
  if (name !== "chat") loadRecents();
}
window.addEventListener("hashchange", () => { if (chat.skipHash) { chat.skipHash = false; return; } route(); });

/* drop a file anywhere to import it (into the chat, or into the library when on Documents) */
window.addEventListener("dragover", (e) => { e.preventDefault(); view.classList.add("dropzone-outline"); });
window.addEventListener("dragleave", (e) => { if (!e.relatedTarget) view.classList.remove("dropzone-outline"); });
window.addEventListener("drop", (e) => {
  e.preventDefault(); view.classList.remove("dropzone-outline");
  const files = [...(e.dataTransfer?.files || [])]; if (!files.length) return;
  if ($("#chatroot")) attachFile(files[0]); else if ($("#docpage")) importFiles(files);
});
/* ---------- accounts: sign in / sign up, account menu, memory ---------- */
let currentUser = null, allowRegistration = false;
const initials = (name) => name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0].toUpperCase()).join("") || "?";

function authScreen(mode) {
  document.body.classList.add("signed-out"); closeDrawer(); drawer.classList.remove("open");
  const reg = mode === "register";
  $("#auth").innerHTML = `<form class="auth-card" id="authform" novalidate>
    <div class="brand auth-brand">${ic("doc", 20)}InsurFlow</div>
    <h1>${reg ? "Create your account" : "Welcome back"}</h1>
    <p class="muted">${reg ? "Your documents and chats stay private to your account." : "Sign in to your documents."}</p>
    ${reg ? `<label>Name<input name="name" autocomplete="name" maxlength="80" required></label>` : ""}
    <label>Email<input name="email" type="email" autocomplete="email" maxlength="254" required></label>
    <label>Password<input name="password" type="password" autocomplete="${reg ? "new-password" : "current-password"}" maxlength="200" required></label>
    ${reg ? '<small class="muted">At least 10 characters.</small>' : ""}
    <p class="auth-error" id="autherr" role="alert"></p>
    <button class="btn primary wide" type="submit">${reg ? "Create account" : "Sign in"}</button>
    ${allowRegistration ? `<p class="switch">${reg ? "Already have an account?" : "New here?"} <a href="#" id="switchmode">${reg ? "Sign in" : "Create an account"}</a></p>` : ""}</form>`;
  const form = $("#authform"); form.querySelector("input").focus();
  const sw = $("#switchmode"); if (sw) sw.onclick = (e) => { e.preventDefault(); authScreen(reg ? "login" : "register"); };
  form.onsubmit = async (e) => {
    e.preventDefault();
    const btn = form.querySelector("button[type=submit]"); btn.disabled = true; $("#autherr").textContent = "";
    try {
      const res = await fetch(`/api/auth/${reg ? "register" : "login"}`, {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(Object.fromEntries(new FormData(form)))});
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.error || "Could not sign in. Please try again.");
      signedIn(body);
    } catch (err) { $("#autherr").textContent = err.message; btn.disabled = false; }
  };
}
function signedIn(data) {
  csrf = data.csrf; currentUser = data.user; allowRegistration = !!data.allow_registration;
  document.body.classList.remove("signed-out"); $("#auth").innerHTML = "";
  renderAccount(); resetChat(); mountHistory();
  if (!location.hash) history.replaceState(null, "", "#/chat");
  route();
}
async function signOut() {
  try { await post("/api/auth/logout"); } catch (e) { /* the session may already be gone */ }
  csrf = ""; currentUser = null; resetChat(); view.innerHTML = ""; history.replaceState(null, "", "#/chat");
  authScreen("login");
}
function passwordDialog(title, text, label) {
  return new Promise((resolve) => {
    const d = document.createElement("dialog");
    d.innerHTML = `<form method="dialog"><h2>${esc(title)}</h2><p class="muted">${esc(text)}</p>
      <label class="field">Password<input type="password" name="pw" autocomplete="current-password" required></label>
      <div class="dlg-actions"><button type="button" value="no">Cancel</button><button class="danger-fill" value="yes">${esc(label)}</button></div></form>`;
    let settled = false;
    const done = (value) => { if (settled) return; settled = true; resolve(value); d.remove(); };
    d.addEventListener("close", () => done(null));
    d.querySelector("[value=no]").onclick = () => done(null);
    d.querySelector("form").onsubmit = (e) => { e.preventDefault(); done(d.querySelector("input").value || null); };
    document.body.append(d); d.showModal(); d.querySelector("input").focus();
  });
}
function renderAccount() {
  const u = currentUser;
  $("#account").innerHTML = `<button class="acct" id="acctbtn" aria-haspopup="menu" aria-expanded="false"><span class="avatar-s">${esc(initials(u.name))}</span>
    <span class="acct-name"><b>${esc(u.name)}</b><small>${esc(u.email)}</small></span>${ic("chev", 14)}</button>
    <div class="acct-menu" id="acctmenu" role="menu" hidden>
      <button role="menuitem" id="mem">${ic("spark", 16)} Memory</button>
      <button role="menuitem" id="logout">${ic("x", 16)} Sign out</button>
      <button role="menuitem" id="delacct" class="danger">${ic("trash", 16)} Delete account</button></div>`;
  const menu = $("#acctmenu"), btn = $("#acctbtn");
  btn.onclick = (e) => {
    e.stopPropagation(); menu.hidden = !menu.hidden; btn.setAttribute("aria-expanded", String(!menu.hidden));
    if (!menu.hidden) setTimeout(() => document.addEventListener("click", function off() { menu.hidden = true; document.removeEventListener("click", off); }, {once: true}), 0);
  };
  $("#logout").onclick = signOut;
  $("#mem").onclick = openMemory;
  $("#delacct").onclick = async () => {
    const pw = await passwordDialog("Delete your account?", "Your documents, files, chats and memory will be permanently removed. This cannot be undone.", "Delete everything");
    if (!pw) return;
    try { await api("/api/auth/me", {method: "DELETE", headers: {"Content-Type": "application/json"}, body: JSON.stringify({password: pw})}); toast("Account deleted"); csrf = ""; currentUser = null; resetChat(); view.innerHTML = ""; authScreen("login"); }
    catch (err) { toast(err.message, "err"); }
  };
}
async function openMemory() {
  const data = await api("/api/memories");
  const rows = data.items.map((m) => `<div class="mem"><span class="tag ${m.kind === "semantic" ? "accent" : ""}">${m.kind === "semantic" ? "About you" : "Asked"}</span><span class="mem-text">${esc(m.content)}</span>
    <button class="quiet icon-btn" data-mem="${m.id}" aria-label="Forget this">${ic("trash", 16)}</button></div>`).join("");
  openDrawer("Memory", `<label class="switch-row"><span><b>Remember across chats</b> ${help("Short notes about you and what you asked, kept to help in future chats. They are never used as a source for answers. Turn this off to stop saving and using them.", "left")}</span>
      <input type="checkbox" id="memtoggle" role="switch" ${data.enabled ? "checked" : ""}></label>
    <div class="section">Saved notes (${data.items.length})</div>${rows || '<p class="muted">Nothing saved yet. Notes appear here after you ask questions.</p>'}`,
    data.items.length ? `<button class="btn danger" id="memclear">${ic("trash", 16)} Forget everything</button>` : "");
  $("#memtoggle").onchange = async (e) => { try { await patch("/api/memories", {enabled: e.target.checked}); toast(e.target.checked ? "Memory is on" : "Memory is off"); } catch (err) { toast(err.message, "err"); e.target.checked = !e.target.checked; } };
  drawer.querySelectorAll("[data-mem]").forEach((b) => (b.onclick = async () => { await del(`/api/memories/${b.dataset.mem}`); openMemory(); }));
  const clear = $("#memclear"); if (clear) clear.onclick = async () => { if (await confirmDialog("Forget everything?", "All saved notes will be deleted.", "Forget")) { await del("/api/memories"); toast("Memory cleared"); openMemory(); } };
}

async function startApp() {
  const me = await fetch("/api/auth/me", {credentials: "same-origin"}).then((r) => r.json());
  allowRegistration = !!me.allow_registration;
  if (me.authenticated) signedIn(me); else authScreen("login");
}

startApp();
