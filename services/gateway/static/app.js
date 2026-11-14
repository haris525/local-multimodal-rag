"use strict";

const MAX_SELECTED = 10;
const TYPE_LABEL = { pdf: "PDF", spreadsheet: "SHEET", text: "TEXT", image: "IMAGE" };
const ROUTE_LABEL = {
  sql: "SQL", filter: "SQL filter + search", search: "hybrid search", deep: "deep search", exhaustive: "read every section",
  memory_save: "saved to memory", memory_recall: "from memory", memory_forget: "memory updated",
  history: "from this chat", assistant: "about the assistant", no_documents: "no documents selected",
};
// Turns answered without searching documents: the footer shows no source count for these.
const NO_SEARCH_ROUTES = new Set(["memory_save", "memory_recall", "memory_forget", "history", "assistant", "no_documents"]);

const $ = (sel) => document.querySelector(sel);
const state = {
  docs: [],
  selected: new Set(),  // loaded per user in init()
  me: null,  // the signed-in user (GET /api/auth/session)
  leaving: false,  // on the way to the sign-in page
  threadId: null,
  threads: [],
  busy: false,  // an answer is streaming into the open chat: the chat cannot be switched, started or deleted
  view: 0,  // which chat the page shows; a chat that loads after another was opened is dropped
  pollTimer: null,
  reindexing: new Set(),  // documents whose reindex request is on its way: their Retry and Reindex wait
};

function load(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function save(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) el.append(child);
  }
  return el;
}

/* ---------- API access: CSRF header, sign-in redirect ---------- */

const CSRF = { "X-Requested-With": "local-rag" };

// Every API call sends the header the gateway requires for cookie requests; a 401 means the session ended.
// keep401: the caller handles 401 itself (a wrong current password in the account forms).
async function apiFetch(path, options = {}, { keep401 = false } = {}) {
  const response = await fetch(`/api${path}`, { ...options, headers: { ...CSRF, ...(options.headers || {}) } });
  if (response.status === 401 && !keep401) {
    toLogin();
    throw new Error("signed out");
  }
  return response;
}

function toLogin() {
  if (state.leaving) return;
  state.leaving = true;
  const owner = state.me?.user_id ? `&for=${state.me.user_id}` : "";  // the link goes back only to this user
  location.replace(`/login.html${location.hash ? "#next=" + encodeURIComponent(location.hash) + owner : ""}`);
}

const selKey = () => `selected:${state.me.user_id}`;  // document selection, per user on a shared browser

async function api(path, options = {}, flags = {}) {
  const response = await apiFetch(path, options, flags);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* not JSON */ }
    if (response.status === 403 && detail === "password change required") toLogin();
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.status === 204 ? null : response.json();
}

function markdown(text) {
  // Answers can echo document text: no forms, inputs or inline styles, even when sanitized.
  return DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: false }), {
    FORBID_TAGS: ["form", "input", "button", "textarea", "select", "style"],
    FORBID_ATTR: ["style"],
  });
}

/* ---------- service status ---------- */

async function refreshStatus() {
  const box = $("#status");
  try {
    const response = await apiFetch("/ready");
    const body = await response.json();
    const down = Object.entries(body.checks || {}).filter(([, ok]) => !ok).map(([name]) => name);
    box.className = `status ${down.length ? "bad" : "ok"}`;
    $("#status-text").textContent = down.length ? `waiting for: ${down.join(", ")}` : "all services ready";
  } catch {
    box.className = "status bad";
    $("#status-text").textContent = "gateway unreachable";
  }
  setTimeout(refreshStatus, 10000);
}

/* ---------- documents ---------- */

async function refreshDocs() {
  clearTimeout(state.pollTimer);
  try {
    state.docs = await api("/documents");
    const known = new Set(state.docs.filter(usable).map((d) => d.doc_id));
    for (const id of [...state.selected]) if (!known.has(id)) state.selected.delete(id);
    renderDocs();
    settleUploadMsg();
  } catch (err) {
    $("#upload-msg").textContent = `Could not load documents: ${err.message}`;
  }
  const pending = state.docs.some((d) => d.status === "queued" || d.status === "processing");
  state.pollTimer = setTimeout(refreshDocs, pending ? 2000 : 15000);
}

function renderDocs() {
  const list = $("#docs");
  list.replaceChildren();
  if (!state.docs.length) {
    list.append(h("li", { class: "muted" }, "No documents yet."));
  }
  for (const doc of state.docs) {
    const ready = usable(doc);
    const selected = state.selected.has(doc.doc_id);
    const full = state.selected.size >= MAX_SELECTED && !selected;
    const note = doc.status === "failed" ? doc.error : doc.status === "processing" ? doc.progress : doc.warnings.join("; ");
    list.append(
      h("li", { class: `doc${selected ? " selected" : ""}` },
        h("input", {
          type: "checkbox", checked: selected, disabled: !ready || full,
          title: !ready ? "Available when processing finishes" : full ? `At most ${MAX_SELECTED} documents` : "Use in chat",
          onchange: (e) => toggleSelect(doc.doc_id, e.target.checked),
        }),
        h("div", {},
          h("div", { class: "doc-name" }, doc.name),
          h("div", { class: "doc-meta" },
            h("span", { class: "badge" }, TYPE_LABEL[doc.doc_type] || doc.doc_type),
            h("span", { class: `pill ${doc.status}` }, doc.status),
            ready ? h("span", {}, `${doc.n_chunks} chunks${doc.tables.length ? `, ${doc.tables.length} tables` : ""}${doc.status !== "ready" ? " (previous index)" : ""}`) : null),
          note ? h("div", { class: `doc-note${doc.status === "failed" ? " err" : ""}` }, note,
            doc.status === "failed" ? h("button", { class: "link retry", type: "button", onclick: () => reindex(doc),
              disabled: state.reindexing.has(doc.doc_id), title: "Process this document again" }, "Retry") : null) : null),
        h("div", { class: "doc-actions" },
          h("button", { class: "icon-btn", type: "button", "data-icon": "refresh", onclick: () => reindex(doc),
            disabled: busyDoc(doc) || state.reindexing.has(doc.doc_id),
            title: busyDoc(doc) ? "Reindex is available once processing has finished" : "Reindex",
            "aria-label": "Reindex" }),
          h("button", { class: "icon-btn danger", type: "button", title: "Delete", "aria-label": "Delete", "data-icon": "trash", onclick: () => removeDoc(doc) }))));
  }
  $("#sel-count").textContent = `${state.selected.size} of ${MAX_SELECTED} selected`;
  if (state.me) save(selKey(), [...state.selected]);
  updateSend();
}

// Waiting or being processed: a reindex is refused (409) until it is ready or failed.
function busyDoc(doc) {
  return doc.status === "queued" || doc.status === "processing";
}

// Usable once a complete index exists; stays usable on that index during a reindex.
function usable(doc) {
  return doc.status === "ready" || doc.n_chunks > 0;
}

function toggleSelect(docId, on) {
  if (on && state.selected.size >= MAX_SELECTED) return renderDocs();
  if (on) state.selected.add(docId); else state.selected.delete(docId);
  renderDocs();
}

// One request per file, so the gateway's request limit equals the per-file limit (MAX_UPLOAD_MB)
// and one oversized or broken file does not fail the others.
async function upload(files) {
  files = [...files];
  if (!files.length) return;
  const msg = $("#upload-msg");
  const lines = [];
  const uploaded = new Set();
  for (const [i, file] of files.entries()) {
    msg.textContent = `Uploading ${i + 1} of ${files.length}: ${file.name}...`;
    const form = new FormData();
    form.append("files", file);
    try {
      const [r] = await api("/documents", { method: "POST", body: form });
      if (!r.error) uploaded.add(r.name);
      lines.push(r.error ? h("div", { class: "err" }, `${r.name}: ${r.error}`) : h("div", {}, `${r.name}: ${r.document.status}`));
    } catch (err) {
      lines.push(h("div", { class: "err" }, `${file.name}: ${err.message}`));
    }
    refreshDocs();
  }
  msg.replaceChildren(...lines);
  state.lastUpload = uploaded;
  settleUploadMsg();
}

// Once every file of the last upload is ready or failed, the list shows its status: drop the "queued" lines.
function settleUploadMsg() {
  if (!state.lastUpload?.size) return;
  const status = new Map(state.docs.map((d) => [d.name, d.status]));
  if ([...state.lastUpload].some((name) => !status.has(name) || ["queued", "processing"].includes(status.get(name)))) return;
  state.lastUpload = null;
  $("#upload-msg").querySelectorAll("div:not(.err)").forEach((line) => line.remove());
}

// One request per document at a time: a second click while the first is on its way does nothing.
async function reindex(doc) {
  if (state.reindexing.has(doc.doc_id)) return;
  state.reindexing.add(doc.doc_id);
  renderDocs();
  try {
    await api(`/documents/${doc.doc_id}/reindex`, { method: "POST" });
  } catch (err) {
    alert(err.message);
  } finally {
    state.reindexing.delete(doc.doc_id);
  }
  refreshDocs();
}

async function removeDoc(doc) {
  if (!confirm(`Delete "${doc.name}" and its index?`)) return;
  try { await api(`/documents/${doc.doc_id}`, { method: "DELETE" }); } catch (err) { alert(err.message); }
  state.selected.delete(doc.doc_id);
  refreshDocs();
}

/* ---------- chat ---------- */

function updateSend() {
  $("#send").disabled = state.busy;
  for (const id of ["#chat-new", "#new-chat"]) $(id).disabled = state.busy;
  $("#question").placeholder = state.selected.size
    ? "Ask about the selected documents..."
    : "Select documents to search them, or tell me something to remember...";
}

// Sources by their number [n]. Numbers need not run 1..N: a Read all answer keeps the passage numbers of the
// document (for example 1, 3 and 15), so a citation is looked up by number, never by position.
function sourcesByNumber(sources) {
  return new Map(sources.map((s, i) => [s.n ?? i + 1, s]));
}

function linkCitations(container, sources, onClick) {
  const byNumber = sourcesByNumber(sources);
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) if (/\[\d+(?:\s*[,;]\s*\d+)*\]/.test(walker.currentNode.nodeValue)) nodes.push(walker.currentNode);
  for (const node of nodes) {
    if (node.parentElement.closest("code, pre")) continue;
    const frag = document.createDocumentFragment();
    let last = 0;
    const text = node.nodeValue;
    for (const m of text.matchAll(/\[(\d+(?:\s*[,;]\s*\d+)*)\]/g)) {
      frag.append(text.slice(last, m.index));
      for (const n of m[1].split(/\s*[,;]\s*/).map(Number)) {
        const src = byNumber.get(n);
        frag.append(h("cite", { class: `ref${src ? "" : " invalid"}`, title: src ? src.doc_name : "not a source", onclick: () => src && onClick(n) }, String(n)));
      }
      last = m.index + m[0].length;
    }
    frag.append(text.slice(last));
    node.replaceWith(frag);
  }
}

function showSources(sources, focus) {
  const list = $("#sources-list");
  list.replaceChildren(...sources.map((s) => {
    const ranks = Object.entries(s.ranks || {}).map(([name, rank]) => h("span", { class: "badge" }, `${name} #${rank}`));
    if (typeof s.score === "number") ranks.push(h("span", { class: "badge" }, `rerank ${s.score.toFixed(2)}`));
    const text = h("div", { class: "src-text" });
    text.innerHTML = markdown(s.text);
    return h("div", { class: `src${s.n === focus ? " focus" : ""}`, id: `src-${s.n}` },
      h("div", { class: "src-head" }, h("span", { class: "src-n" }, `[${s.n}]`), h("span", { class: "src-doc" }, s.doc_name)),
      h("div", { class: "src-loc" }, s.location ? `${s.kind}, ${s.location}` : s.kind),
      ranks.length ? h("div", { class: "src-ranks" }, ranks) : null,
      s.asset ? h("img", { src: `/api/documents/${s.doc_id}/assets/${s.asset}`, alt: `Figure from ${s.doc_name}`, loading: "lazy" }) : null,
      text);
  }));
  openPanel("Sources");
  if (focus) document.getElementById(`src-${focus}`)?.scrollIntoView({ block: "nearest" });
}

// The side panel shows one view at a time. Each opening gets a new token; a loader writes into the panel only
// while its token is current, so a slow answer never lands in a panel that shows something else by then.
const panel = { mode: null, token: 0 };

function openPanel(title) {
  panel.mode = title;
  panel.token += 1;
  $("#panel-title").textContent = title;
  $("#sources").hidden = false;
  $(".layout").classList.add("with-sources");
  return panel.token;
}

function panelShows(token) {
  return token === panel.token && !$("#sources").hidden;
}

const LOGIN_FAILURE = { bad_password: "wrong password", unknown_user: "unknown username", disabled: "account disabled or being deleted",
  credentials_changed: "password changed, reset or account disabled during the sign-in" };

function auditSummary(e) {
  const d = e.detail || {};
  if (e.action.startsWith("chat.")) {
    const route = ROUTE_LABEL[d.route] || d.route || "";
    return `${d.question || ""} | ${route}, ${(d.sources || []).length} sources, cited ${(d.citations || []).join(", ") || "none"}${d.error ? ` | ${d.error}` : ""}`;
  }
  if (e.action === "document.ready") return `${d.name}: ${d.chunks} chunks, ${d.images} images, ${(d.tables || []).length} tables, ${d.seconds} s`;
  if (e.action === "auth.login_succeeded") return d.ip ? `from ${d.ip}` : "";
  if (e.action === "auth.login_failed") {
    return [LOGIN_FAILURE[d.reason] || d.reason, d.ip && `from ${d.ip}`, d.username && `name tried: ${d.username}`].filter(Boolean).join(", ");
  }
  if (e.action === "auth.locked") {
    return `${d.scope === "ip" ? "address" : "account"} locked until ${new Date(d.until).toLocaleString()}${d.ip ? `, from ${d.ip}` : ""}`;
  }
  if (e.action === "auth.token_created") return `token "${d.name}", expires ${new Date(d.expires_at).toLocaleDateString()}`;
  if (e.action.startsWith("memory.")) {
    const origin = { auto: "learned from a chat", user: "added by you" }[d.source] || d.source;
    return [d.text && `"${d.text}"`, d.replaces && `replaces "${d.replaces}"`, origin && `(${origin})`].filter(Boolean).join(" ");
  }
  const summary = [d.name, d.username, d.role && `role ${d.role}`, d.error, d.status].filter(Boolean).join(" | ");
  return summary || AUDIT_WORDS[e.action] || e.action.replace(/[._]/g, " ");  // never an empty line
}

const AUDIT_WORDS = {
  "auth.logout": "signed out",
  "auth.disabled_by_admin": "account disabled by an administrator",
  "auth.enabled_by_admin": "account enabled by an administrator",
  "auth.password_changed": "password changed",
  "auth.password_reset_by_admin": "password reset by an administrator",
  "auth.security_notice_dismissed": "security notice dismissed",
  "thread.deleted": "chat deleted",
};

// A deleted account's details are crypto-shredded; an altered ciphertext cannot be read. Both are plain notes.
function auditNote(detail) {
  if (detail?._shredded) return h("div", { class: "muted audit-note" }, "Details erased with the account.");
  if (detail?._unreadable) return h("div", { class: "error audit-note" }, "Details unreadable: the stored record was changed.");
  return null;
}

// One event: a card that opens to the full detail, or a plain card with the note when there is no detail to show.
function auditCard(number, e, where) {
  const top = [
    h("div", { class: "src-head" }, h("span", { class: "src-n" }, number), h("span", { class: "src-doc" }, e.action)),
    h("div", { class: "src-loc" }, `${new Date(e.time).toLocaleString()}, ${where}`),
  ];
  const note = auditNote(e.detail);
  if (note) return h("div", { class: "src audit" }, ...top, note);
  return h("details", { class: "src audit" },
    h("summary", {}, ...top, h("div", { class: "audit-line" }, auditSummary(e))),
    h("pre", { class: "audit-json" }, JSON.stringify(e.detail, null, 2)));
}

// The whole chain is checked; a user learns where it broke only when the first altered event is one of theirs.
function auditVerdict(check) {
  if (check.ok) return h("div", { class: "audit-ok" }, `Hash chain verified: ${check.events} of your events, unaltered.`);
  if (check.first_bad) return h("div", { class: "audit-bad" }, `The log was altered at your event #${check.first_bad}.`);
  return h("div", { class: "audit-bad" }, "The audit log was altered (not in your events). Tell your administrator.");
}

function auditEntry(e) {
  return auditCard(`#${e.user_seq}`, e, e.service);
}

const AUDIT_PAGE = 200;

// Your own events, newest first, numbered in your own sequence (#1 is your first event).
async function showAudit() {
  const list = $("#sources-list");
  const token = openPanel("Audit log");
  list.replaceChildren(h("div", { class: "muted" }, "Loading..."));
  let events, check;
  try {
    [events, check] = await Promise.all([api(`/audit?limit=${AUDIT_PAGE}`), api("/audit/verify")]);
  } catch (err) {
    if (panelShows(token)) list.replaceChildren(h("div", { class: "error" }, `Could not load the audit log: ${err.message}`));
    return;
  }
  if (!panelShows(token)) return;
  const more = h("button", { class: "link audit-more", type: "button" }, "Older events");
  more.addEventListener("click", async () => {
    more.disabled = true;
    try {
      const older = await api(`/audit?limit=${AUDIT_PAGE}&before=${events[events.length - 1].user_seq}`);
      events = events.concat(older);
      more.before(...older.map(auditEntry));
      if (older.length < AUDIT_PAGE) more.remove();
    } catch (err) {
      more.textContent = `Could not load more: ${err.message}`;
    } finally {
      more.disabled = false;
    }
  });
  list.replaceChildren(...[
    auditVerdict(check),
    state.me?.role === "admin" ? adminAuditTools() : null,
    ...events.map(auditEntry),
    events.length ? null : h("div", { class: "muted" }, "No events yet."),
    events.length >= AUDIT_PAGE ? more : null,
  ].filter(Boolean));
}

function hideSources() {
  panel.mode = null;
  panel.token += 1;
  $("#sources").hidden = true;
  $(".layout").classList.remove("with-sources");
}

async function* sse(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      if (data.length) yield [event, JSON.parse(data.join("\n"))];
    }
  }
}

function assistantView() {
  const steps = h("ol", { class: "steps", hidden: true });
  const reasoningText = h("div");
  const reasoning = h("details", { class: "reasoning", hidden: true }, h("summary", {}, "Reasoning"), reasoningText);
  const memory = h("div", { class: "mem-chips", hidden: true });
  const answer = h("div", { class: "answer typing" });
  const notes = h("ul", { class: "answer-notes", hidden: true });
  const meta = h("div", { class: "meta" });
  const el = h("div", { class: "msg assistant" }, steps, reasoning, memory, answer, notes, meta);
  return { el, steps, reasoning, reasoningText, memory, answer, notes, meta };
}

function refreshMemoryPanel() {
  if (panel.mode === "Memory" && !$("#sources").hidden) showMemory();
}

// One chip per remembered or forgotten item. Live chips get Undo: delete a saved item, re-add a forgotten one.
function addMemoryChips(view, event, live) {
  view.memory.hidden = false;
  for (const item of event.items || []) {
    const saved = event.action === "saved";
    const label = h("span", { class: "mem-chip-text" }, `${saved ? "Saved to memory" : "Forgot"}: ${item.text}`);
    const undo = live ? h("button", { class: "link", type: "button" }, "Undo") : null;
    const chip = h("div", { class: `mem-chip${saved ? "" : " forgotten"}`, role: "status" }, label, undo);
    undo?.addEventListener("click", async () => {
      undo.disabled = true;
      try {
        if (saved) await api(`/memories/${item.memory_id}`, { method: "DELETE" });
        else await api("/memories", { method: "POST", headers: { "Content-Type": "application/json" },
                                      body: JSON.stringify({ text: item.text, source: "user" }) });
        label.textContent = `${saved ? "Removed from memory" : "Restored to memory"}: ${item.text}`;
        chip.classList.add("undone");
        undo.remove();
      } catch (err) {
        undo.disabled = false;
        label.textContent = `Could not undo: ${err.message}`;
      }
      refreshMemoryPanel();
    });
    view.memory.append(chip);
  }
}

// What did not go as planned in this answer (a "notice" event: the planner fell back to search, the answer hit
// its length limit, deep search ran out of steps ...), in the words the server chose.
function addNotice(view, data) {
  view.notes.hidden = false;
  view.notes.append(h("li", { role: "status" }, data.message || data.code));
}

// Asks the same question again in the open chat, for an answer that failed or was interrupted, with the
// documents (those still usable), mode and Think setting it was first asked with.
function retryButton(question, turn = {}) {
  return h("button", { class: "link retry", type: "button", title: "Ask this question again",
    onclick: () => { if (!state.busy && question) { restoreTurn(turn); ask(question); } } }, "Retry");
}

function restoreTurn(turn) {
  if (Array.isArray(turn.doc_ids)) {
    const ready = new Set(state.docs.filter(usable).map((d) => d.doc_id));
    const ids = turn.doc_ids.filter((d) => ready.has(d)).slice(0, MAX_SELECTED);
    if (ids.length) {
      state.selected = new Set(ids);
      renderDocs();
    }
  }
  if (turn.mode) {
    $("#deep").checked = turn.mode === "deep";
    $("#exhaustive").checked = turn.mode === "exhaustive";
  }
  if (typeof turn.think === "boolean") $("#think").checked = turn.think;
}

// The answer mode the composer's toggles ask for. Deep search and Read all exclude each other (init wires them).
function chatMode() {
  if ($("#exhaustive").checked) return "exhaustive";
  return $("#deep").checked ? "deep" : "single";
}

function failure(text, question, turn) {
  return h("p", { class: "error" }, text, question ? " " : null, question ? retryButton(question, turn) : null);
}

function stepItem(data) {
  const args = Object.values(data.args || {}).map((v) => (Array.isArray(v) ? v.join(", ") : String(v))).join(" | ");
  const tail = data.error ? h("span", { class: "error" }, ` (${data.error})`) : data.sources ? `: ${data.sources.length} sources` : "";
  return h("li", {}, h("code", {}, data.tool), ` ${args}`, tail);
}

function renderAnswer(view, text, sources) {
  view.answer.innerHTML = markdown(text);
  linkCitations(view.answer, sources, (n) => showSources(sources, n));
}

function renderFooter(view, info, sources) {
  const perDoc = Object.entries(info.per_doc || {}).map(([name, n]) => `${name}: ${n}`).join(", ");
  const searched = !(NO_SEARCH_ROUTES.has(info.route) && !sources.length);
  const parts = [searched ? `${sources.length} sources${perDoc ? ` (${perDoc})` : ""}` : null, ROUTE_LABEL[info.route] || info.route];
  if (info.timings?.total_ms) parts.push(`retrieval ${(info.timings.total_ms / 1000).toFixed(1)} s`);
  if (info.timings?.generate_ms) parts.push(`answer ${(info.timings.generate_ms / 1000).toFixed(1)} s`);
  if (info.invalid_citations?.length) parts.push(`invalid citations: ${info.invalid_citations.join(", ")}`);
  view.meta.replaceChildren(
    ...parts.filter(Boolean).map((p) => h("span", {}, p)),
    ...(sources.length ? [h("button", { class: "link", type: "button", onclick: () => showSources(sources) }, "Show sources")] : []));
}

function addUserMessage(text) {
  const box = $("#messages");
  box.querySelector(".empty")?.remove();
  box.append(h("div", { class: "msg user" }, text));
}

function showHistory(detail) {
  const box = $("#messages");
  box.replaceChildren();
  let question = "";
  for (const msg of detail.messages) {
    if (msg.role === "user") {
      addUserMessage(msg.content);
      question = msg.content;
      continue;
    }
    const meta = msg.meta || {};
    const view = assistantView();
    view.answer.classList.remove("typing");
    for (const step of meta.steps || []) {
      view.steps.hidden = false;
      view.steps.append(stepItem(step));
    }
    for (const event of meta.memory || []) addMemoryChips(view, event, false);
    renderAnswer(view, msg.content || "", meta.sources || []);
    for (const note of meta.notices || []) addNotice(view, note);
    const turn = { doc_ids: meta.doc_ids, mode: meta.mode, think: meta.think };
    if (meta.interrupted) view.answer.append(failure("This answer was interrupted before it finished.", question, turn));
    else if (meta.error) view.answer.append(failure(`Error: ${meta.error}`, question, turn));
    renderFooter(view, meta, meta.sources || []);
    box.append(view.el);
  }
  if (!detail.messages.length) box.append(h("div", { class: "empty" }, "Empty chat. Ask a question."));
  box.scrollTop = box.scrollHeight;
}

async function ask(question) {
  state.view += 1;  // a chat still loading is dropped: this answer belongs to the chat on screen
  addUserMessage(question);
  const view = assistantView();
  const box = $("#messages");
  box.append(view.el);
  box.scrollTop = box.scrollHeight;

  state.busy = true;
  updateSend();
  const turn = { doc_ids: [...state.selected], mode: chatMode(), think: $("#think").checked };
  let text = "";
  let sources = [];
  let info = {};
  let frame = 0;
  let finished = false;  // set by "done" or "error"; a stream that just stops was interrupted
  const render = () => {
    frame = 0;
    renderAnswer(view, text, sources);
    box.scrollTop = box.scrollHeight;
  };
  try {
    if (!state.threadId) {  // first question of a new chat creates its thread
      const thread = await api("/threads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: question.slice(0, 80), doc_ids: turn.doc_ids }),
      });
      setThread(thread.thread_id);
    }
    const response = await apiFetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        doc_ids: turn.doc_ids,
        thread_id: state.threadId,
        think: turn.think,
        mode: turn.mode,
      }),
    });
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).detail; } catch { /* not JSON */ }
      if (response.status === 404 && String(detail).startsWith("thread")) setThread(null);
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    for await (const [event, data] of sse(response)) {
      if (event === "step") {
        view.steps.hidden = false;
        view.steps.append(stepItem(data));
      } else if (event === "notice") {
        addNotice(view, data);
      } else if (event === "memory") {
        addMemoryChips(view, data, true);
        refreshMemoryPanel();
      } else if (event === "sources") {
        sources = data.sources;
        info = data;
      } else if (event === "reasoning") {
        view.reasoning.hidden = false;
        view.reasoningText.textContent += data.text;
      } else if (event === "token") {
        text += data.text;
        if (!frame) frame = requestAnimationFrame(render);
      } else if (event === "done") {
        finished = true;
        render();
        renderFooter(view, { ...info, ...data }, sources);
      } else if (event === "error") {
        finished = true;
        throw new Error(data.message);
      }
    }
    if (!finished) {
      render();
      view.answer.append(failure("The answer was interrupted before it finished.", question, turn));
    }
  } catch (err) {
    view.answer.append(failure(`Error: ${err.message}`, question, turn));
  } finally {
    view.answer.classList.remove("typing");
    state.busy = false;
    updateSend();
    refreshThreads();
  }
}

/* ---------- threads ---------- */

function when(iso) {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

function setThread(id) {
  state.threadId = id;
  history.replaceState(null, "", id ? `#t=${id}` : location.pathname);
}

async function refreshThreads() {
  try { state.threads = await api("/threads"); } catch { state.threads = []; }
  renderThreads();
}

function renderThreads() {
  const list = $("#threads");
  list.replaceChildren();
  if (!state.threads.length) list.append(h("li", { class: "muted" }, "No chats yet."));
  for (const t of state.threads) {
    const questions = Math.ceil(t.message_count / 2);
    list.append(h("li", { class: `thread${t.thread_id === state.threadId ? " selected" : ""}`, onclick: () => openThread(t.thread_id) },
      h("div", {},
        h("div", { class: "doc-name" }, t.title),
        h("div", { class: "doc-meta" }, `${questions} question${questions === 1 ? "" : "s"}, ${when(t.updated_at)}${t.summarized ? ", summarized" : ""}`)),
      h("button", { class: "icon-btn danger", type: "button", title: "Delete chat", "aria-label": "Delete chat", "data-icon": "trash",
        onclick: (e) => { e.stopPropagation(); removeThread(t); } })));
  }
}

async function openThread(id, { fromLink = false } = {}) {
  if (state.busy) return;
  const view = ++state.view;
  try {
    const detail = await api(`/threads/${id}`);
    if (view !== state.view) return;  // another chat was opened, or a new one started, meanwhile
    setThread(id);
    const ready = new Set(state.docs.filter(usable).map((d) => d.doc_id));
    const restored = detail.thread.doc_ids.filter((d) => ready.has(d));
    if (restored.length) state.selected = new Set(restored);
    renderDocs();
    hideSources();
    showHistory(detail);
    renderThreads();
  } catch (err) {
    if (view !== state.view) return;
    setThread(null);
    if (fromLink) return newChat();  // a stale or foreign link in the address bar: start clean, no error
    $("#messages").replaceChildren(h("div", { class: "empty" }, `Could not open that chat: ${err.message}`));
  }
}

function newChat() {
  if (state.busy) return;  // the answer streaming now belongs to the open chat; start a new one when it is done
  state.view += 1;
  setThread(null);
  hideSources();
  $("#messages").replaceChildren(h("div", { class: "empty" }, "New chat. Ask about the selected documents, or tell me something to remember."));
  renderThreads();
}

async function removeThread(thread) {
  if (state.busy && thread.thread_id === state.threadId) return alert("Wait for the answer to finish before deleting this chat.");
  if (!confirm(`Delete the chat "${thread.title}"?`)) return;
  try { await api(`/threads/${thread.thread_id}`, { method: "DELETE" }); } catch (err) { alert(err.message); }
  if (thread.thread_id === state.threadId) newChat();
  refreshThreads();
}

function showTab(name) {
  $("#tab-docs").classList.toggle("active", name === "docs");
  $("#tab-chats").classList.toggle("active", name === "chats");
  $("#docs-pane").hidden = name !== "docs";
  $("#chats-pane").hidden = name !== "chats";
  if (name === "chats") refreshThreads();
}

/* ---------- memory ---------- */

async function showMemory() {
  const list = $("#sources-list");
  const token = openPanel("Memory");
  const input = h("input", { class: "mem-input", type: "text", maxlength: "500", placeholder: "Something to remember in every chat..." });
  const form = h("form", {
    class: "mem-form",
    onsubmit: async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (text.length < 3) return;
      try {
        await api("/memories", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, source: "user" }) });
      } catch (err) {  // e.g. 422 for text that looks like a password or key: say why, keep the input
        form.parentElement?.querySelector(".mem-error")?.remove();
        form.after(h("div", { class: "error mem-error", role: "alert" }, err.message));
        return;
      }
      showMemory();
    },
  }, input, h("button", { class: "primary", type: "submit" }, "Add"));
  let items;
  try {
    items = await api("/memories");
  } catch (err) {
    if (panelShows(token)) list.replaceChildren(h("div", { class: "error" }, `Could not load memory: ${err.message}`));
    return;
  }
  if (!panelShows(token)) return;
  list.replaceChildren(
    h("div", { class: "muted mem-help" }, "Facts and preferences used in every chat to shape answers, never as a source of document facts. Learned when you mention them, or added here."),
    form,
    ...items.map((m) => h("div", { class: "src mem" },
      h("div", { class: "mem-text" }, m.text),
      h("div", { class: "src-loc" }, `${m.source === "auto" ? "learned from a chat" : "added by you"}, ${when(m.created_at)}`),
      h("button", { class: "icon-btn danger mem-del", type: "button", title: "Forget", "aria-label": "Forget", "data-icon": "close",
        onclick: async (e) => {
          const item = e.currentTarget.closest(".mem");  // currentTarget is gone after the await
          try {
            await api(`/memories/${m.memory_id}`, { method: "DELETE" });
            showMemory();
          } catch (err) {
            item.append(h("div", { class: "error" }, `Could not forget: ${err.message}`));
          }
        } }))),
    ...(items.length ? [] : [h("div", { class: "mem-empty" }, "Nothing remembered yet. Mention a preference in a chat, or add one above.")]));
}

/* ---------- wiring ---------- */

// An administrator reset this account's password. Shown at every sign-in until the user dismisses it: changing
// the temporary password does not clear it, so an admin who reset it and signed in as the user cannot hide it.
function renderNotice(notice) {
  const box = $("#notice");
  box.hidden = !notice;
  if (!notice) return;
  const dismiss = h("button", { type: "button", class: "link" }, "Dismiss");
  dismiss.addEventListener("click", async () => {
    dismiss.disabled = true;
    try {
      await api("/auth/security-notice/dismiss", { method: "POST" });
      box.hidden = true;
    } catch (err) {
      dismiss.disabled = false;
      if (!state.leaving) box.append(h("span", { class: "error" }, ` Could not dismiss: ${err.message}`));
    }
  });
  box.replaceChildren(
    h("span", {}, `An administrator reset your password on ${new Date(notice.at).toLocaleString()}. If you did not `
      + "ask for this, tell your administrator, and check Audit log and API tokens."),
    dismiss);
}

async function init() {
  let session;
  try {
    session = await api("/auth/session");
    state.me = session.user;
  } catch (err) {
    if (!state.leaving) $("#status-text").textContent = `cannot check the session: ${err.message}`;
    return;
  }
  if (state.me.must_change_password) return toLogin();
  state.selected = new Set(load(selKey(), []));
  try { localStorage.removeItem("selected"); } catch { /* storage unavailable */ }  // pre-account key
  renderUserMenu();
  renderNotice(session.security_notice);
  const drop = $("#drop");
  $("#file").addEventListener("change", (e) => { upload([...e.target.files]); e.target.value = ""; });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); upload([...e.dataTransfer.files]); });
  $("#sel-clear").addEventListener("click", () => { state.selected.clear(); renderDocs(); });
  $("#sources-close").addEventListener("click", hideSources);
  $("#audit-open").addEventListener("click", showAudit);
  $("#memory-open").addEventListener("click", showMemory);
  $("#tab-docs").addEventListener("click", () => showTab("docs"));
  $("#tab-chats").addEventListener("click", () => showTab("chats"));
  $("#chat-new").addEventListener("click", newChat);
  $("#new-chat").addEventListener("click", newChat);
  $("#deep").addEventListener("change", () => { if ($("#deep").checked) $("#exhaustive").checked = false; });
  $("#exhaustive").addEventListener("change", () => { if ($("#exhaustive").checked) $("#deep").checked = false; });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("#question").value.trim();
    if (!q || state.busy) return;
    $("#question").value = "";
    ask(q);
  });
  $("#question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  refreshStatus();
  await refreshDocs();
  const linked = location.hash.match(/^#t=([0-9a-f]{32})$/);
  if (linked) openThread(linked[1], { fromLink: true });
  refreshThreads();
}

document.addEventListener("DOMContentLoaded", init);
