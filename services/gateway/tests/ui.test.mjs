// The web UI's guards against stale async results and the Users panel's server-given actions
// (docs/STATE_MACHINES.md, M17, M19, M21). Runs app.js, account.js and admin.js against a minimal DOM stub,
// and checks the pages' markup:
//
//     node --test services/gateway/tests/ui.test.mjs

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const STATIC = new URL("../static/", import.meta.url);

class El {
  constructor(tag) {
    Object.assign(this, { tagName: tag, children: [], attrs: {}, listeners: {}, hidden: false, disabled: false,
      className: "", value: "", open: false });
    this.classList = { add() {}, remove() {}, toggle() {}, contains: () => false };
  }
  append(...c) { this.adopt(c); this.children.push(...c); }
  replaceChildren(...c) { this.adopt(c); this.children = c; }
  adopt(c) { for (const x of c) if (x instanceof El) x.parent = this; }
  replaceWith(n) { const s = this.parent.children; s[s.indexOf(this)] = n; n.parent = this.parent; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((x) => x !== this); }
  addEventListener(kind, fn) { (this.listeners[kind] ||= []).push(fn); }
  setAttribute(k, v) { this.attrs[k] = v; }
  querySelector() { return new El("div"); }
  querySelectorAll() { return []; }
  get childElementCount() { return this.children.length; }
  get textContent() { return this.children.map((c) => (typeof c === "string" ? c : c.textContent)).join(""); }
  set textContent(v) { this.children = [String(v)]; }
  set innerHTML(v) { this.children = [String(v)]; }
  before() {} after() {} focus() {} scrollIntoView() {} closest() { return null; }
}

// A browser for one test: fetch answers from `routes` (path -> [delay ms, body, status] or a function giving one).
function browser(routes = {}) {
  const ids = {};
  const document = {
    querySelector: (sel) => (ids[sel] ||= new El(sel)),
    createElement: (tag) => new El(tag),
    addEventListener() {},
    createDocumentFragment: () => new El("fragment"),
    createTreeWalker: () => ({ nextNode: () => false }),
  };
  const requests = [];
  const fetch = (url, opts = {}) => {
    requests.push({ url, body: opts.body ? JSON.parse(opts.body) : null });
    const route = routes[url.replace(/^\/api/, "")];
    const [delay, body, status = 200] = (typeof route === "function" ? route() : route) || [0, []];
    // A string body is a server-sent event stream, read through response.body.getReader().
    const stream = typeof body === "string" ? { getReader: () => {
      let sent = false;
      return { read: async () => (sent ? { done: true } : ((sent = true), { done: false, value: new TextEncoder().encode(body) })) };
    } } : null;
    return new Promise((resolve) => setTimeout(() => resolve({
      status, ok: status < 300, statusText: "", json: async () => body, body: stream }), delay));
  };
  const context = vm.createContext({
    document, fetch, setTimeout, clearTimeout, console, confirm: () => true, alert() {},
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    location: { hash: "", pathname: "/", replace() {} }, history: { replaceState() {} },
    requestAnimationFrame: (f) => setTimeout(f, 0), NodeFilter: { SHOW_TEXT: 4 }, TextDecoder, TextEncoder,
    marked: { parse: (t) => t }, DOMPurify: { sanitize: (t) => t },
  });
  for (const file of ["app.js", "account.js", "admin.js"]) {
    vm.runInContext(readFileSync(new URL(file, STATIC), "utf8"), context, { filename: file });
  }
  const run = (code) => vm.runInContext(code, context);
  return { run, requests, $: (sel) => document.querySelector(sel), wait: (ms) => new Promise((r) => setTimeout(r, ms)) };
}

const labels = (el) => el.children.filter((c) => c instanceof El && c.tagName === "button").map((c) => c.textContent);

test("a slow panel load never lands in the panel opened after it (M19)", async () => {
  const b = browser({ "/memories": [60, [{ memory_id: "m1", text: "likes tea", source: "user", created_at: "t" }]],
    "/auth/tokens": [0, []] });
  b.run("state.me = { user_id: 'u1', username: 'amy', role: 'user' }");
  const memory = b.run("showMemory()");
  await b.run("showTokens()");
  await memory;
  assert.equal(b.$("#panel-title").textContent, "API tokens");
  assert.match(b.$("#sources-list").children[0].textContent, /Tokens let scripts/);
});

test("a new chat is refused while an answer streams into the open one (M17)", () => {
  const b = browser();
  b.run("state.threadId = 'a'.repeat(32); state.busy = true; newChat()");
  assert.equal(b.run("state.threadId"), "a".repeat(32));
  b.run("state.busy = false; newChat()");
  assert.equal(b.run("state.threadId"), null);
});

test("a chat that loads after another was opened is dropped (M17)", async () => {
  const slow = "1".repeat(32), fast = "2".repeat(32);
  const detail = (id) => ({ thread: { thread_id: id, doc_ids: [] }, messages: [] });
  const b = browser({ [`/threads/${slow}`]: [60, detail(slow)], [`/threads/${fast}`]: [0, detail(fast)] });
  const first = b.run(`openThread("${slow}")`);
  await b.run(`openThread("${fast}")`);
  await first;
  assert.equal(b.run("state.threadId"), fast);
});

test("the Users panel draws the actions the server gives, nothing it derives itself (M21)", () => {
  const b = browser();
  const row = b.run(`userRow({ user_id: "u2", username: "carl", role: "user", state: "disabled", disabled: true,
    must_change_password: false, actions: ["enable", "delete"], active_sessions: 0, created_at: "2026-01-01" },
    { user_id: "u1" })`);
  const buttons = row.children.find((c) => c.className === "user-actions");
  assert.deepEqual(labels(buttons), ["Enable", "Delete"]);
  assert.doesNotMatch(row.children[0].textContent, /YOU/);  // only the viewer's own row says YOU
  const stuck = b.run(`userRow({ user_id: "u3", username: "eve", role: "user", state: "deleting", disabled: true,
    actions: ["delete"], deletion: { running: false, error: "threads: disk full" }, active_sessions: 0,
    created_at: "2026-01-01" }, { user_id: "u1" })`);
  const bar = stuck.children.find((c) => c.className === "user-actions");
  assert.deepEqual(labels(bar), ["Retry delete"]);
  assert.match(bar.textContent, /threads: disk full/);
  const mine = b.run(`userRow({ user_id: "u1", username: "root", role: "admin", state: "active", actions: [],
    active_sessions: 1, created_at: "2026-01-01" }, { user_id: "u1" })`);
  assert.match(mine.children[0].textContent, /YOU/);
  assert.deepEqual(labels(mine.children.find((c) => c.className === "user-actions")), []);
});

const user = (over) => ({ user_id: "u2", username: "carl", role: "user", state: "active", disabled: false,
  must_change_password: false, actions: ["disable", "delete"], active_sessions: 0, created_at: "2026-01-01", ...over });

test("a deletion followed in the background updates only its row", async () => {
  const deleting = user({ user_id: "u3", username: "eve", state: "deleting", actions: ["delete"],
    deletion: { running: true, error: null } });
  let lists = 0;
  const b = browser({ "/admin/users": () => [0, ++lists === 1 ? [user(), deleting] : [user()]] });
  b.run("state.me = { user_id: 'u1', username: 'root', role: 'admin' }; pendingDeletion = { user_id: 'u3', username: 'eve' }");
  await b.run("showUsers()");
  const list = b.$("#sources-list");
  const form = list.children.find((c) => c.tagName === "form");
  form.marker = "half-typed";
  const carl = list.children.find((c) => c.className === "src user-row" && c.textContent.includes("carl"));
  carl.marker = "confirm open";
  await b.wait(3200);
  assert.equal(lists, 2);
  assert.equal(list.children.find((c) => c.tagName === "form").marker, "half-typed");  // the same element
  assert.equal(list.children.find((c) => c.textContent.includes("carl")).marker, "confirm open");
  assert.ok(!list.children.some((c) => c.className === "src user-row" && c.textContent.startsWith("eve")));
  assert.match(list.children.find((c) => c.attrs.role === "status").textContent, /Deleted eve/);
});

test("Reindex waits for a document that is queued or being processed", () => {
  const b = browser();
  b.run(`state.me = { user_id: "u1" }; state.docs = ["queued", "processing", "ready", "failed"].map((status, i) => ({
    doc_id: String(i).repeat(32), name: status + ".txt", doc_type: "text", status, n_chunks: status === "ready" ? 3 : 0,
    tables: [], warnings: [], progress: "", error: null })); renderDocs()`);
  const items = b.$("#docs").children;
  const reindex = items.map((li) => li.children[2].children[0]);
  assert.deepEqual(reindex.map((btn) => btn.attrs.disabled !== undefined), [true, true, false, false]);
});

test("a temporary password that arrives after the admin moved on is shown, not lost", async () => {
  const bob = user({ user_id: "u4", username: "bob", actions: ["reset_password"] });
  const b = browser({ "/admin/users": [0, [bob]], "/admin/users/u4/reset-password": [60, { temporary_password: "tmp-pass-123" }],
    "/memories": [0, []] });
  b.run("state.me = { user_id: 'u1', username: 'root', role: 'admin' }");
  const row = b.run("userRow(" + JSON.stringify(bob) + ", state.me)");
  const reset = row.children.find((c) => c.className === "user-actions").children.find((c) => c.textContent === "Reset password");
  reset.listeners.click[0]();  // the admin confirms ...
  await b.run("showMemory()");  // ... and opens Memory before the answer
  await b.wait(120);
  assert.equal(b.$("#panel-title").textContent, "Users");
  assert.match(b.$("#sources-list").textContent, /tmp-pass-123/);
  await b.run("showMemory()");
  await b.run("showUsers({ fresh: true })");  // shown once: a fresh open no longer has it
  assert.doesNotMatch(b.$("#sources-list").textContent, /tmp-pass-123/);
});

test("a sign-in refused because the password changed meanwhile has a plain label", () => {
  const b = browser();
  const line = b.run(`auditSummary({ action: "auth.login_failed", detail: { reason: "credentials_changed", ip: "10.0.0.1" } })`);
  assert.match(line, /password changed, reset or account disabled during the sign-in/);
});

test("a chat that loads while a question is being sent is dropped", async () => {
  const slow = "1".repeat(32), made = "2".repeat(32);
  let created = false;  // POST /threads makes the chat; later GETs list chats
  const b = browser({ [`/threads/${slow}`]: [60, { thread: { thread_id: slow, doc_ids: [] }, messages: [] }],
    "/threads": () => (created ? [0, []] : ((created = true), [0, { thread_id: made }])),
    "/chat": [0, { detail: "documents not ready yet" }, 409] });
  const opening = b.run(`openThread("${slow}")`);
  await b.run(`ask("what is new?")`);
  await opening;
  assert.equal(b.run("state.threadId"), made);
});

const events = (...list) => list.map(([name, data]) => `event: ${name}
data: ${JSON.stringify(data)}

`).join("");
const text = (el) => (typeof el === "string" ? el : el.textContent);
// POST /threads makes the chat (its first call); later GETs list the chats.
const threadsRoute = () => { let made = false; return () => (made ? [0, []] : ((made = true), [0, { thread_id: "a".repeat(32) }])); };
const find = (el, pred) => (pred(el) ? el : (el.children || []).filter((c) => c instanceof El).map((c) => find(c, pred)).find(Boolean));

test("a notice in the answer stream is shown with the answer", async () => {
  const stream = events(["notice", { code: "planner_failed", message: "The SQL planner could not make a plan." }],
    ["sources", { route: "search", sources: [] }], ["token", { text: "Answer" }],
    ["notice", { code: "answer_cut", message: "The answer reached its length limit." }],
    ["done", { citations: [], invalid_citations: [], timings: {} }]);
  const b = browser({ "/threads": threadsRoute(), "/chat": [0, stream] });
  await b.run(`ask("which state grew fastest?")`);
  const notes = find(b.$("#messages"), (el) => el.className === "answer-notes");
  assert.equal(notes.hidden, false);
  assert.deepEqual(notes.children.map(text), ["The SQL planner could not make a plan.", "The answer reached its length limit."]);
});

test("a failed or interrupted answer offers Retry, which asks the same question again", async () => {
  let asked = 0;
  const b = browser({ "/threads": threadsRoute(),
    "/chat": () => (++asked, [0, events(["sources", { route: "search", sources: [] }], ["token", { text: "Half" }])]) });
  await b.run(`ask("how many?")`);
  const retry = find(b.$("#messages"), (el) => el.tagName === "button" && text(el) === "Retry");
  assert.ok(retry, "an interrupted answer has a Retry button");
  retry.listeners.click[0]();
  await b.wait(20);
  assert.equal(asked, 2);
  const users = b.$("#messages").children.filter((c) => c.className === "msg user").map(text);
  assert.deepEqual(users, ["how many?", "how many?"]);
});

test("a failed document offers Retry, which reindexes it", async () => {
  let reindexed = 0;
  const b = browser({ [`/documents/${"1".repeat(32)}/reindex`]: () => (++reindexed, [0, {}]), "/documents": [0, []] });
  b.run(`state.me = { user_id: "u1" }; state.docs = [{ doc_id: "1".repeat(32), name: "census.xlsx", doc_type: "spreadsheet",
    status: "failed", n_chunks: 0, tables: [], warnings: [], progress: "", error: "models: busy" }]; renderDocs();
    refreshDocs = async () => {}`);  // no polling timer left behind
  const retry = find(b.$("#docs"), (el) => el.tagName === "button" && text(el) === "Retry");
  assert.ok(retry);
  await retry.listeners.click[0]();
  assert.equal(reindexed, 1);
});

test("Retry on a saved failed answer asks again with that turn's documents, mode and Think (L18)", async () => {
  const one = "1".repeat(32), two = "2".repeat(32), gone = "9".repeat(32);
  const doc = (id) => ({ doc_id: id, name: `${id[0]}.pdf`, doc_type: "pdf", status: "ready", n_chunks: 3, tables: [],
    warnings: [], progress: "" });
  const b = browser({ "/threads": [0, []], "/chat": [0, events(["sources", { route: "deep", sources: [] }],
    ["done", { citations: [], invalid_citations: [], timings: {} }])] });
  b.run(`state.docs = [${JSON.stringify(doc(one))}, ${JSON.stringify(doc(two))}]; state.selected = new Set(["${two}"]);
    $("#deep").checked = false; $("#think").checked = false; state.threadId = "a".repeat(32);
    showHistory({ messages: [{ role: "user", content: "compare them" }, { role: "assistant", content: "",
      meta: { error: "llm: busy", sources: [], mode: "deep", think: true, doc_ids: ["${one}", "${gone}"] } }] })`);
  find(b.$("#messages"), (el) => el.tagName === "button" && text(el) === "Retry").listeners.click[0]();
  await b.wait(20);
  const chat = b.requests.find((r) => r.url.endsWith("/chat")).body;
  assert.deepEqual([chat.question, chat.doc_ids, chat.mode, chat.think], ["compare them", [one], "deep", true]);
  assert.equal(b.$("#deep").checked, true);
});

test("Retry of an exhaustive answer turns Read all back on, Deep search off, and sends mode exhaustive", async () => {
  const one = "1".repeat(32);
  const doc = { doc_id: one, name: "a.pdf", doc_type: "pdf", status: "ready", n_chunks: 3, tables: [], warnings: [],
    progress: "" };
  const b = browser({ "/threads": [0, []], "/chat": [0, events(
    ["step", { tool: "map", args: { sections: "1 of 2", where: "a.pdf: Intro (p. 1)" }, sources: [1], error: null }],
    ["sources", { route: "exhaustive", sources: [] }], ["done", { citations: [], invalid_citations: [], timings: {} }])] });
  b.run(`state.docs = [${JSON.stringify(doc)}]; state.selected = new Set(["${one}"]);
    $("#deep").checked = true; $("#exhaustive").checked = false; $("#think").checked = false; state.threadId = "a".repeat(32);
    showHistory({ messages: [{ role: "user", content: "list every limitation" }, { role: "assistant", content: "",
      meta: { error: "llm: busy", sources: [], mode: "exhaustive", think: false, doc_ids: ["${one}"] } }] })`);
  find(b.$("#messages"), (el) => el.tagName === "button" && text(el) === "Retry").listeners.click[0]();
  await b.wait(20);
  const chat = b.requests.find((r) => r.url.endsWith("/chat")).body;
  assert.deepEqual([chat.question, chat.mode], ["list every limitation", "exhaustive"]);
  assert.deepEqual([b.$("#exhaustive").checked, b.$("#deep").checked], [true, false]);
  assert.equal(b.run("chatMode()"), "exhaustive");
  b.run(`$("#exhaustive").checked = false; $("#deep").checked = false`);
  assert.equal(b.run("chatMode()"), "single");
});

test("a failed document's Retry sends one reindex however often it is clicked (L18)", async () => {
  let reindexed = 0;
  const id = "1".repeat(32);
  const b = browser({ [`/documents/${id}/reindex`]: () => (++reindexed, [30, {}]), "/documents": [0, []] });
  b.run(`state.me = { user_id: "u1" }; state.docs = [{ doc_id: "${id}", name: "census.xlsx", doc_type: "spreadsheet",
    status: "failed", n_chunks: 0, tables: [], warnings: [], progress: "", error: "models: busy" }]; renderDocs();
    refreshDocs = async () => {}`);
  const retry = () => find(b.$("#docs"), (el) => el.tagName === "button" && text(el) === "Retry");
  const first = retry().listeners.click[0]();
  assert.ok("disabled" in retry().attrs);  // re-rendered as waiting
  await retry().listeners.click[0]();
  await first;
  assert.equal(reindexed, 1);
});

test("a saved answer shows its notices, and Retry when it failed", () => {
  const b = browser();
  b.run(`showHistory({ messages: [
    { role: "user", content: "q1" },
    { role: "assistant", content: "partial", meta: { notices: [{ code: "research_budget", message: "Used all steps." }],
      error: "llm: busy", sources: [] } }] })`);
  const box = b.$("#messages");
  assert.equal(text(find(box, (el) => el.className === "answer-notes")), "Used all steps.");
  assert.ok(find(box, (el) => el.tagName === "button" && text(el) === "Retry"));
});

test("the pages load only files that exist, nothing inline (CSP), and the top bar holds every control", () => {
  for (const page of ["index.html", "login.html"]) {
    const html = readFileSync(new URL(page, STATIC), "utf8");
    for (const [, ref] of html.matchAll(/<(?:script|link)\b[^>]*?\b(?:src|href)="([^"]+)"/g)) {
      assert.ok(existsSync(new URL(ref, STATIC)), `${page} loads ${ref}, which does not exist`);
    }
    assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)[^>]*>/, `${page} has an inline script`);
    assert.doesNotMatch(html, /<style\b|\sstyle=|\son[a-z]+=/, `${page} has an inline style or handler`);
  }
  const bar = readFileSync(new URL("index.html", STATIC), "utf8").match(/<header class="topbar">([\s\S]*?)<\/header>/);
  assert.ok(bar, "index.html has the plain top bar");
  for (const id of ["memory-open", "audit-open", "user-menu", "user-name", "password-open", "tokens-open", "users-open",
    "signout", "theme-toggle", "status", "status-text"]) {
    assert.match(bar[1], new RegExp(`id="${id}"`), `the top bar has #${id}`);
  }
  assert.doesNotMatch(bar[1], /<canvas/);
});

test("a citation resolves by source number, also when the numbers skip (Read all keeps passage numbers)", () => {
  const b = browser();
  b.run(`
    var textNode = { nodeValue: "Ages 25 [1], 30 [3] and 35 [15]; [2] is not a source.", parentElement: { closest: () => null },
      replaceWith(f) { this.out = f; } };
    document.createTreeWalker = () => {
      let done = false;
      return { currentNode: textNode, nextNode: () => (done ? false : (done = true)) };
    };
    var opened = [];
    linkCitations({}, [{ n: 1, doc_name: "a.txt" }, { n: 3, doc_name: "b.txt" }, { n: 15, doc_name: "c.txt" }],
      (n) => opened.push(n));
  `);
  const cites = b.run("textNode.out.children.filter((c) => typeof c !== 'string' && c.tagName === 'cite')");
  assert.deepEqual(cites.map((c) => [c.textContent, c.className, c.attrs.title]), [
    ["1", "ref", "a.txt"], ["3", "ref", "b.txt"], ["15", "ref", "c.txt"], ["2", "ref invalid", "not a source"]]);
  for (const c of cites) c.listeners.click[0]();
  assert.deepEqual([...b.run("opened")], [1, 3, 15]);
});
