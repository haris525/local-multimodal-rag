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
    } } : body && typeof body.getReader === "function" ? body : null;
    return new Promise((resolve) => setTimeout(() => resolve({
      status, ok: status < 300, statusText: "", json: async () => body, body: stream }), delay));
  };
  const context = vm.createContext({
    document, fetch, setTimeout, clearTimeout, console, confirm: () => true, alert() {},
    addEventListener() {},
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    location: { hash: "", pathname: "/", replace() {} }, history: { replaceState() {} },
    requestAnimationFrame: (f) => setTimeout(f, 0), NodeFilter: { SHOW_TEXT: 4 }, TextDecoder, TextEncoder,
    cancelAnimationFrame: clearTimeout,
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

test("a saved chat opens without the separately loaded diagram script", () => {
  const b = browser();
  b.run(`globalThis.renderDiagrams = undefined;
    showHistory({ messages: [{ role: "user", content: "my question" },
      { role: "assistant", content: "saved answer", meta: { sources: [] } }] })`);
  assert.match(b.$("#messages").textContent, /saved answer/);
});

test("a diagram enhancement failure does not interrupt the answer or leave Send disabled", async () => {
  const b = browser({ "/threads": threadsRoute(), "/chat": [0, events(["sources", { route: "search", sources: [] }],
    ["token", { text: "answer" }], ["done", { citations: [], timings: {} }])] });
  b.run(`var originalCreate = document.createElement;
    document.createElement = (tag) => {
      const node = originalCreate(tag);
      node.querySelectorAll = (selector) => {
        if (selector === "pre > code.language-mermaid") throw new Error("diagram enhancement failed");
        return [];
      };
      return node;
    };`);
  await b.run('ask("my question")');
  assert.equal(b.run("state.busy"), false);
  assert.match(b.$("#messages").textContent, /answer/);
});

test("a waiting answer shows a loader, elapsed time and real stage changes until done", async () => {
  let stream;
  const body = new ReadableStream({ start(controller) { stream = controller; } });
  const b = browser({ "/threads": threadsRoute(), "/chat": [0, body] });
  b.run('$("#exhaustive").checked = true');
  b.run('var diagramRenders = 0; renderChatDiagrams = () => { diagramRenders++; }');
  const asking = b.run('ask("explain the constitution")');
  const activity = find(b.$("#messages"), (el) => el.className === "answer-activity");
  const label = find(activity, (el) => el.className === "activity-label");
  const elapsed = find(activity, (el) => el.className === "activity-time");
  assert.equal(activity.hidden, false, "feedback appears before the first server response");
  assert.match(activity.textContent, /Preparing your answer.*Read all may take several minutes/s);
  assert.equal(label.attrs.role, "status");
  assert.equal(elapsed.attrs["aria-hidden"], "true", "elapsed seconds do not interrupt screen reader announcements");
  try {
    await b.wait(1100);
    assert.match(elapsed.textContent, /^[1-9]\d*s elapsed$/);
    const send = async (...list) => {
      stream.enqueue(new TextEncoder().encode(events(...list)));
      await b.wait(10);
    };
    await send(["memory", { action: "saved", items: [{ memory_id: "m1", text: "new to the constitution" }] }],
      ["token", { text: "Saved to memory. " }]);
    assert.equal(label.textContent, "Preparing your answer...", "a memory confirmation is not the answer stream");
    await send(["step", { tool: "sections", args: { plan: "2 map calls" } }]);
    assert.equal(label.textContent, "Reading your documents...");
    await send(["step", { tool: "reduce", args: {} }]);
    assert.equal(label.textContent, "Combining what I found...");
    await send(["sources", { route: "exhaustive", sources: [] }], ["reasoning", { text: "Considering the sources" }]);
    assert.equal(label.textContent, "Thinking...");
    await send(["token", { text: "The Constitution" }]);
    assert.equal(label.textContent, "Writing your answer...");
    assert.equal(activity.hidden, false, "the loader remains while the answer streams");
    assert.equal(b.run("diagramRenders"), 0, "streaming tokens do not recreate diagram frames");
    assert.equal(find(b.$("#messages"), (el) => el.className === "answer").textContent, "Saved to memory. The Constitution",
      "answer tokens render while the server stream is still open");
    await send(["done", { citations: [], timings: {} }]);
    assert.equal(activity.hidden, true, "done hides the loader even before the connection closes");
  } finally {
    stream.close();
    await asking;
  }
  assert.equal(b.run("state.busy"), false);
  assert.equal(b.run("diagramRenders"), 1, "completed answers render their diagrams once");
  const stopped = elapsed.textContent;
  await b.wait(1050);
  assert.equal(elapsed.textContent, stopped, "the timer stops when the request ends");
});

test("a failed or interrupted request always stops its loader", async () => {
  for (const route of [[0, { detail: "busy" }, 503], [0, events(["error", { message: "failed" }])],
    [0, events(["token", { text: "Partial" }], ["error", { message: "failed after a token" }])],
    [0, events(["token", { text: "Partial" }])]]) {
    const b = browser({ "/threads": threadsRoute(), "/chat": route });
    await b.run('ask("what happened?")');
    await b.wait(20);  // allow a queued token render to run after the error/interruption
    assert.equal(find(b.$("#messages"), (el) => el.className === "answer-activity").hidden, true);
    assert.equal(b.run("state.busy"), false);
    assert.ok(find(b.$("#messages"), (el) => el.tagName === "button" && text(el) === "Retry"));
  }
});

test("saved answers have no running loader, including interrupted answers", () => {
  const b = browser();
  b.run(`showHistory({ messages: [{ role: "user", content: "q" },
    { role: "assistant", content: "answer", meta: { sources: [] } },
    { role: "assistant", content: "partial", meta: { sources: [], interrupted: true } }] })`);
  const assistants = b.$("#messages").children.filter((el) => el.className === "msg assistant");
  assert.equal(assistants.length, 2);
  for (const el of assistants) assert.equal(find(el, (child) => child.className === "answer-activity").attrs.hidden, "");
});

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

test("a long document warning shows two lines until More opens it; an error always shows in full", () => {
  const b = browser();
  const long = "Table p11_table11, column Fact Check: mostly numbers stored as text; ".repeat(4);
  b.run(`state.docs = [
    { doc_id: "d1", name: "a.pdf", doc_type: "pdf", status: "ready", n_chunks: 3, tables: [], warnings: [${JSON.stringify(long)}] },
    { doc_id: "d2", name: "b.pdf", doc_type: "pdf", status: "ready", n_chunks: 3, tables: [], warnings: ["short note"] },
    { doc_id: "d3", name: "c.pdf", doc_type: "pdf", status: "failed", n_chunks: 0, tables: [], warnings: [], error: ${JSON.stringify(long)} },
  ]; renderDocs();`);
  const walk = (el, out = []) => { if (el instanceof Object && el.children) { out.push(el); el.children.forEach((c) => walk(c, out)); } return out; };
  const notes = () => walk(b.$("#docs")).filter((el) => /(^| )doc-note( |$)/.test(el.className));
  const more = () => walk(b.$("#docs")).filter((el) => el.className === "link note-more");
  let [n1, n2, n3] = notes();
  assert.equal(n1.className, "doc-note clamp");
  assert.equal(n1.attrs.title, long);
  assert.equal(n2.className, "doc-note");
  assert.equal(n3.className, "doc-note err");
  assert.deepEqual(more().map((x) => x.textContent), ["More"]);
  more()[0].listeners.click[0]();
  [n1] = notes();
  assert.equal(n1.className, "doc-note");
  assert.equal(n1.attrs.title, undefined);
  assert.deepEqual(more().map((x) => [x.textContent, x.attrs["aria-expanded"]]), [["Less", "true"]]);
  more()[0].listeners.click[0]();
  assert.equal(notes()[0].className, "doc-note clamp");
});

// New versions: a re-upload of a file (same name, new content) replaces the old document once it is ready.
const v1 = "1".repeat(32), v2 = "2".repeat(32), other = "3".repeat(32);
const csv = (doc_id, status, name = "sales.csv") => ({ doc_id, name, doc_type: "spreadsheet", status,
  n_chunks: status === "ready" ? 4 : 0, tables: [], warnings: [], progress: "", error: status === "failed" ? "parse error" : null });

test("a ticked file stays on its old version while the new one is processed, then follows it", async () => {
  const lists = [[csv(v1, "ready"), csv(other, "ready", "regions.csv")],
    [csv(v2, "processing"), csv(v1, "ready"), csv(other, "ready", "regions.csv")],
    [csv(v2, "ready"), csv(other, "ready", "regions.csv")]];
  let n = 0;
  const b = browser({ "/documents": () => [0, lists[Math.min(n++, 2)]] });
  b.run("state.me = { user_id: 'u1' }");
  await b.run("refreshDocs()");
  b.run(`toggleSelect("${v1}", true); toggleSelect("${other}", true)`);
  await b.run("refreshDocs()");
  assert.deepEqual([...b.run("state.selected")], [v1, other]);  // the old version answers meanwhile
  const old = b.$("#docs").children[1];
  assert.match(text(old), /A newer upload of this file is being processed\. This version is used until it is ready\./);
  const reindex = old.children[2].children[0];
  assert.ok(reindex.attrs.disabled !== undefined);
  assert.equal(reindex.attrs.title, "Waits for the newer upload of this file");
  await b.run("refreshDocs()");
  assert.deepEqual([...b.run("state.selected")], [v2, other]);
  b.run("clearTimeout(state.pollTimer)");
});

test("a failed old version offers no Retry while a newer upload of it is processed", () => {
  const b = browser();
  b.run(`state.docs = [${JSON.stringify(csv(v2, "queued"))}, ${JSON.stringify(csv(v1, "failed"))}]; renderDocs()`);
  const old = b.$("#docs").children[1];
  assert.equal(find(old, (el) => el.tagName === "button" && text(el) === "Retry"), undefined);
  assert.match(text(old), /A newer upload of this file is being processed\.$/);
});

test("a ticked file deleted elsewhere is not followed to a new upload of its name that is not ready", async () => {
  const b = browser({ "/documents": [0, [csv(v2, "processing")]] });
  b.run(`state.me = { user_id: 'u1' }; state.docs = [${JSON.stringify(csv(v1, "ready"))}]; rememberNames([["${v1}", "sales.csv"]]);
    state.selected = new Set(["${v1}"])`);
  await b.run("refreshDocs()");  // v1 deleted elsewhere while v2 is not ready: nothing usable of that name
  assert.deepEqual([...b.run("state.selected")], []);
  b.run("clearTimeout(state.pollTimer)");
});

test("a saved chat and Retry restore a replaced file as its new version", async () => {
  const thread = "a".repeat(32);
  const detail = { thread: { thread_id: thread, doc_ids: [v1, other] }, messages: [
    { role: "user", content: "total?" },
    { role: "assistant", content: "42 [1]", meta: { sources: [{ n: 1, doc_id: v1, doc_name: "sales.csv", kind: "table", text: "x" }],
      doc_ids: [v1, other], mode: "fast", error: "llm: busy" } }] };
  const b = browser({ [`/threads/${thread}`]: [0, detail], "/threads": [0, []], "/chat": [0, events(["sources", { route: "sql", sources: [] }],
    ["done", { citations: [], invalid_citations: [], timings: {} }])] });
  b.run(`state.docs = [${JSON.stringify(csv(v2, "ready"))}, ${JSON.stringify(csv(other, "failed", "regions.csv"))}]`);
  await b.run(`openThread("${thread}")`);
  assert.deepEqual([...b.run("state.selected")], [v2]);  // regions.csv is listed but failed: left out
  b.run("state.selected = new Set()");
  find(b.$("#messages"), (el) => el.tagName === "button" && text(el) === "Retry").listeners.click[0]();
  await b.wait(20);
  assert.deepEqual(b.requests.find((r) => r.url.endsWith("/chat")).body.doc_ids, [v2]);
});

test("a question about a document replaced meanwhile says so, and its Retry uses the new version", async () => {
  let chats = 0;
  const b = browser({ "/documents": [0, [csv(v2, "ready")]], "/threads": [0, []],
    "/chat": () => (++chats === 1 ? [0, { detail: `unknown documents: ${v1}` }, 404]
      : [0, events(["sources", { route: "sql", sources: [] }], ["done", { citations: [], invalid_citations: [], timings: {} }])]) });
  b.run(`state.me = { user_id: 'u1' }; state.threadId = "a".repeat(32); state.docs = [${JSON.stringify(csv(v1, "ready"))}];
    rememberNames([["${v1}", "sales.csv"]]); state.selected = new Set(["${v1}"])`);
  await b.run("ask('total?')");
  assert.match(text(b.$("#messages")), /a selected document was deleted or replaced by a new version meanwhile/);
  assert.deepEqual([...b.run("state.selected")], [v2]);
  find(b.$("#messages"), (el) => el.tagName === "button" && text(el) === "Retry").listeners.click[0]();
  await b.wait(20);
  assert.deepEqual(b.requests.filter((r) => r.url.endsWith("/chat")).map((r) => r.body.doc_ids), [[v1], [v2]]);
  b.run("clearTimeout(state.pollTimer)");
});

test("the upload message says when the content is already there, and when a file replaces its older version", () => {
  const b = browser();
  const doc = (name, status) => ({ doc_id: v1, name, status });
  const line = (r) => b.run(`uploadLine(${JSON.stringify(r)})`);
  assert.equal(text(line({ name: "sales.csv", document: doc("sales.csv", "ready"), existing: true })),
    "sales.csv: already uploaded, unchanged (ready)");
  assert.equal(text(line({ name: "copy.csv", document: doc("sales.csv", "ready"), existing: true })),
    "copy.csv: the same content is already uploaded as sales.csv");
  const replacing = line({ name: "sales.csv", document: doc("sales.csv", "queued"), replaces: true });
  assert.equal(text(replacing), "sales.csv: queued, replaces the current version when ready");
  assert.equal(replacing.className, "upload-status");  // dropped once the file is ready, like any status line
  assert.equal(line({ name: "x.csv", error: "file is empty" }).className, "err");
});

test("a new version's upload line stays until that version is ready, though the old one of its name is ready", async () => {
  // followed by name, the old version of the name (ready already) would settle the line at once
  let calls = 0;
  let list = [csv(v1, "ready")];
  const b = browser({ "/documents": () => (++calls === 1  // the upload (POST), then the list (GET)
    ? [0, [{ name: "sales.csv", document: csv(v2, "queued"), existing: false, replaces: true }]] : [0, list]) });
  b.run(`var FormData = class { append() {} toString() { return "null"; } };
    state.me = { user_id: "u1" }; state.docs = ${JSON.stringify(list)}`);
  await b.run("upload([{ name: 'sales.csv' }])");
  await b.wait(20);
  assert.match(text(b.$("#upload-msg")), /sales\.csv: queued, replaces the current version when ready/);
  assert.equal(b.run("state.lastUpload?.size"), 1);  // not settled: the old version's "ready" is not this upload's
  list = [csv(v2, "processing"), csv(v1, "ready")];
  await b.run("refreshDocs()");
  assert.equal(b.run("state.lastUpload?.size"), 1);
  list = [csv(v2, "ready")];
  await b.run("refreshDocs()");
  assert.equal(b.run("state.lastUpload"), null);  // settled once the new version itself is ready
  b.run("clearTimeout(state.pollTimer)");
});
