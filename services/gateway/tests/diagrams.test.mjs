import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

const STATIC = new URL("../static/", import.meta.url);

class El {
  constructor(tag, attrs = {}, ...children) {
    Object.assign(this, { tag, attrs, children: [], hidden: false, open: false, listeners: {}, sent: [] });
    if (tag === "iframe") this.contentWindow = { postMessage: (...args) => this.sent.push(args) };
    this.append(...children);
  }
  get isConnected() { return this.connected || !!this.parentElement?.isConnected; }
  append(...children) { for (const child of children) { if (child instanceof El) child.parentElement = this; this.children.push(child); } }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  replaceWith(node) {
    const parent = this.parentElement;
    parent.children[parent.children.indexOf(this)] = node;
    node.parentElement = parent;
    this.parentElement = null;
  }
  insertBefore(node, next) { this.children.splice(this.children.indexOf(next), 0, node); node.parentElement = this; }
  closest() { return (this.className || this.attrs.class) === "mermaid-card" ? this : this.parentElement?.closest() || null; }
  get textContent() { return this.children.map((c) => typeof c === "string" ? c : c.textContent).join(""); }
  set textContent(value) { this.children = [value]; }
  querySelectorAll(selector) {
    const found = [];
    const walk = (el) => {
      if (selector === "pre > code.language-mermaid" && el.tag === "code") found.push(el);
      if (selector === "iframe.mermaid-frame" && el.tag === "iframe") found.push(el);
      for (const child of el.children) if (child instanceof El) walk(child);
    };
    walk(this);
    return found;
  }
}

function browser(...sources) {
  const root = new El("main"); root.connected = true;
  for (const source of sources) root.append(new El("pre", {}, new El("code", { class: "language-mermaid" }, source)));
  const listeners = {}, timers = new Map();
  const document = { documentElement: { dataset: { theme: "dark" } }, querySelectorAll: (s) => root.querySelectorAll(s),
    createElement: (tag) => new El(tag), addEventListener() {} };
  const context = vm.createContext({ document, h: (tag, attrs, ...children) => new El(tag, attrs, ...children),
    addEventListener: (type, fn) => { listeners[type] = fn; },
    setTimeout: (fn) => { const id = {}; timers.set(id, fn); return id; }, clearTimeout: (id) => timers.delete(id) });
  vm.runInContext(readFileSync(new URL("app.js", STATIC), "utf8"), context);
  const render = () => { context.root = root; vm.runInContext("renderChatDiagrams(root)", context); };
  return { root, render, listeners, timers, document, frames: () => root.querySelectorAll("iframe.mermaid-frame") };
}

test("Mermaid keeps the original code and loads a local, opaque-origin frame only once", () => {
  const source = "graph TD\n A[Bill] --> B[Law]";
  const b = browser(source); b.render();
  const frame = b.frames()[0], card = b.root.children[0];
  assert.equal(frame.attrs.src, "/diagram.html");
  assert.equal(frame.attrs.sandbox, "allow-scripts");
  assert.equal(frame.attrs.loading, "lazy");
  assert.equal(b.timers.size, 0, "offscreen diagrams do not time out before lazy loading");
  assert.match(card.textContent, /View diagram code/);
  assert.ok(card.textContent.endsWith(source));
  frame.listeners.load();
  assert.equal(frame.sent[0][0].source, source);
  assert.equal(frame.sent[0][0].dark, true);
  assert.equal(frame.sent[0][1], "*", "opaque sandbox requires a wildcard target, guarded by Window identity");
  assert.equal(b.timers.size, 1);
  b.render();
  assert.equal(b.frames().length, 1);
});

test("diagram messages require their own connected frame and a finite, bounded height", () => {
  const b = browser("graph TD\n A-->B"); b.render();
  const frame = b.frames()[0], status = b.root.children[0].children[0];
  frame.listeners.load();
  const message = (source, origin, height) => b.listeners.message({ source, origin, data: { type: "mermaid-rendered", height } });
  message({}, "null", 500);
  message(frame.contentWindow, "https://example.invalid", 500);
  message(frame.contentWindow, "null", NaN);
  assert.equal(status.hidden, false);
  message(frame.contentWindow, "null", 100000);
  assert.equal(frame.attrs.height, "640");
  assert.equal(status.hidden, true);
  assert.equal(b.timers.size, 0);
  frame.parentElement = null;
  message(frame.contentWindow, "null", 200);
  assert.equal(frame.attrs.height, "640", "late messages cannot affect a removed answer");
});

test("invalid or failed diagrams reveal code, and theme changes resend the source", () => {
  for (const timeout of [false, true]) {
    const b = browser("not a diagram"); b.render();
    const frame = b.frames()[0], card = b.root.children[0];
    frame.listeners.load();
    b.document.documentElement.dataset.theme = "light";
    b.listeners.themechange();
    assert.equal(frame.sent.at(-1)[0].dark, false);
    if (timeout) [...b.timers.values()][0]();
    else b.listeners.message({ source: frame.contentWindow, origin: "null", data: { type: "mermaid-error" } });
    assert.equal(frame.hidden, true);
    assert.equal(card.children.at(-1).open, true);
    assert.match(card.textContent, /could not be displayed.*not a diagram/s);
    assert.equal(b.timers.size, 0);
  }
});

test("diagram text size and count are bounded without discarding the source", () => {
  const b = browser(...Array(6).fill("graph TD\n A-->B")); b.render();
  assert.equal(b.frames().length, 4);
  assert.match(b.root.children[5].textContent, /first four.*graph TD/s);
  const big = browser("x".repeat(20001)); big.render();
  assert.equal(big.frames().length, 0);
  assert.ok(big.root.textContent.endsWith("x".repeat(20001)));
});

test("only the isolated diagram document allows inline styles; chat keeps its strict policy", () => {
  const nginx = readFileSync(new URL("../default.conf.template", STATIC), "utf8");
  const location = nginx.match(/location = \/diagram\.html \{([\s\S]*?)\n    \}/)[1];
  assert.match(location, /sandbox allow-scripts/);
  assert.doesNotMatch(location, /allow-same-origin|unsafe-eval/);
  for (const part of ["connect-src 'none'", "img-src 'none'", "font-src 'none'", "frame-ancestors 'self'", "form-action 'none'"]) {
    assert.ok(location.includes(part));
  }
  assert.equal((nginx.match(/unsafe-inline/g) || []).length, 1);
  for (const name of ["X-Content-Type-Options", "Referrer-Policy", "Content-Security-Policy"]) assert.ok(location.includes(name));
});
