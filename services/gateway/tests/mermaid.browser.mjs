// Real Chromium check against a running gateway; no login, model requests or user data.
// CHROME_BIN=/path/to/chrome node services/gateway/tests/mermaid.browser.mjs http://localhost:8000
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const origin = new URL(process.argv[2] || "http://localhost:8000").origin;
const chrome = process.env.CHROME_BIN || (process.platform === "darwin"
  ? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" : "chromium");
const profile = await mkdtemp(join(tmpdir(), "local-rag-mermaid-browser-"));
const child = spawn(chrome, ["--headless=new", "--disable-gpu", "--no-first-run", "--disable-sync",
  "--disable-background-networking", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"],
{ stdio: ["ignore", "ignore", "pipe"] });
let ws;
try {
  const endpoint = await new Promise((resolve, reject) => {
    let stderr = "";
    const timer = setTimeout(() => reject(new Error("Chrome did not start")), 15000);
    child.on("error", (err) => { clearTimeout(timer); reject(err); });
    child.stderr.on("data", (data) => {
      stderr += data;
      const match = stderr.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) { clearTimeout(timer); resolve(match[1]); }
    });
  });
  ws = new WebSocket(endpoint);
  await new Promise((resolve, reject) => { ws.addEventListener("open", resolve); ws.addEventListener("error", reject); });
  let sequence = 0;
  const waiting = new Map(), exceptions = [], childSessions = new Set(), externalOrigins = new Set();
  const call = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => { waiting.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 15000);
    waiting.set(id, { resolve, reject, timer });
    ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
  ws.addEventListener("message", ({ data }) => {
    const message = JSON.parse(data);
    if (message.id && waiting.has(message.id)) {
      const task = waiting.get(message.id); waiting.delete(message.id); clearTimeout(task.timer);
      if (message.error) task.reject(new Error(message.error.message)); else task.resolve(message.result);
    } else if (message.method === "Runtime.exceptionThrown") exceptions.push(message.params.exceptionDetails.text);
    else if (message.method === "Target.attachedToTarget" && message.params.targetInfo.type === "iframe") {
      childSessions.add(message.params.sessionId);
      call("Network.enable", {}, message.params.sessionId).catch(() => {});
    } else if (message.method === "Network.requestWillBeSent" && /^https?:/.test(message.params.request.url)) {
      const requestOrigin = new URL(message.params.request.url).origin;
      if (requestOrigin !== origin) externalOrigins.add(requestOrigin);
    }
  });
  const { targetId } = await call("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await call("Target.attachToTarget", { targetId, flatten: true });
  await call("Target.setAutoAttach", { autoAttach: true, waitForDebuggerOnStart: false, flatten: true }, sessionId);
  await call("Page.enable", {}, sessionId);
  await call("Runtime.enable", {}, sessionId);
  await call("Network.enable", {}, sessionId);
  await call("Emulation.setDeviceMetricsOverride", { width: 1400, height: 1000, deviceScaleFactor: 1, mobile: false }, sessionId);
  // Exercise the real static UI while skipping only its authenticated startup. Backend auth remains intact.
  await call("Page.addScriptToEvaluateOnNewDocument", { source: `
    const original = document.addEventListener.bind(document);
    document.addEventListener = (type, listener, ...rest) => {
      if (type !== "DOMContentLoaded" || listener.name !== "init") original(type, listener, ...rest);
    };` }, sessionId);
  const evaluate = async (expression, sid = sessionId, contextId) => {
    const result = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true,
      ...(contextId ? { contextId } : {}) }, sid);
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  };
  const until = async (check, description) => {
    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) { if (await check()) return; await new Promise((r) => setTimeout(r, 100)); }
    throw new Error(`Timed out: ${description}`);
  };
  await call("Page.navigate", { url: origin }, sessionId);
  await until(() => evaluate('document.readyState === "complete" && typeof renderDiagrams === "function"'), "UI load");
  const source = await readFile(new URL("fixtures/bill-passing.mmd", import.meta.url), "utf8");
  const display = async (code) => {
    const markdown = `\`\`\`mermaid\n${code}\n\`\`\`\n\n### Legend\n- Decisions use diamonds. See the source [1].`;
    const history = { messages: [{ role: "user", content: "Show a diagram with a legend" },
      { role: "assistant", content: markdown, meta: { route: "search", sources: [{ n: 1, doc_name: "reference.txt" }] } }] };
    await evaluate(`showHistory(${JSON.stringify(history)}); document.querySelector(".mermaid-card").scrollIntoView();`);
    await until(() => evaluate('document.querySelector(".mermaid-status").hidden || document.querySelector(".mermaid-code").open'), "diagram rendering");
  };
  const inDiagram = async (expression) => {
    for (const sid of childSessions) {
      try { return await evaluate(expression, sid); } catch { /* frame replaced by the next fixture */ }
    }
    const { frameTree } = await call("Page.getFrameTree", {}, sessionId);
    const frame = frameTree.childFrames?.find((f) => f.frame.url.endsWith("/diagram.html"));
    assert.ok(frame, "diagram frame exists");
    const { executionContextId } = await call("Page.createIsolatedWorld", { frameId: frame.frame.id, worldName: "diagram-test" }, sessionId);
    return evaluate(expression, sessionId, executionContextId);
  };
  await display(source);
  assert.equal(await evaluate('document.querySelector(".mermaid-status").hidden'), true, "the full bill diagram renders");
  assert.ok((await inDiagram('document.querySelector("svg").textContent')).includes("Becomes Law"));
  assert.equal(await evaluate('!!document.querySelector(".answer h3") && !!document.querySelector("cite.ref")'), true, "legend and citations remain");
  assert.equal(await evaluate('document.querySelector("iframe").contentDocument === null'), true, "frame has an isolated origin");
  await evaluate('document.documentElement.dataset.theme = "light"; dispatchEvent(new Event("themechange"));');
  await until(() => inDiagram('document.documentElement.dataset.theme === "light"'), "theme change");
  if (process.env.DIAGRAM_SCREENSHOT) {
    const image = await call("Page.captureScreenshot", { format: "png" }, sessionId);
    await writeFile(process.env.DIAGRAM_SCREENSHOT, Buffer.from(image.data, "base64"));
  }
  await display("sequenceDiagram\n Reader->>Agent: Ask a question\n Agent-->>Reader: Answer with sources");
  assert.ok((await inDiagram('document.querySelector("svg").textContent')).includes("Answer with sources"));
  await display('%%{init: {"securityLevel":"loose","htmlLabels":true}}%%\nflowchart TD\n A["<img src=https://example.invalid/leak onerror=parent.diagramAttack=1>"] --> B[Safe]\n click A "javascript:parent.diagramAttack=1"');
  assert.equal(await evaluate('globalThis.diagramAttack === undefined'), true);
  assert.equal(await inDiagram('!document.querySelector("#diagram script, #diagram foreignObject, #diagram image, #diagram a, #diagram [onload], #diagram [onclick]")'), true);
  assert.equal(await inDiagram('(() => { try { void document.cookie; return false; } catch { return true; } })()'), true, "diagram cannot read cookies");
  await display("not a valid Mermaid diagram");
  assert.equal(await evaluate('document.querySelector(".mermaid-code").open && document.querySelector(".mermaid-frame").hidden'), true);
  assert.equal(exceptions.length, 0, `no unhandled browser errors: ${exceptions.join(", ")}`);
  assert.equal(externalOrigins.size, 0, `no external diagram requests: ${[...externalOrigins].join(", ")}`);
  console.log("Browser checks passed: flowchart/sequence, legend/citations, theme, isolated origin, hostile markup, invalid-code fallback, no external requests.");
  await call("Browser.close");
} finally {
  ws?.close();
  child.kill("SIGTERM");
  await new Promise((resolve) => { if (child.exitCode !== null) resolve(); else { child.once("exit", resolve); setTimeout(resolve, 3000); } });
  await rm(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 100 });
}
