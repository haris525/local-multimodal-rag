"use strict";

// Each diagram runs in an opaque-origin frame, with no access to the chat, cookies or network.
const diagramFrames = new WeakMap();
const DIAGRAM_LIMIT = 4;
const DIAGRAM_TEXT_LIMIT = 20000;

function sendDiagram(frame) {
  const item = diagramFrames.get(frame.contentWindow);
  if (!item || !frame.isConnected) return;
  frame.contentWindow.postMessage({ type: "render-mermaid", source: item.source,
    dark: document.documentElement.dataset.theme === "dark" }, "*");  // opaque sandbox origin
}

addEventListener("message", (event) => {
  const item = diagramFrames.get(event.source);
  if (!item || !item.frame.isConnected || event.origin !== "null") return;
  if (event.data?.type === "mermaid-rendered" && Number.isFinite(event.data.height)) {
    clearTimeout(item.timer);
    item.frame.hidden = false;
    item.frame.setAttribute("height", Math.max(120, Math.min(640, Math.ceil(event.data.height))));
    item.status.hidden = true;
  } else if (event.data?.type === "mermaid-error") {
    item.fail();
  }
});

addEventListener("themechange", () => {
  for (const frame of document.querySelectorAll("iframe.mermaid-frame")) sendDiagram(frame);
});

// Run after the answer settles or a saved answer is opened. Re-rendering Markdown for each streamed token
// must not recreate frames or ask Mermaid to parse incomplete fences on every token.
function renderDiagrams(root) {
  let count = 0;
  for (const code of root.querySelectorAll("pre > code.language-mermaid")) {
    const pre = code.parentElement;
    if (pre.closest(".mermaid-card")) continue;
    const source = code.textContent;
    const status = h("div", { class: "mermaid-status", role: "status" }, "Rendering diagram…");
    const details = h("details", { class: "mermaid-code" }, h("summary", {}, "View diagram code"));
    const card = h("figure", { class: "mermaid-card" }, status, details);
    pre.replaceWith(card);
    details.append(pre);  // retain the exact, text-only code and its fallback
    if (++count > DIAGRAM_LIMIT || source.length > DIAGRAM_TEXT_LIMIT) {
      status.textContent = count > DIAGRAM_LIMIT ? "Only the first four diagrams are displayed. More diagrams are available as code."
        : "This diagram is too large to display here. Its code is below.";
      details.open = true;
      continue;
    }
    const frame = h("iframe", { class: "mermaid-frame", src: "/diagram.html", sandbox: "allow-scripts",
      title: "Mermaid diagram", height: "180", loading: "lazy", referrerpolicy: "no-referrer" });
    const item = { frame, source, status, timer: null, fail() {
      clearTimeout(item.timer);
      frame.hidden = true;
      status.hidden = false;
      status.textContent = "This diagram could not be displayed. Its code is below.";
      details.open = true;
    } };
    frame.addEventListener("load", () => {
      if (!frame.isConnected) return;
      clearTimeout(item.timer);
      item.timer = setTimeout(item.fail, 15000);
      sendDiagram(frame);
    });
    card.insertBefore(frame, details);
    diagramFrames.set(frame.contentWindow, item);
  }
}
