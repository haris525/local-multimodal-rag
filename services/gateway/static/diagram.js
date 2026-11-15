"use strict";

// Served with its own restrictive CSP and embedded with sandbox="allow-scripts" (no same-origin).
// The chat's CSP stays strict; only this isolated document permits Mermaid's generated styles.
let rendering = false;
let pending = null;
let sequence = 0;
const diagram = document.getElementById("diagram");
const reportSize = () => parent.postMessage({ type: "mermaid-rendered", height: document.documentElement.scrollHeight }, "*");

async function draw() {
  if (rendering) return;
  rendering = true;
  while (pending) {
    const { source, dark } = pending;
    pending = null;
    try {
      document.documentElement.dataset.theme = dark ? "dark" : "light";
      mermaid.initialize({ startOnLoad: false, securityLevel: "strict", htmlLabels: false,
        theme: dark ? "dark" : "default", fontFamily: "system-ui, sans-serif",
        maxTextSize: 20000, maxEdges: 200, suppressErrorRendering: true,
        secure: ["secure", "securityLevel", "startOnLoad", "maxTextSize", "maxEdges", "suppressErrorRendering",
          "htmlLabels", "dompurifyConfig", "theme", "themeVariables", "themeCSS", "fontFamily"] });
      const { svg } = await mermaid.render(`diagram-${++sequence}`, source);
      diagram.innerHTML = DOMPurify.sanitize(svg, { USE_PROFILES: { svg: true, svgFilters: true },
        FORBID_TAGS: ["script", "foreignObject", "image", "a"] });
      // Keep natural dimensions, with scrollbars for large graphs rather than unreadably tiny labels.
      const image = diagram.querySelector("svg");
      if (!image) throw new Error("No diagram");
      const width = image.viewBox.baseVal.width;
      if (width > 0) image.setAttribute("width", String(Math.ceil(width)));
      reportSize();
    } catch {
      diagram.replaceChildren();
      parent.postMessage({ type: "mermaid-error" }, "*");
    }
  }
  rendering = false;
}

addEventListener("message", (event) => {
  if (event.source !== parent || event.data?.type !== "render-mermaid") return;
  if (typeof event.data.source !== "string" || event.data.source.length > 20000) {
    parent.postMessage({ type: "mermaid-error" }, "*");
    return;
  }
  pending = { source: event.data.source, dark: event.data.dark === true };
  draw();
});
