// Theme: dark by default; the toggle switches to light and the choice is remembered in this browser.
// Loaded synchronously in <head> so the page never paints in the wrong theme.
(() => {
  const root = document.documentElement;
  let choice = "dark";
  try { if (localStorage.getItem("theme") === "light") choice = "light"; } catch { /* storage blocked: stay dark */ }
  root.dataset.theme = choice;

  document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("theme-toggle");
    if (!button) return;
    const sync = () => {
      const dark = root.dataset.theme === "dark";
      button.setAttribute("aria-pressed", String(dark));
      button.title = dark ? "Switch to light mode" : "Switch to dark mode";
    };
    button.addEventListener("click", () => {
      root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch { /* applied now, not remembered */ }
      sync();
    });
    sync();
  });
})();
