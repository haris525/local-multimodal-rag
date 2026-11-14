"use strict";

// The account menu in the header: change password, API tokens, Users (admins, admin.js), sign out. Loaded after
// app.js and uses its helpers ($, h, api, apiFetch, openPanel, panelShows, when, state). init() in app.js calls
// renderUserMenu().

const JSON_HEADERS = { "Content-Type": "application/json" };

function renderUserMenu() {
  const menu = $("#user-menu");
  const close = () => { menu.open = false; };
  $("#user-name").textContent = state.me.username;
  $("#users-open").hidden = state.me.role !== "admin";  // the auth service checks the role on every admin call
  $("#users-open").addEventListener("click", () => { close(); showUsers({ fresh: true }); });
  $("#password-open").addEventListener("click", () => { close(); showPassword(); });
  $("#tokens-open").addEventListener("click", () => { close(); showTokens(); });
  $("#signout").addEventListener("click", () => { close(); signOut(); });
  document.addEventListener("click", (e) => { if (!menu.contains(e.target)) close(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
}

async function signOut() {
  state.leaving = true;
  try { await fetch("/api/auth/logout", { method: "POST", headers: CSRF }); } catch { /* signed out here anyway */ }
  try {
    for (const key of Object.keys(localStorage)) if (key.startsWith("selected:")) localStorage.removeItem(key);
  } catch { /* storage unavailable */ }
  location.replace("/login.html");
}

function field(label, attrs) {
  const input = h("input", { class: "mem-input", ...attrs });
  return [h("label", { class: "acct-field" }, label, input), input];
}

function secretOnce(value, note) {
  const copy = h("button", { class: "link", type: "button" }, "Copy");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(value);
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "Copy failed: select the text";
    }
  });
  return h("div", { class: "secret-once", role: "status" }, h("code", {}, value), copy, h("div", { class: "muted" }, note));
}

function showPassword() {
  openPanel("Change password");
  const [currentField, current] = field("Current password", { type: "password", autocomplete: "current-password", required: true, maxlength: "1024" });
  const [nextField, next] = field("New password (at least 12 characters)", { type: "password", autocomplete: "new-password", required: true, maxlength: "256" });
  const [confirmField, confirm] = field("Repeat the new password", { type: "password", autocomplete: "new-password", required: true, maxlength: "256" });
  const msg = h("div", { class: "acct-msg", role: "status" });
  const submit = h("button", { class: "primary", type: "submit" }, "Change password");
  const form = h("form", { class: "acct-form" }, currentField, nextField, confirmField, submit, msg);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    msg.className = "acct-msg";
    if (next.value !== confirm.value) {
      msg.textContent = "The new passwords do not match.";
      return;
    }
    submit.disabled = true;
    try {
      await api("/auth/password", { method: "POST", headers: JSON_HEADERS,
        body: JSON.stringify({ current_password: current.value, new_password: next.value }) }, { keep401: true });
      form.reset();
      msg.className = "acct-msg ok";
      msg.textContent = "Password changed. Your other sessions and API tokens were signed out.";
    } catch (err) {
      msg.textContent = err.message === "current password is wrong" ? "Current password is wrong." : err.message;
    } finally {
      submit.disabled = false;
    }
  });
  $("#sources-list").replaceChildren(
    h("div", { class: "muted mem-help" }, "Long and memorable beats short and complex. Changing it signs out your other sessions and revokes your API tokens."),
    form);
}

async function showTokens(created = null) {
  const list = $("#sources-list");
  const token = openPanel("API tokens");
  list.replaceChildren(h("div", { class: "muted" }, "Loading..."));
  let tokens;
  try {
    tokens = await api("/auth/tokens");
  } catch (err) {
    if (panelShows(token)) list.replaceChildren(h("div", { class: "error" }, `Could not load your tokens: ${err.message}`));
    return;
  }
  if (!panelShows(token)) return;
  const [nameField, name] = field("Name", { type: "text", required: true, maxlength: "60", placeholder: "for example: eval" });
  const [daysField, days] = field("Valid for (days)", { type: "number", required: true, min: "1", max: "365", value: "30" });
  const [passField, pass] = field("Your password", { type: "password", autocomplete: "current-password", required: true, maxlength: "1024" });
  const msg = h("div", { class: "acct-msg", role: "status" });
  const submit = h("button", { class: "primary", type: "submit" }, "Create token");
  const form = h("form", { class: "acct-form" }, nameField, daysField, passField, submit, msg);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    submit.disabled = true;
    try {
      const token = await api("/auth/tokens", { method: "POST", headers: JSON_HEADERS,
        body: JSON.stringify({ name: name.value.trim(), days: Number(days.value), current_password: pass.value }) }, { keep401: true });
      showTokens(token);
    } catch (err) {
      msg.textContent = err.message === "current password is wrong" ? "Your password is wrong." : err.message;
      submit.disabled = false;
    }
  });
  const rows = tokens.map((t) => {
    const revoke = h("button", { class: "link", type: "button" }, "Revoke");
    revoke.addEventListener("click", async () => {
      if (!confirm(`Revoke the token "${t.name}"? Scripts using it stop working.`)) return;
      try { await api(`/auth/tokens/${t.token_id}`, { method: "DELETE" }); } catch (err) { alert(err.message); }
      showTokens();
    });
    return h("div", { class: "src token-row" },
      h("div", { class: "mem-text" }, t.name),
      h("div", { class: "src-loc" }, `created ${when(t.created_at)} - last used ${when(t.last_used_at)} - expires ${new Date(t.expires_at).toLocaleDateString()}`),
      revoke);
  });
  const help = "Tokens let scripts (smoke tests, evaluation) use the API as you: send them as 'Authorization: Bearer <token>'. They cannot manage accounts or tokens.";
  list.replaceChildren(...[
    h("div", { class: "muted mem-help" }, help),
    created ? secretOnce(created.token, `Shown once. Copy it now: set RAG_TOKEN or EVAL_TOKEN in your shell, never in .env. Expires ${new Date(created.expires_at).toLocaleDateString()}.`) : null,
    form,
    ...rows,
    rows.length ? null : h("div", { class: "mem-empty" }, "No tokens yet."),
  ].filter(Boolean));
}
