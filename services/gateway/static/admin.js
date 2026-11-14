"use strict";

// Admin tools: the Users panel (create, disable, reset, unlock, delete accounts) and, in the Audit log panel,
// "Whole log" and "Security events". Loaded after app.js and account.js and uses their helpers
// ($, h, api, apiFetch, openPanel, panelShows, panel, when, state, auditCard, showAudit, secretOnce, field, JSON_HEADERS). Admins
// manage accounts; they never see another user's documents, chats or memories, and the auth service enforces the role on every call.
// Each account comes with its state and the actions this admin may take now (auth's account machine): the panel
// shows those and derives nothing itself.

const ROLE_LABEL = { admin: "ADMIN", user: "USER" };
const STATE_PILL = {
  active: ["ready", "active"],
  must_change: ["queued", "must change password"],
  disabled: ["failed", "disabled"],
  deleting: ["failed", "deleting"],
};
const DELETION_POLL_MS = 3000;
// A temporary password stays on top of the Users panel until "Done", or until the panel is opened again after
// it was shown; one that arrives while another panel is open brings the Users panel back (it is shown once).
let shownSecret = null;
let pendingDeletion = null;  // {user_id, username} of a deletion still running when its request answered

function timeOf(iso) {
  const t = new Date(iso);
  return t.toDateString() === new Date().toDateString() ? t.toLocaleTimeString() : t.toLocaleString();
}

function lastSignIn(iso) {
  return iso ? `last sign-in ${when(iso)}` : "never signed in";
}

function userPills(u) {
  const [kind, label] = STATE_PILL[u.state] || ["queued", String(u.state)];
  const pills = [h("span", { class: `pill ${kind}` }, label)];
  if (u.locked_until) pills.push(h("span", { class: "pill queued" }, `locked until ${timeOf(u.locked_until)}`));
  if (u.must_change_password && u.state === "disabled") pills.push(h("span", { class: "pill queued" }, "must change password"));
  return pills;
}

function rowButton(label, onclick, extra = "") {
  return h("button", { class: `link${extra}`, type: "button", onclick }, label);
}

// One action on one account. Errors stay in the row; success re-renders the panel (with an optional notice),
// unless another panel was opened meanwhile and there is no temporary password to show.
async function adminAction(row, request, notice) {
  const msg = row.querySelector(".user-msg");
  for (const b of row.querySelectorAll("button")) b.disabled = true;
  msg.textContent = "";
  try {
    const result = await request();
    const text = typeof notice === "function" ? notice(result) : notice;
    if (panel.mode === "Users" || (shownSecret && !shownSecret.shown)) showUsers({ notice: text });
  } catch (err) {
    for (const b of row.querySelectorAll("button")) b.disabled = false;
    msg.textContent = err.message;
  }
}

async function deleteAccount(u) {
  const response = await apiFetch(`/admin/users/${u.user_id}`, {
    method: "DELETE", headers: JSON_HEADERS, body: JSON.stringify({ confirm_username: u.username }),
  });
  if (response.status === 204) return { ok: true };
  let body = {};
  try { body = await response.json(); } catch { /* not JSON */ }
  if (response.status === 202) return { ok: false, running: true };  // still purging: the list follows it
  if (response.status === 502) return { ok: false, step: body.failed_step, detail: body.detail };
  throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
}

function deletionNotice(u, result) {
  pendingDeletion = result.running ? { user_id: u.user_id, username: u.username } : null;
  if (result.ok) return { text: `Deleted ${u.username} and all their data.` };
  if (result.running) return { text: `Deleting ${u.username}: the data is still being removed. This list updates when it finishes.` };
  return { error: true, text: `Deletion of ${u.username} did not finish (failed at ${result.step || "an unknown step"}). Use Retry delete in its row.` };
}

// The outcome of a deletion that was still running when its request answered, once the list shows it.
function settledDeletion(users) {
  if (!pendingDeletion) return null;
  const u = users.find((x) => x.user_id === pendingDeletion.user_id);
  const name = pendingDeletion.username;
  if (u?.deletion?.running) return { text: `Deleting ${name}: the data is still being removed. This list updates when it finishes.` };
  pendingDeletion = null;
  if (!u) return { text: `Deleted ${name} and all their data.` };
  return { error: true, text: `Deletion of ${name} did not finish (${u.deletion?.error || "unknown error"}). Use Retry delete in its row.` };
}

function setNotice(box, notice) {
  box.className = `acct-msg${notice?.text && !notice.error ? " ok" : ""}`;
  box.hidden = !notice?.text;
  box.textContent = notice?.text || "";
}

// Deletions still running when the list was drawn: every few seconds only their rows (and the notice) are
// brought up to date, so a half-typed new user, an open delete confirmation or a row's error stays as it is.
function followDeletions(token, rows, notice) {
  setTimeout(async () => {
    if (!panelShows(token)) return;
    let users;
    try {
      users = await api("/admin/users");
    } catch {
      return followDeletions(token, rows, notice);  // try again at the next tick
    }
    if (!panelShows(token)) return;
    const byId = new Map(users.map((u) => [u.user_id, u]));
    for (const [id, row] of [...rows]) {
      const u = byId.get(id);
      if (u?.deletion?.running) continue;
      rows.delete(id);
      if (u) row.replaceWith(userRow(u, state.me)); else row.remove();
      if (pendingDeletion?.user_id === id) setNotice(notice, settledDeletion(users));
    }
    if (rows.size) followDeletions(token, rows, notice);
  }, DELETION_POLL_MS);
}

function deletionStatus(u) {
  if (u.deletion?.running) return h("span", { class: "muted" }, "Deleting: the data services are removing its data.");
  return h("span", { class: "error" }, `Deletion did not finish${u.deletion?.error ? ` (${u.deletion.error})` : ""}.`);
}

const patchUser = (u, body) => () => api(`/admin/users/${u.user_id}`, { method: "PATCH", headers: JSON_HEADERS, body: JSON.stringify(body) });

// One button per action the server allows for this account now (admin_view's "actions").
const ACTION_BUTTON = {
  disable: (row, u) => rowButton("Disable", () => {
    if (!confirm(`Disable ${u.username}? They are signed out at once and their API tokens stop working.`)) return;
    adminAction(row, patchUser(u, { disabled: true }), { text: `${u.username} is disabled and signed out.` });
  }),
  enable: (row, u) => rowButton("Enable", () => adminAction(row, patchUser(u, { disabled: false }), { text: `${u.username} can sign in again.` })),
  make_admin: (row, u) => rowButton("Make admin", () => {
    if (!confirm(`Make ${u.username} an admin? Admins manage accounts but cannot read anyone's documents or chats.`)) return;
    adminAction(row, patchUser(u, { role: "admin" }), { text: `${u.username} is now an admin.` });
  }),
  make_user: (row, u) => rowButton("Make user", () => adminAction(row, patchUser(u, { role: "user" }), { text: `${u.username} is now a user.` })),
  reset_password: (row, u) => rowButton("Reset password", () => {
    if (!confirm(`Reset the password of ${u.username}? They are signed out everywhere and need the temporary password you give them.`)) return;
    adminAction(row, () => api(`/admin/users/${u.user_id}/reset-password`, { method: "POST" }), (result) => {
      shownSecret = { value: result.temporary_password, note: `Temporary password for ${u.username}. Shown once. Their sessions and API tokens were revoked; they choose a new password at the next sign-in.` };
      return null;
    });
  }),
  unlock: (row, u) => rowButton("Unlock", () => adminAction(row, () => api(`/admin/users/${u.user_id}/unlock`, { method: "POST" }),
    { text: `${u.username} can try to sign in again. A lock on their network address, if any, ends on its own.` })),
  delete: (row, u) => (u.state === "deleting"
    ? rowButton("Retry delete", () => adminAction(row, () => deleteAccount(u), (result) => deletionNotice(u, result)))
    : rowButton("Delete", () => confirmDelete(row, u), " danger")),
};

// Delete asks for the username in the row itself: the button stays disabled until it matches.
function confirmDelete(row, u) {
  const slot = row.querySelector(".user-confirm");
  if (slot.childElementCount) return slot.replaceChildren();
  const input = h("input", { class: "mem-input", type: "text", autocomplete: "off", autocapitalize: "none", spellcheck: "false",
    "aria-label": `Type ${u.username} to confirm` });
  const go = h("button", { class: "primary danger", type: "submit", disabled: true }, "Delete account");
  const cancel = rowButton("Cancel", () => slot.replaceChildren());
  input.addEventListener("input", () => { go.disabled = input.value.trim().toLowerCase() !== u.username; });
  const form = h("form", { class: "user-delete" },
    h("div", { class: "muted" }, "Type ", h("strong", {}, u.username),
      " to delete the account with all its documents, chats and memories. Its audit details are erased. This cannot be undone."),
    h("div", { class: "user-delete-bar" }, input, go, cancel));
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (input.value.trim().toLowerCase() !== u.username) return;
    adminAction(row, () => deleteAccount(u), (result) => deletionNotice(u, result));
  });
  slot.replaceChildren(form);
  input.focus();
}

function userRow(u, me) {
  const self = u.user_id === me.user_id;
  const row = h("div", { class: "src user-row" });
  const actions = [];
  if (self) actions.push(h("span", { class: "muted" }, "Your account: change your password from the user menu."));
  if (u.state === "deleting") actions.push(deletionStatus(u));
  for (const name of u.actions || []) {
    if (ACTION_BUTTON[name]) actions.push(ACTION_BUTTON[name](row, u));
  }
  const sessions = `${u.active_sessions} active session${u.active_sessions === 1 ? "" : "s"}`;
  row.append(
    h("div", { class: "src-head" },
      h("span", { class: "src-doc" }, u.username),
      h("span", { class: "badge" }, ROLE_LABEL[u.role] || u.role),
      self ? h("span", { class: "badge" }, "YOU") : null),
    h("div", { class: "doc-meta" }, ...userPills(u)),
    h("div", { class: "src-loc" }, `${lastSignIn(u.last_login_at)}, ${sessions}, created ${new Date(u.created_at).toLocaleDateString()}`),
    h("div", { class: "user-actions" }, ...actions),
    h("div", { class: "user-confirm" }),
    h("div", { class: "user-msg error", role: "alert" }));
  return row;
}

function createForm() {
  const [nameField, name] = field("Username", { type: "text", required: true, maxlength: "64", autocomplete: "off",
    autocapitalize: "none", spellcheck: "false", placeholder: "3-64 characters: a-z 0-9 . _ -" });
  const role = h("select", { class: "mem-input" }, h("option", { value: "user" }, "User"), h("option", { value: "admin" }, "Admin"));
  const roleField = h("label", { class: "acct-field" }, "Role", role);
  const msg = h("div", { class: "acct-msg", role: "status" });
  const submit = h("button", { class: "primary", type: "submit" }, "Create user");
  const form = h("form", { class: "acct-form" }, nameField, roleField, submit, msg);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    submit.disabled = true;
    msg.textContent = "";
    try {
      const created = await api("/admin/users", { method: "POST", headers: JSON_HEADERS,
        body: JSON.stringify({ username: name.value.trim(), role: role.value }) });
      shownSecret = { value: created.temporary_password,
        note: `Temporary password for ${created.user.username}. Shown once. Give it to them; they choose their own password at the first sign-in.` };
      showUsers();
    } catch (err) {
      msg.textContent = err.message === "username taken" ? "That username is taken." : err.message;
      submit.disabled = false;
    }
  });
  return form;
}

// fresh: opened from the menu, so a temporary password already shown and a followed deletion are not shown
// again. While a deletion runs in the background, its row follows it (followDeletions).
async function showUsers({ notice = null, fresh = false } = {}) {
  const list = $("#sources-list");
  const token = openPanel("Users");
  if (fresh) {
    if (shownSecret?.shown) shownSecret = null;
    pendingDeletion = null;
    list.replaceChildren(h("div", { class: "muted" }, "Loading..."));
  }
  let users;
  try {
    users = await api("/admin/users");
  } catch (err) {
    if (panelShows(token)) list.replaceChildren(h("div", { class: "error" }, `Could not load the accounts: ${err.message}`));
    return;
  }
  if (!panelShows(token)) return;
  const status = h("div", { role: "status" });
  setNotice(status, settledDeletion(users) || notice);
  let secret = null;
  if (shownSecret) {
    shownSecret.shown = true;
    secret = secretOnce(shownSecret.value, shownSecret.note);
    secret.querySelector(".muted").before(rowButton("Done", () => { shownSecret = null; secret.remove(); }));
  }
  const rows = users.map((u) => [u, userRow(u, state.me)]);
  list.replaceChildren(...[
    h("div", { class: "muted mem-help" },
      "Accounts on this server. Admins manage accounts but cannot read anyone's documents, chats or memories. New users and reset passwords get a temporary password that must be changed at the next sign-in."),
    secret,
    status,
    createForm(),
    ...rows.map(([, row]) => row),
  ].filter(Boolean));
  const running = new Map(rows.filter(([u]) => u.deletion?.running).map(([u, row]) => [u.user_id, row]));
  if (running.size) followDeletions(token, running, status);
}

/* ---------- audit log: whole-log check and security events (admins) ---------- */

function adminAuditTools() {
  const result = h("div", { class: "audit-admin-result", role: "status" });
  const whole = rowButton("Whole log", async () => {
    whole.disabled = true;
    result.className = "audit-admin-result muted";
    result.textContent = "Checking every account's events...";
    try {
      const check = await api("/admin/audit/verify");
      result.className = `audit-admin-result ${check.ok ? "audit-ok" : "audit-bad"}`;
      result.textContent = check.ok
        ? `Whole log verified: ${check.events} events from every account, unaltered. Chain head ${String(check.head).slice(0, 16)}.`
        : `The whole log was altered at event #${check.first_bad_seq} (numbered across all accounts).`;
    } catch (err) {
      result.className = "audit-admin-result error";
      result.textContent = `Could not check the whole log: ${err.message}`;
    } finally {
      whole.disabled = false;
    }
  });
  const security = rowButton("Security events", () => showSecurityEvents());
  return h("div", { class: "audit-admin" }, h("div", { class: "audit-admin-bar" }, whole, security), result);
}

function securityWho(e, names) {
  const d = e.detail || {};
  if (e.user_id === "anonymous") return d.username ? `unknown name "${d.username}"` : "unknown name";
  const who = e.username || "deleted account";
  if (!e.actor || e.actor === e.user_id) return who;
  const by = names[e.actor] || (e.actor === "operator" ? "the operator (auth-admin)" : e.actor === "bootstrap" ? "bootstrap" : "a deleted admin");
  return `${who}, by ${by}`;
}

function securityEntry(e, names) {
  return auditCard(`#${e.seq}`, e, securityWho(e, names));  // seq numbers the whole log, every account
}

async function showSecurityEvents() {
  const list = $("#sources-list");
  const token = openPanel("Security events");
  list.replaceChildren(h("div", { class: "muted" }, "Loading..."));
  let events, users;
  try {
    [events, users] = await Promise.all([api("/admin/audit/security?limit=200"), api("/admin/users")]);
  } catch (err) {
    if (panelShows(token)) list.replaceChildren(h("div", { class: "error" }, `Could not load the security events: ${err.message}`));
    return;
  }
  if (!panelShows(token)) return;
  const names = Object.fromEntries(users.map((u) => [u.user_id, u.username]));
  const back = rowButton("Back to my events", showAudit);
  const more = rowButton("Older events", async () => {
    more.disabled = true;
    try {
      const older = await api(`/admin/audit/security?limit=200&before=${events[events.length - 1].seq}`);
      events = events.concat(older);
      more.before(...older.map((e) => securityEntry(e, names)));
      if (older.length < 200) more.remove();
    } catch (err) {
      more.textContent = `Could not load more: ${err.message}`;
    } finally {
      more.disabled = false;
    }
  });
  list.replaceChildren(...[
    h("div", { class: "audit-admin-bar" }, back),
    h("div", { class: "muted mem-help" },
      "Sign-in and account events of every account: sign-ins, failed attempts, lockouts, API tokens and admin actions. Documents, chats and memories never appear here."),
    ...events.map((e) => securityEntry(e, names)),
    events.length ? null : h("div", { class: "mem-empty" }, "No security events yet."),
    events.length >= 200 ? more : null,
  ].filter(Boolean));
}
