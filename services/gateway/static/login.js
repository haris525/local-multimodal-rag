"use strict";

// Sign-in page: sign in, or change the password when the account requires it, then go back to the app.

const CSRF = { "X-Requested-With": "local-rag" };
const JSON_HEADERS = { ...CSRF, "Content-Type": "application/json" };
const $ = (sel) => document.querySelector(sel);
let justTyped = "";  // the password from this page's sign-in, so a forced change does not ask for it again

function show(id) {
  for (const sel of ["#login-form", "#change-form", "#setup"]) $(sel).hidden = sel !== id;
}

function message(text) {
  $("#login-msg").textContent = text;
}

async function body(response) {
  try { return await response.json(); } catch { return {}; }
}

// Only a chat link (#t=<id>) is carried through the sign-in page, and only back to the user it came from:
// on a shared browser the next person to sign in starts on a clean page, not on someone else's chat id.
let signedInUser = null;

function goNext() {
  const m = location.hash.match(/^#next=(%23t%3D[0-9a-f]{32})(?:&for=([0-9a-f-]{36}))?$/);
  const mine = m && (!m[2] || m[2] === signedInUser);
  location.replace(mine ? "/" + decodeURIComponent(m[1]) : "/");
}

function showChange(session) {
  const reset = session.password_reset_by_admin_at;
  $("#pw-notice").textContent = reset
    ? `An administrator reset your password on ${new Date(reset).toLocaleString()}. Choose a new one.`
    : "Choose a new password before you continue.";
  $("#pw-current").value = justTyped;
  $("#pw-current-field").hidden = Boolean(justTyped);
  show("#change-form");
  (justTyped ? $("#pw-new") : $("#pw-current")).focus();
}

async function checkSession() {
  let response;
  try {
    response = await fetch("/api/auth/session", { headers: CSRF });
  } catch {
    show("#login-form");
    message("The server cannot be reached. Try again in a moment.");
    return;
  }
  const data = await body(response);
  if (response.ok) {
    signedInUser = data.user.user_id;
    if (data.user.must_change_password) showChange(data);
    else goNext();
  } else if (data.setup_required) {
    show("#setup");
  } else {
    show("#login-form");
    $("#login-user").focus();
  }
}

async function signIn(e) {
  e.preventDefault();
  message("");
  const button = e.submitter || $("#login-form button");
  button.disabled = true;
  try {
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ username: $("#login-user").value.trim(), password: $("#login-pass").value }),
    });
    if (response.ok) {
      const data = await body(response);
      signedInUser = data.user.user_id;
      if (!data.user.must_change_password) return goNext();
      justTyped = $("#login-pass").value;
      $("#login-pass").value = "";
      const session = await fetch("/api/auth/session", { headers: CSRF });
      showChange(session.ok ? await body(session) : {});
    } else if (response.status === 401) {
      message("Invalid username or password.");
    } else if (response.status === 429) {
      const retryAfter = Number(response.headers.get("Retry-After"));
      message(retryAfter ? `Too many attempts. Try again in ${Math.ceil(retryAfter / 60)} min.` : "Too many attempts. Try again later.");
    } else {
      message("Sign-in failed.");
    }
  } catch {
    message("Sign-in failed: the server cannot be reached.");
  } finally {
    button.disabled = false;
  }
}

async function changePassword(e) {
  e.preventDefault();
  message("");
  if ($("#pw-new").value !== $("#pw-confirm").value) {
    message("The new passwords do not match.");
    return;
  }
  const button = e.submitter || $("#change-form button");
  button.disabled = true;
  try {
    const response = await fetch("/api/auth/password", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ current_password: $("#pw-current").value, new_password: $("#pw-new").value }),
    });
    if (response.ok) {
      justTyped = "";
      return goNext();
    }
    const data = await body(response);
    if (response.status === 400) message(data.detail || "Choose a different password.");
    else if (response.status === 401 && data.detail === "current password is wrong") {
      $("#pw-current-field").hidden = false;
      message("Current password is wrong.");
    } else if (response.status === 401) {
      show("#login-form");
      message("Your session ended. Sign in again.");
    } else if (response.status === 429) message("Too many attempts. Try again later.");
    else message("The password could not be changed.");
  } catch {
    message("The server cannot be reached.");
  } finally {
    button.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("#login-form").addEventListener("submit", signIn);
  $("#change-form").addEventListener("submit", changePassword);
  checkSession();
});
