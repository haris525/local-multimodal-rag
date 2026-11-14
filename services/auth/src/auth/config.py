import re
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_common.identity import Signer, load_public_keys

LOCAL_HOSTS = ("localhost", "127.0.0.1")
USERNAME_RE = re.compile(r"^[a-z0-9._-]{3,64}$")
USERNAME_RULE = "Use 3-64 characters: lowercase letters, digits, dot, underscore or hyphen."


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUTH_", extra="ignore")

    db_path: Path = Path("/data/auth.db")
    public_origin: str = "http://localhost:8000"  # compose: ${PUBLIC_ORIGIN}
    allow_insecure_http: bool = False
    idle_timeout_min: int = 720
    absolute_timeout_days: int = 7
    touch_interval_s: int = 60
    token_default_days: int = 30
    token_max_days: int = 365
    lockout_threshold: int = 5
    lockout_window_min: int = 15
    lockout_base_min: int = 15
    lockout_max_hours: int = 24
    ip_failure_limit: int = 20
    ip_window_min: int = 15
    ip_block_min: int = 15
    # Callers whose X-Real-IP is believed: host names, addresses or networks, comma-separated. Anyone else is
    # counted under its own address (another container on the backend network cannot pick addresses).
    trusted_proxies: str = "gateway"
    min_password_length: int = 12
    max_password_length: int = 256
    argon2_time_cost: int = 3
    argon2_memory_kib: int = 65536
    argon2_parallelism: int = 4
    max_concurrent_hashes: int = 4
    signing_key_file: Path = Path("/run/secrets/auth_signing_key")
    public_keys_file: Path = Path("/run/secrets/identity_public_keys")
    bootstrap_admin: str | None = None
    bootstrap_password_file: Path | None = None  # e.g. /run/secrets/bootstrap_admin_password
    ingest_url: str = "http://ingest-api:8001"
    threads_url: str = "http://threads:8006"
    audit_url: str | None = "http://audit:8004"
    rag_url: str = "http://rag:8002"
    # An admin's DELETE answers within this many seconds (202 while the purge goes on); below the gateway's
    # 120 s timeout for /api/admin/.
    admin_delete_wait_s: float = 60.0
    deletion_resume_delay_s: float = 60.0  # after start-up, then hourly: resume deletions that stopped
    log_level: str = "INFO"

    @model_validator(mode="after")
    def origin_is_safe(self) -> "Settings":
        """Fail fast: the origin decides the cookie name and the Secure flag."""
        if not re.match(r"^https?://[^/]+$", self.public_origin):
            raise ValueError("AUTH_PUBLIC_ORIGIN must look like https://host[:port], with no path or trailing slash")
        host = urlsplit(self.public_origin).hostname or ""
        if self.public_origin.startswith("http://") and host not in LOCAL_HOSTS and not self.allow_insecure_http:
            raise ValueError(f"refusing plain http for {host!r}: serve the app over https (DEPLOYMENT section 5), "
                             "or set AUTH_ALLOW_INSECURE_HTTP=true")
        if self.min_password_length < 8 or self.max_password_length < self.min_password_length:
            raise ValueError("password length limits must be 8 <= min <= max")
        return self

    @property
    def secure_cookies(self) -> bool:
        return self.public_origin.startswith("https://")

    @property
    def cookie_name(self) -> str:
        return "__Host-rag_session" if self.secure_cookies else "rag_session"


def valid_username(username: str) -> str:
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        raise ValueError(USERNAME_RULE)
    return username


def load_signer(s: Settings, clock: Callable[[], float] = time.time) -> Signer:
    """The signing key, checked against the published public key, so a stale identity_public.json fails here."""
    signer = Signer.from_file(s.signing_key_file, "auth", clock=clock)
    if load_public_keys(s.public_keys_file).get("auth") != signer.public_key_raw():
        raise RuntimeError(f"the 'auth' key in {s.public_keys_file} does not match {s.signing_key_file}; "
                           "rerun scripts/init_secrets.py --force and restart every service")
    return signer
