"""What a benchmark ran against, recorded in every report: the git commit of this checkout, the rag and llm
image ids, the model name and LLM_CTX, and rag's sampling settings when the container sets them.

Everything is read without changing anything: `git rev-parse`/`git status` and `docker ps`/`docker inspect` of the
compose project's containers. Only an allow-list of non-secret settings is copied from the rag container's
environment (secrets live in files, never in its environment, but the list keeps reports free of anything else).
A value that cannot be read is None, and the report says unknown; nothing fails because Docker is not reachable
(for example inside the eval container)."""

import json
import re
import subprocess
from collections.abc import Callable

from rag_eval.paths import REPO_ROOT

SAFE_SETTING = re.compile(r"^RAG_(LLM_MODEL|LLM_CTX|LLM_PARALLEL|TEMPERATURE|TOP_P|TOP_K|THINK_TEMPERATURE|THINK_TOP_P|"
                          r"AGENT_[A-Z_]+|[A-Z_]*SEED[A-Z_]*)$")
Runner = Callable[..., subprocess.CompletedProcess]


def _out(args: list[str], runner: Runner) -> str | None:
    try:
        done = runner(args, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _container(project: str, service: str, runner: Runner) -> tuple[str | None, dict[str, str]]:
    """(image id, environment) of the project's container for this compose service."""
    ids = _out(["docker", "ps", "--filter", f"label=com.docker.compose.project={project}",
                "--filter", f"label=com.docker.compose.service={service}", "--format", "{{.ID}}"], runner)
    if not ids:
        return None, {}
    raw = _out(["docker", "inspect", "--format", "{{.Image}}\t{{json .Config.Env}}", ids.splitlines()[0]], runner)
    if not raw or "\t" not in raw:
        return None, {}
    image, env_json = raw.split("\t", 1)
    try:
        pairs = json.loads(env_json) or []
    except ValueError:
        pairs = []
    return image, dict(p.split("=", 1) for p in pairs if "=" in p)


def detect(project: str = "local-rag", app_commit: str | None = None, runner: Runner = subprocess.run) -> dict:
    env: dict = {"git_commit": None, "app_commit": app_commit, "rag_image": None, "llm_image": None, "model": None,
                 "llm_ctx": None, "rag_settings": {}, "compose_project": project}
    if REPO_ROOT:
        head = _out(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], runner)
        if head:
            dirty = _out(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"], runner)
            env["git_commit"] = head + ("-dirty" if dirty else "")
    image, settings = _container(project, "rag", runner)
    if image:
        env["rag_image"] = image
        env["rag_settings"] = {k: v for k, v in sorted(settings.items()) if SAFE_SETTING.match(k)}
        env["model"] = settings.get("RAG_LLM_MODEL")
        env["llm_ctx"] = settings.get("RAG_LLM_CTX")
    env["llm_image"] = _container(project, "llm", runner)[0]
    return env
