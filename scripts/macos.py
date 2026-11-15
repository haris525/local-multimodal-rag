"""Apple Silicon launcher. No Python packages needed for this helper; see docs/MACOS.md.

Native inference stays in two foreground terminals; Docker owns the application and data.
This helper never starts/stops other projects, edits Docker Desktop settings, or deletes volumes.
"""

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_REVISION = "a956a852c9c8bc1fb07206364838ddef299178c0"
LLAMA_SOURCE = ROOT / ".runtime" / "llama.cpp"
LLAMA_SERVER = LLAMA_SOURCE / "build" / "bin" / "llama-server"
GGUF = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
MMPROJ = "mmproj-BF16.gguf"
HF_MODELS = ("Qwen3-Embedding-0.6B", "Qwen3-VL-Embedding-2B", "Qwen3-Reranker-0.6B")


def environment():
    env = os.environ.copy()
    env.setdefault("COMPOSE_PARALLEL_LIMIT", "2")  # bound simultaneous first-build jobs, too
    env.setdefault("HF_HUB_DISABLE_XET", "1")  # regular HTTPS; avoids Xet's extra memory/threads
    env["MAC_MODELS_DIR"] = str(Path(env.get("MAC_MODELS_DIR", ROOT / "models")).expanduser().resolve())
    # Two slots, each 16k by default. These same values go to llama-server AND rag/ingest in Compose.
    env["LLM_CTX"] = env.get("MAC_LLM_CTX", "32768")
    env["LLM_MODEL"] = "qwen3.8-27b"
    env["LLM_GGUF"] = GGUF
    env["LLM_MMPROJ"] = MMPROJ
    # Keep Think and exhaustive prompts within a 16k request slot.
    env.setdefault("RAG_THINK_TOKENS", "4096")
    env.setdefault("RAG_SOURCE_BUDGET", "8000")
    env.setdefault("RAG_VERIFY_EVIDENCE_TOKENS", "8000")
    env.setdefault("RAG_EXHAUSTIVE_GROUP_TOKENS", "4000")
    env.setdefault("RAG_EXHAUSTIVE_NOTES_TOKENS", "8000")
    return env


def run(args, env=None):
    subprocess.run([str(a) for a in args], cwd=ROOT, env=env or environment(), check=True)


def compose_args(args):
    return ["docker", "compose", "-f", str(ROOT / "compose.yaml"),
            "-f", str(ROOT / "compose.macos.yaml"), *args]


def model_problems(folder):
    problems = []
    for name in (GGUF, MMPROJ):
        path = folder / "gguf" / name
        if not path.is_file() or path.stat().st_size < 1_000_000:
            problems.append(str(path))
    for name in HF_MODELS:
        path = folder / "hf" / name
        if not (path / "config.json").is_file() or not (path / "tokenizer.json").is_file():
            problems.append(str(path) + " (config/tokenizer missing)")
        index = path / "model.safetensors.index.json"
        if index.is_file():
            weights = set(json.loads(index.read_text())["weight_map"].values())
        else:
            weights = {"model.safetensors"}
        if any(not (path / w).is_file() for w in weights):
            problems.append(str(path) + " (weights missing)")
    return problems


def require_models(env):
    problems = model_problems(Path(env["MAC_MODELS_DIR"]))
    if problems:
        raise SystemExit("Models incomplete. Run: python3 scripts/macos.py setup\n" + "\n".join(problems))


def free_port(port):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            raise SystemExit(f"Port {port} is already in use; stop its owner before starting this service.") from None


def doctor(env):
    print(f"Machine: {platform.system()} / {platform.machine()}")
    print("Models:", env["MAC_MODELS_DIR"])
    print("Native llama-server:", "ready" if LLAMA_SERVER.is_file() else "not built")
    for tool in ("docker", "uv", "git", "clang"):
        print(f"{tool}: {shutil.which(tool) or 'MISSING'}")
    problems = model_problems(Path(env["MAC_MODELS_DIR"]))
    print("Model files:", "present (not an inference test)" if not problems else "incomplete")
    for problem in problems:
        print("  missing:", problem)
    hub = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    print("Qwen cache folders (a folder alone does not prove a complete model):")
    for cached in sorted(hub.glob("models--*Qwen*")):
        print(" ", cached.name.removeprefix("models--").replace("--", "/"))
    if shutil.which("docker"):
        run(["docker", "info", "--format", "Docker: {{.ServerVersion}}, VM memory: {{.MemTotal}} bytes"], env)
        run(compose_args(["config", "--quiet"]), env)


def setup(env):
    for tool in ("uv", "docker", "git", "clang"):
        if not shutil.which(tool):
            raise SystemExit(f"Install {tool} first; see docs/MACOS.md.")
    run(compose_args(["config", "--quiet"]), env)
    LLAMA_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not LLAMA_SOURCE.exists():
        run(["git", "init", LLAMA_SOURCE], env)
        run(["git", "-C", LLAMA_SOURCE, "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp.git"], env)
    # Fetch a fixed revision, not a moving branch. Re-running setup repairs an incomplete build.
    run(["git", "-C", LLAMA_SOURCE, "fetch", "--depth", "1", "origin", LLAMA_REVISION], env)
    run(["git", "-C", LLAMA_SOURCE, "checkout", "--detach", LLAMA_REVISION], env)
    cmake = ["uv", "tool", "run", "--from", "cmake==4.2.3", "cmake"]
    run([*cmake, "-S", LLAMA_SOURCE, "-B", LLAMA_SOURCE / "build", "-DGGML_METAL=ON",
         "-DGGML_NATIVE=ON", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF"], env)
    run([*cmake, "--build", LLAMA_SOURCE / "build", "--config", "Release", "--target", "llama-server", "-j", "4"], env)
    run(["uv", "sync", "--project", ROOT / "services/models", "--frozen", "--no-dev"], env)
    run(["uv", "run", "--with", "huggingface_hub==2.4.0", ROOT / "scripts/fetch_models.py",
         "--dest", env["MAC_MODELS_DIR"], "--reuse-cache"], env)
    require_models(env)
    secret_names = ("auth_signing.key", "rag_signing.key", "identity_public.json", "qdrant_api.key", "qdrant_local.yaml")
    present = [(ROOT / "secrets" / name).is_file() for name in secret_names]
    if any(present) and not all(present):
        raise SystemExit("Incomplete secrets/ folder; restore it from backup before proceeding (docs/DEPLOYMENT.md).")
    if not all(present):
        run(["uv", "run", "--with", "cryptography", ROOT / "scripts/init_secrets.py"], env)
    run(compose_args(["build"]), env)
    print("Setup complete. Follow docs/MACOS.md to start the two model terminals and the containers.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "setup", "llm", "models", "compose"))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    env = environment()
    if args.command == "compose":
        run(compose_args(args.args), env)
        return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This launcher requires an Apple Silicon Mac. Other hosts: docs/GETTING_STARTED.md.")
    if args.command == "doctor":
        doctor(env)
    elif args.command == "setup":
        setup(env)
    elif args.command == "llm":
        require_models(env)
        if not LLAMA_SERVER.is_file():
            raise SystemExit("Run python3 scripts/macos.py setup first.")
        free_port(18080)
        run([LLAMA_SERVER, "--model", Path(env["MAC_MODELS_DIR"]) / "gguf" / GGUF,
             "--mmproj", Path(env["MAC_MODELS_DIR"]) / "gguf" / MMPROJ,
             "--alias", env["LLM_MODEL"], "--n-gpu-layers", "999", "--ctx-size", env["LLM_CTX"],
             "--parallel", "2", "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
             "--image-max-tokens", "1024", "--jinja", "--host", "127.0.0.1", "--port", "18080"], env)
    else:
        require_models(env)
        free_port(18083)
        # Offline local paths: no automatic downloads or CUDA packages on macOS.
        for role, name in zip(("TEXT", "IMAGE", "RERANK"), HF_MODELS):
            env[f"MODELS_{role}_MODEL_PATH"] = str(Path(env["MAC_MODELS_DIR"]) / "hf" / name)
        env.update(MODELS_DEVICE="mps", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        env.setdefault("MODELS_TEXT_BATCH", "4")
        env.setdefault("MODELS_IMAGE_BATCH", "1")
        env.setdefault("MODELS_RERANK_BATCH", "4")
        env.setdefault("MODELS_MAX_WAITING", "8")
        env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
        # PyTorch's default low watermark (1.4) is invalid below our smaller high cap.
        # Keep adaptive garbage collection below the cap, including when the cap is customized.
        env.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO",
                       str(min(0.4, float(env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]) * 0.8)))
        run(["uv", "run", "--project", ROOT / "services/models", "--frozen", "--no-dev", "uvicorn",
             "models_svc.api:app", "--host", "127.0.0.1", "--port", "18083", "--no-access-log"], env)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        pass
