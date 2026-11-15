"""Mac launch contracts without loading models, contacting Docker, or changing host services."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import macos


def test_mac_environment_uses_one_context_and_expands_model_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("MAC_MODELS_DIR", str(tmp_path / "models with spaces"))
    monkeypatch.setenv("MAC_LLM_CTX", "49152")
    monkeypatch.setenv("LLM_CTX", "65536")
    env = macos.environment()
    assert env["LLM_CTX"] == "49152"
    assert env["MAC_MODELS_DIR"] == str(tmp_path / "models with spaces")


def test_missing_shard_is_reported_even_when_a_weight_file_exists(tmp_path):
    for name in macos.HF_MODELS:
        folder = tmp_path / "hf" / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}")
        (folder / "tokenizer.json").write_text("{}")
        (folder / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"a": "first.safetensors", "b": "second.safetensors"}}))
        (folder / "first.safetensors").write_bytes(b"present")
    problems = macos.model_problems(tmp_path)
    assert all(any(name in p and "weights missing" in p for p in problems) for name in macos.HF_MODELS)


@pytest.mark.parametrize("command,port", [("llm", "18080"), ("models", "18083")])
@pytest.mark.parametrize("high_watermark", ["0.5", "0.25"])
def test_native_servers_only_bind_loopback(monkeypatch, tmp_path, command, port, high_watermark):
    calls = []
    binary = tmp_path / "llama-server"
    binary.touch()
    monkeypatch.setattr(macos, "LLAMA_SERVER", binary)
    monkeypatch.setattr(macos.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(macos.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", high_watermark)
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    monkeypatch.setattr(macos, "require_models", lambda env: None)
    monkeypatch.setattr(macos, "free_port", lambda port: None)
    monkeypatch.setattr(macos, "run", lambda args, env: calls.append(([str(a) for a in args], env)))
    monkeypatch.setattr(sys, "argv", ["macos.py", command])
    macos.main()
    args, env = calls[0]
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--port") + 1] == port
    if command == "llm":
        assert args[args.index("--ctx-size") + 1] == env["LLM_CTX"]
        assert args[args.index("--parallel") + 1] == "2"
    else:
        assert env["MODELS_DEVICE"] == "mps"
        assert env["HF_HUB_OFFLINE"] == "1"
        assert 0 < float(env["PYTORCH_MPS_LOW_WATERMARK_RATIO"]) < float(high_watermark)


def test_compose_arguments_are_passed_without_a_shell(monkeypatch):
    calls = []
    monkeypatch.setattr(macos, "run", lambda args, env: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["macos.py", "compose", "logs", "--tail", "10", "rag"])
    macos.main()
    assert calls[0] == macos.compose_args(["logs", "--tail", "10", "rag"])
