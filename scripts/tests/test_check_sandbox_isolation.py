"""The pure parts of scripts/check_sandbox_isolation.py; the live run needs the deployed stack.

    uv run --with httpx --with pytest pytest -q scripts/tests
"""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_sandbox_isolation as iso  # noqa: E402

GOOD = {
    "HostConfig": {"ReadonlyRootfs": True, "CapDrop": ["ALL"], "CapAdd": None, "SecurityOpt": ["no-new-privileges:true"],
                   "Privileged": False, "Memory": 4 << 30, "MemorySwap": 4 << 30, "NanoCpus": 4_000_000_000,
                   "PidsLimit": 256, "Tmpfs": {"/scratch": "size=2g", "/tmp": "size=512m"}, "PortBindings": {}},
    "Config": {"User": "sandbox", "Env": ["PATH=/app/.venv/bin:/usr/bin", "GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305",
                                          "SANDBOX_MAX_FILES=50", "HOME=/nonexistent"],
               "Labels": {"com.docker.compose.project": "local-rag"}},
    "Mounts": [],
    "NetworkSettings": {"Networks": {"local-rag_sandbox": {}}},
}
NETWORK = {"Internal": True, "Containers": {"a": {"Name": "local-rag-rag-1"}, "b": {"Name": "local-rag-sandbox-1"}},
           "IPAM": {"Config": [{"Subnet": "172.31.241.0/24"}]}, "EnableIPv6": False,
           "Options": {"com.docker.network.bridge.gateway_mode_ipv4": "isolated"}}
RAG_ENV = ["PATH=/app/.venv/bin", "RAG_SANDBOX_SUBNET=172.31.241.0/24"]


def failing(checks) -> list[str]:
    return [c.name for c in checks if not c.ok]


def test_a_contained_sandbox_passes_every_static_check():
    assert failing(iso.static_checks(GOOD, NETWORK, iso.services(NETWORK, "local-rag"), RAG_ENV)) == []


@pytest.mark.parametrize("change, name", [
    (lambda c: c["HostConfig"].update(ReadonlyRootfs=False), "root filesystem read-only"),
    (lambda c: c["HostConfig"].update(CapAdd=["NET_RAW"]), "every capability dropped"),
    (lambda c: c["HostConfig"].update(SecurityOpt=[]), "no-new-privileges"),
    (lambda c: c["HostConfig"].update(PidsLimit=0), "pid limit set"),
    (lambda c: c["HostConfig"].update(MemorySwap=-1), "no swap beyond the memory limit"),
    (lambda c: c["Mounts"].append({"Type": "volume", "Destination": "/data"}), "no volume or bind mount"),
    (lambda c: c["HostConfig"].update(PortBindings={"8008/tcp": [{"HostPort": "8008"}]}), "no published port"),
    (lambda c: c["Config"].update(User=""), "not running as root (image user)"),
    (lambda c: c["Config"]["Env"].append("RAG_SIGNING_KEY_FILE=/run/secrets/x"), "no secret-like variable in its environment"),
    (lambda c: c["NetworkSettings"]["Networks"].update({"local-rag_backend": {}}), "on the sandbox network only"),
    (lambda c: c["HostConfig"].update(Init=True), "no init process (the server is PID 1)"),
])
def test_each_static_rule_fails_on_its_own(change, name):
    container = copy.deepcopy(GOOD)
    change(container)
    assert failing(iso.static_checks(container, NETWORK, {"rag", "sandbox"}, RAG_ENV)) == [name]


def test_the_network_must_be_internal_and_hold_rag_and_the_sandbox_only():
    crowded = copy.deepcopy(NETWORK)
    crowded["Containers"]["c"] = {"Name": "local-rag-ingest-api-1"}
    assert iso.services(crowded, "local-rag") == {"rag", "sandbox", "ingest-api"}
    assert failing(iso.static_checks(GOOD, {**crowded, "Internal": False}, iso.services(crowded, "local-rag"),
                                     RAG_ENV)) == ["the sandbox network is internal (no route out)",
                                                   "the sandbox network holds rag and the sandbox only"]


@pytest.mark.parametrize("network, env", [({**NETWORK, "IPAM": {"Config": []}}, RAG_ENV),
                                          (NETWORK, ["RAG_SANDBOX_SUBNET="]),
                                          (NETWORK, ["RAG_SANDBOX_SUBNET=10.0.0.0/24"])])
def test_rag_must_refuse_the_networks_own_subnet(network, env):
    assert failing(iso.static_checks(GOOD, network, {"rag", "sandbox"}, env)) == [
        "the sandbox network has a fixed subnet, and rag refuses requests from it"]


def test_probe_results_are_read_from_the_last_line_and_a_broken_probe_is_a_failure():
    reply = {"stdout": "noise\n" + json.dumps([["no DNS for outside names", True, ""], ["x", False, "why"]])}
    assert [(c.name, c.ok, c.detail) for c in iso.probe_results(reply)] == [("no DNS for outside names", True, ""),
                                                                            ("x", False, "why")]
    broken = iso.probe_results({"stdout": "", "stderr": "Traceback ...\nNameError: x"})
    assert [(c.name, c.ok) for c in broken] == [("probe ran", False)] and "NameError" in broken[0].detail
    assert iso.probe_results({"http_status": 503, "detail": "busy"})[0].detail == "busy"


def test_the_previous_runs_folder_is_read_from_its_output():
    assert iso.leftover_cwd({"stdout": "CWD=/scratch/run-ab/work\n"}) == "/scratch/run-ab/work"
    assert iso.leftover_cwd({"stdout": "nothing"}) is None


@pytest.mark.parametrize("probe", [iso.PROBE_CONTAINMENT.replace("__OTHERS__", repr(iso.OTHERS)).replace("__SCAN__", "True"),
                                   iso.PROBE_LEAVE,
                                   iso.PROBE_AFTER.replace("__CWD__", repr("/x")), iso.PROBE_MEMORY, iso.CLIENT])
def test_every_probe_is_valid_python(probe):
    compile(probe, "probe", "exec")


def test_the_probes_name_every_other_service_of_the_stack():
    compose = (Path(__file__).resolve().parents[2] / "compose.yaml").read_text()
    for service in iso.OTHERS:
        assert f"\n  {service}:\n" in compose


def test_the_host_uid_check_is_skipped_off_linux_and_finds_strangers_on_linux(monkeypatch):
    stack = iso.Stack(".")
    monkeypatch.setattr(iso.sys, "platform", "win32")
    assert stack.host_uid_check("cid").skipped is True
    monkeypatch.setattr(iso.sys, "platform", "linux")

    class Done:
        def __init__(self, out):
            self.returncode, self.stdout, self.stderr = 0, out, ""

    outputs = {"ps": Done("    0     1\n10101   500\n10101   900\n 1000   901\n"), "docker": Done("PID\n500\n")}
    monkeypatch.setattr(stack, "run", lambda args, **kw: outputs[args[0]])
    check = stack.host_uid_check("cid")
    assert check.ok is False and "900" in check.detail
    outputs["ps"] = Done("10101   500\n")
    assert stack.host_uid_check("cid").ok is True


@pytest.mark.parametrize("change, name", [
    ({"Options": {}}, "the sandbox network is in isolated gateway mode (no host address on the bridge)"),
    ({"EnableIPv6": True}, "the sandbox network has no IPv6"),
    ({"IPAM": {"Config": [{"Subnet": "172.31.241.0/24", "Gateway": "172.31.241.1"}]}},
     "the sandbox network has no gateway address (nothing on the host side of the bridge)"),
])
def test_the_network_must_be_isolated_and_ipv4_only(change, name):
    assert failing(iso.static_checks(GOOD, {**NETWORK, **change}, {"rag", "sandbox"}, RAG_ENV)) == [name]


def test_the_compose_file_keeps_python_off_and_the_network_isolated():
    compose = (Path(__file__).resolve().parents[2] / "compose.yaml").read_text()
    network = compose.split("\n  sandbox:\n    internal: true\n", 1)[1].split("\n  # ", 1)[0]
    assert "enable_ipv6: false" in network and "com.docker.network.bridge.gateway_mode_ipv4: isolated" in network
    assert "RAG_DEEP_PYTHON: ${RAG_DEEP_PYTHON:-off}" in compose and "RAG_SANDBOX_URL: ${RAG_SANDBOX_URL:-}" in compose
    assert "  sandbox:\n    profiles: [sandbox]\n" in compose
