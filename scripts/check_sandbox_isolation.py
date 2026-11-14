"""Live proof of the Python sandbox's containment (compose.yaml `sandbox`; DEPLOYMENT section 4.16).

    python scripts/check_sandbox_isolation.py [--compose-dir .] [--skip-memory] [--skip-port-scan]

Run it from the repository folder after every deploy and after any change to compose.yaml or the sandbox image.
It needs the stack up with the sandbox profile (COMPOSE_PROFILES=sandbox; rag and sandbox at least) and the docker
CLI; it uses no model and no user data, and it never touches a volume. It checks:

- the container, from `docker inspect`: read-only root, every capability dropped, no-new-privileges, memory, CPU
  and pid limits, no volume or bind mount, no published port, the sandbox network only, and that network internal
  and holding rag and the sandbox and nothing else; no secret-like variable in its environment;
- the network's fixed subnet, and rag told the same subnet (it refuses every request from it); the network in
  isolated gateway mode (no address on the host side of the bridge) and without IPv6;
- from inside a run, as model-written code would: no route to the internet and no DNS for outside names, no
  connection to any other service of the stack (auth, threads, audit, qdrant, llm, models, sql, ingest-api,
  gateway), no gateway address on the network and no route through one, and no open TCP port at all (every port,
  scanned in parallel) and no answer on common UDP ports on any host-side address (a route's gateway, or the
  subnet's first address when no container holds it: in isolated mode Docker may give it to a container); on Docker Desktop, host.docker.internal, gateway.docker.internal and
  kubernetes.docker.internal do not resolve, and the Desktop VM's 192.168.65.0/24 has no route and answers on no
  common TCP or UDP port; rag
  refusing every request from the network (403, health checks included), a read-only root and noexec scratch, no
  secrets in the environment or the filesystem, an unprivileged user with no capabilities, seccomp refusing the
  kernel keyring and io_uring, the RLIMITs in place, the SysV sysctls in place, a fork bomb stopped by the
  process limit, the server's open file limit and its OOM score below the run's (both at the run's start: a run can
  lower its own score later, DEPLOYMENT 4.16);
- across runs: a run past its time limit is killed with every process it started, one that left its group with
  setsid() included; files in the working folder, /tmp, /dev/shm and the scratch folder, and SysV shared memory,
  message queues and semaphores (also ones the run handed to uid 0 with IPC_SET) are gone before the next run (the
  message queue and semaphore set prove the server's removal: only shared memory is also removed by the kernel,
  kernel.shm_rmid_forced, once no process is attached); a run cannot read the server's memory or environment;
- memory: a run that allocates past the container's limit is killed and the sandbox keeps working
  (--skip-memory leaves this out);
- from sql, threads and ingest-api: the sandbox cannot be reached;
- on a Linux Docker host: no process outside the sandbox container runs as uid 10101 (RLIMIT_NPROC counts every
  process of a uid on the host). On Docker Desktop the host is a VM and this check is skipped.

Runs go through rag's network path: a small client started inside the rag container posts each probe to the
sandbox, exactly as rag.sandbox does. Prints a table; exit code 0 when every check passed, 1 otherwise, 2 when the
stack cannot be inspected. Tests of the pure parts: uv run --with pytest pytest -q scripts/tests
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass

SERVICE = "sandbox"
OTHERS = {"auth": 8007, "threads": 8006, "audit": 8004, "qdrant": 6333, "llm": 8080, "models": 8003, "sql": 8005,
          "ingest-api": 8001, "gateway": 8000}
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|^RAG_|^QDRANT|^AUTH_|^INGEST_|^EVAL_", re.IGNORECASE)
PUBLIC_NAMES = {"GPG_KEY"}  # the python base image's release-signing key fingerprint: public, not a credential


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False


# --- the container, from docker inspect ---------------------------------------------------------------------------


def static_checks(container: dict, network: dict | None, members: set[str], rag_env: list[str] | None = None
                  ) -> list[Check]:
    """What `docker inspect` of the sandbox container (and of its network, and of rag's environment) must show."""
    host, config = container.get("HostConfig") or {}, container.get("Config") or {}
    out = [
        Check("root filesystem read-only", bool(host.get("ReadonlyRootfs"))),
        Check("every capability dropped", "ALL" in [c.upper() for c in host.get("CapDrop") or []]
              and not host.get("CapAdd"), f"drop {host.get('CapDrop')}, add {host.get('CapAdd')}"),
        Check("no-new-privileges", any("no-new-privileges" in o for o in host.get("SecurityOpt") or []),
              str(host.get("SecurityOpt"))),
        Check("not privileged", not host.get("Privileged")),
        Check("memory limit set", (host.get("Memory") or 0) > 0, str(host.get("Memory"))),
        Check("no swap beyond the memory limit", host.get("MemorySwap") in (host.get("Memory"), None)
              and (host.get("Memory") or 0) > 0, f"memory {host.get('Memory')}, memory+swap {host.get('MemorySwap')}"),
        Check("CPU limit set", (host.get("NanoCpus") or 0) > 0 or (host.get("CpuQuota") or 0) > 0),
        Check("pid limit set", (host.get("PidsLimit") or 0) > 0, str(host.get("PidsLimit"))),
        Check("no volume or bind mount", not [m for m in container.get("Mounts") or [] if m.get("Type") != "tmpfs"],
              str([m.get("Destination") for m in container.get("Mounts") or []])),
        Check("tmpfs scratch and /tmp", {"/scratch", "/tmp"} <= set(host.get("Tmpfs") or {}), str(host.get("Tmpfs"))),
        Check("no published port", not any(host.get("PortBindings") or {}), str(host.get("PortBindings"))),
        Check("not running as root (image user)", str(config.get("User") or "") not in ("", "0", "root"),
              str(config.get("User"))),
        Check("no init process (the server is PID 1)", not host.get("Init"), str(host.get("Init"))),
    ]
    names = [k.split("=", 1)[0] for k in config.get("Env") or []]
    secret_like = [n for n in names if SECRET_NAME.search(n) and n not in PUBLIC_NAMES]
    out.append(Check("no secret-like variable in its environment", not secret_like, ", ".join(secret_like)))
    nets = list(((container.get("NetworkSettings") or {}).get("Networks") or {}))
    out.append(Check("on the sandbox network only", len(nets) == 1 and nets[0].endswith("_sandbox"), str(nets)))
    out.append(Check("the sandbox network is internal (no route out)", bool(network and network.get("Internal"))))
    out.append(Check("the sandbox network holds rag and the sandbox only", members == {"rag", SERVICE},
                     str(sorted(members))))
    subnets = [c.get("Subnet") for c in (((network or {}).get("IPAM") or {}).get("Config") or []) if c.get("Subnet")]
    told = next((e.split("=", 1)[1] for e in rag_env or [] if e.startswith("RAG_SANDBOX_SUBNET=")), "")
    out.append(Check("the sandbox network has a fixed subnet, and rag refuses requests from it",
                     len(subnets) == 1 and told == subnets[0], f"network {subnets}, rag RAG_SANDBOX_SUBNET={told!r}"))
    options = (network or {}).get("Options") or {}
    out.append(Check("the sandbox network is in isolated gateway mode (no host address on the bridge)",
                     options.get("com.docker.network.bridge.gateway_mode_ipv4") == "isolated", str(options)))
    out.append(Check("the sandbox network has no IPv6", not (network or {}).get("EnableIPv6")))
    gateways = [c["Gateway"] for c in (((network or {}).get("IPAM") or {}).get("Config") or []) if c.get("Gateway")]
    out.append(Check("the sandbox network has no gateway address (nothing on the host side of the bridge)",
                     not gateways, str(gateways)))
    return out


# --- probes: model-written code, run by the sandbox ---------------------------------------------------------------

PROBE_HEAD = r'''
import ctypes, json, os, socket, subprocess, sys, time, urllib.error, urllib.request
results = []
def check(name, ok, detail=""):
    results.append([name, bool(ok), str(detail)[:300]])
def connects(host, port, timeout=3):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False
'''

PROBE_CONTAINMENT = PROBE_HEAD + r'''
OTHERS = __OTHERS__
for name, port in [("1.1.1.1", 443), ("8.8.8.8", 53), ("9.9.9.9", 443)]:
    check(f"no route to the internet ({name}:{port})", not connects(name, port))
for name in ("example.com", "pypi.org", "github.com"):
    try:
        socket.getaddrinfo(name, 443)
        check(f"no DNS for outside names ({name})", False, f"{name} resolved")
    except OSError:
        check(f"no DNS for outside names ({name})", True)
for host, port in OTHERS.items():
    check(f"{host}:{port} unreachable", not connects(host, port))
routes = open("/proc/net/route").read().splitlines()[1:]
check("no default route", not any(r.split()[1] == "00000000" for r in routes), "; ".join(routes))
import asyncio


async def scan(host, ports, limit=800, timeout=1.0):
    gate, found = asyncio.Semaphore(limit), []

    async def one(port):
        async with gate:
            try:
                _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
                writer.close()
                found.append(port)
            except Exception:
                pass

    await asyncio.gather(*(one(p) for p in ports))
    return sorted(found)


def dns_query(name):
    labels = b"".join(bytes([len(p)]) + p.encode() for p in name.split(".")) + b"\0"
    return b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00" + labels + b"\x00\x01\x00\x01"


UDP = {53: dns_query("example.com"), 67: b"\x01" + b"\x00" * 239, 69: b"\x00\x01probe\x00octet\x00",
       123: b"\x1b" + b"\x00" * 47, 137: b"\x00" * 50, 161: b"\x30\x00", 500: b"\x00" * 28,
       1900: (b"M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: \"ssdp:discover\"\r\n"
              b"MX: 1\r\nST: ssdp:all\r\n\r\n"),
       5353: dns_query("_services._dns-sd._udp.local"), 11211: b"\x00\x00\x00\x00\x00\x01\x00\x00stats\r\n"}


def udp_answers(host, port, payload, timeout=1.0):
    """True when anything comes back (silence and ICMP port unreachable both count as closed)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        sock.send(payload)
        return bool(sock.recv(2048))
    except OSError:
        return False
    finally:
        sock.close()


def udp_open(host):
    return [p for p, payload in UDP.items() if udp_answers(host, p, payload)]


# The host's side of the bridge: the gateway of any route, and the subnet's first address unless a container holds
# it. In isolated gateway mode the bridge has no address on the host, so Docker may give the first address to a
# container (often the sandbox itself); scanning our own address would only find our own server and TCP
# self-connects, so the members of the network (from docker network inspect) are never treated as the host.
MEMBERS = __MEMBERS__
host_side = set()
via = [line.split()[0] + " via " + socket.inet_ntoa(int(line.split()[2], 16).to_bytes(4, "little"))
       for line in routes if int(line.split()[2], 16)]
check("no route through a gateway (no default route, nothing beyond the sandbox network)", not via, "; ".join(via))
for line in routes:
    dest, gw, mask = int(line.split()[1], 16), int(line.split()[2], 16), int(line.split()[7], 16)
    if gw:
        host_side.add(socket.inet_ntoa(gw.to_bytes(4, "little")))
    if mask:
        first = socket.inet_ntoa(((dest & mask) + (1 << 24)).to_bytes(4, "little"))
        if first not in MEMBERS:
            host_side.add(first)
if not host_side:
    check("no host address on the sandbox network to reach (isolated gateway mode)", True,
          f"members {sorted(MEMBERS)}")
for gateway in sorted(host_side):
    if __SCAN__:
        started = time.monotonic()
        open_ports = asyncio.run(scan(gateway, range(1, 65536)))
        check(f"the bridge's host address ({gateway}) has no open TCP port (all 65535 scanned)",
              not open_ports, f"open: {open_ports[:20]}; {time.monotonic() - started:.0f} s")
    check(f"the bridge's host address ({gateway}) answers on no common UDP port", not udp_open(gateway),
          f"probed {sorted(UDP)}")
for name in ("host.docker.internal", "gateway.docker.internal", "kubernetes.docker.internal"):
    try:
        addrs = sorted({a[4][0] for a in socket.getaddrinfo(name, None)})
        check(f"{name} does not resolve (Docker Desktop)", False, f"resolves to {addrs}")
    except OSError:
        check(f"{name} does not resolve (Docker Desktop)", True)
covered = [r for r in routes if r.split()[1] == "00000000" or
           (int(r.split()[1], 16) & int(r.split()[7], 16)) == (int.from_bytes(socket.inet_aton("192.168.65.0"), "little")
                                                               & int(r.split()[7], 16))]
check("no route to the Docker Desktop VM's 192.168.65.0/24", not covered, "; ".join(covered))
vm = [f"{h}:{p}" for h in ("192.168.65.1", "192.168.65.2", "192.168.65.3", "192.168.65.5", "192.168.65.7",
                           "192.168.65.254") for p in (22, 53, 80, 443, 2375, 2376, 3128, 8080, 11434)
      if connects(h, p, timeout=1)]
vm += [f"{h}:udp{p}" for h in ("192.168.65.1", "192.168.65.2", "192.168.65.7") for p in udp_open(h)]
check("the Docker Desktop VM's 192.168.65.0/24 answers on no common port", not vm, ", ".join(vm))


def status(method, url, body=None):
    req = urllib.request.Request(url, data=body, method=method, headers={"content-type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=5).status
    except urllib.error.HTTPError as e:
        return e.code
    except OSError as e:
        return repr(e)


for method, path, body in (("POST", "/v1/chat", b"{}"), ("GET", "/readyz", None), ("GET", "/healthz", None)):
    got = status(method, "http://rag:8002" + path, body)
    check(f"rag refuses {method} {path} from the sandbox network (403)", got == 403, got)
import errno, platform
NUMBERS = {"x86_64": {"add_key": 248, "request_key": 249, "keyctl": 250, "io_uring_setup": 425},
           "aarch64": {"add_key": 217, "request_key": 218, "keyctl": 219, "io_uring_setup": 425}}
calls = NUMBERS.get(platform.machine(), {})
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
for name, nr in calls.items():
    ctypes.set_errno(0)
    rc = libc.syscall(nr, 0, 0, 0, 0, 0)
    err = ctypes.get_errno()
    check(f"seccomp refuses {name}", rc == -1 and err in (errno.EPERM, errno.ENOSYS), f"rc {rc}, errno {err}")
if not calls:
    check("seccomp refuses the kernel keyring and io_uring", False, f"no syscall numbers for {platform.machine()}")
try:
    soft = next(l.split()[3] for l in open("/proc/1/limits") if l.startswith("Max open files"))
    server_adj = int(open("/proc/1/oom_score_adj").read())
    own_adj = int(open("/proc/self/oom_score_adj").read())
    check("the server can open enough files (RLIMIT_NOFILE of PID 1)", soft == "unlimited" or int(soft) >= 1024,
          f"soft limit {soft}")
    check("the OOM killer prefers the run to the server (scores at the run's start; a run can lower its own)",
          server_adj < own_adj, f"server {server_adj}, run {own_adj}")
    sysctls = {n: open(f"/proc/sys/kernel/{n}").read().split() for n in ("shm_rmid_forced", "shmmax", "shmall", "msgmnb")}
    check("the SysV sysctls are in place (shm_rmid_forced 1, small shmmax, shmall, msgmnb)",
          sysctls["shm_rmid_forced"] == ["1"] and int(sysctls["shmmax"][0]) <= 1 << 30
          and int(sysctls["shmall"][0]) <= 1 << 18 and int(sysctls["msgmnb"][0]) <= 1 << 20, sysctls)
except (OSError, StopIteration, ValueError) as e:
    check("the server's limits are readable", False, repr(e))
for path in ("/should-not-write", "/app/should-not-write", "/etc/should-not-write", "/usr/local/should-not-write"):
    try:
        open(path, "w").close()
        check(f"cannot write {os.path.dirname(path) or '/'}", False, path)
    except OSError:
        check(f"cannot write {os.path.dirname(path) or '/'}", True)
script = "/tmp/probe.sh"
open(script, "w").write("#!/bin/sh\necho ran\n")
os.chmod(script, 0o755)
try:
    subprocess.run([script], capture_output=True, timeout=5)
    check("/tmp is noexec", False)
except PermissionError:
    check("/tmp is noexec", True)
check("no /run/secrets", not os.path.exists("/run/secrets"))
names = sorted(os.environ)
bad = [n for n in names if any(w in n.upper() for w in ("KEY", "TOKEN", "SECRET", "PASSW", "CREDENTIAL"))
       or n.startswith(("RAG_", "QDRANT", "AUTH_", "INGEST_", "SANDBOX_"))]
check("no secret or server variable in the run's environment", not bad, ", ".join(names))
check("unprivileged user", os.getuid() != 0 and os.getgid() != 0, f"uid {os.getuid()} gid {os.getgid()}")
status = dict(line.split(":", 1) for line in open("/proc/self/status").read().splitlines() if ":" in line)
check("no effective capabilities", int(status["CapEff"].strip(), 16) == 0, status["CapEff"].strip())
check("no capabilities to gain", int(status["CapBnd"].strip(), 16) == 0, status["CapBnd"].strip())
check("no-new-privileges in force", status.get("NoNewPrivs", "").strip() == "1")
check("seccomp filter in force", status.get("Seccomp", "").strip() == "2", status.get("Seccomp", "").strip())
import resource
names = ("RLIMIT_NPROC", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_MSGQUEUE")
limits = {n: resource.getrlimit(getattr(resource, n)) for n in names}
check("RLIMITs in place", limits["RLIMIT_MSGQUEUE"] == (0, 0) and limits["RLIMIT_NPROC"][1] != resource.RLIM_INFINITY
      and limits["RLIMIT_FSIZE"][1] != resource.RLIM_INFINITY, limits)
BOMB = """
import os, time
n, pids = 0, []
try:
    while n < 5000:
        pid = os.fork()
        if pid == 0:
            time.sleep(30)
            os._exit(0)
        pids.append(pid)
        n += 1
except OSError:
    pass
print(n)
for p in pids:
    os.kill(p, 9)
"""
bomb = subprocess.run([sys.executable, "-c", BOMB], capture_output=True, text=True, timeout=60)
forked = int((bomb.stdout or "0").strip() or 0)
check("a fork bomb is stopped by the process limit", 0 < forked < 1000, f"{forked} processes before fork failed")
print(json.dumps(results))
'''

PROBE_LEAVE = PROBE_HEAD + r'''
open("left.txt", "w").write("secret of the previous run")
open("/tmp/left.txt", "w").write("secret")
open("/dev/shm/left.txt", "w").write("secret")
open(os.path.join(os.path.dirname(os.getcwd()), "..", "left.txt"), "w").write("secret")
import struct
libc = ctypes.CDLL(None, use_errno=True)


def give_away(ctl, ipc_id, *head):
    """IPC_STAT, then IPC_SET with the owner set to uid 0 (ipc_perm.uid sits after the 4-byte key)."""
    buf = ctypes.create_string_buffer(512)
    if ctl(ipc_id, *head, 2, buf) != 0:
        return f"stat failed {ctypes.get_errno()}"
    struct.pack_into("I", buf, 4, 0)
    return ctl(ipc_id, *head, 1, buf)


shm = libc.shmget(0x5A5A, 4096, 0o1000 | 0o600)  # IPC_CREAT | 0600
msg = libc.msgget(0x5A5B, 0o1000 | 0o600)
sem = libc.semget(0x5A5C, 1, 0o1000 | 0o600)
print(f"SHM={shm} given={give_away(libc.shmctl, shm)}", flush=True)
print(f"MSG={msg} given={give_away(libc.msgctl, msg)}", flush=True)
print(f"SEM={sem} given={give_away(libc.semctl, sem, 0)}", flush=True)
for detached in (True, False):
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], start_new_session=detached)
print("CWD=" + os.getcwd(), flush=True)
time.sleep(600)
'''

PROBE_AFTER = PROBE_HEAD + r'''
mine = {os.getpid(), os.getppid(), 1}
others = []
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        state = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        continue
    if int(pid) not in mine and state != "Z":
        others.append(f"{pid}: {cmd[:80]}")
check("no process of the previous run is left (setsid included)", not others, "; ".join(others))
check("the previous run's folder is gone", not os.path.exists(__CWD__), __CWD__)
check("/tmp is empty", os.listdir("/tmp") == [], os.listdir("/tmp"))
check("/dev/shm is empty", os.listdir("/dev/shm") == [], os.listdir("/dev/shm"))
scratch = os.path.dirname(os.path.dirname(os.getcwd()))
check("the scratch folder holds only this run", os.listdir(scratch) == [os.path.basename(os.path.dirname(os.getcwd()))],
      os.listdir(scratch))
check("the working folder starts empty", sorted(os.listdir(".")) == [], os.listdir("."))
rows = open("/proc/sysvipc/shm").read().splitlines()[1:]
check("no SysV shared memory is left (the kernel also removes it: kernel.shm_rmid_forced)", not rows, rows[:3])
check("the server is PID 1", os.getppid() == 1, os.getppid())
for kind in ("msg", "sem"):  # no kernel setting removes these: an empty table proves the server's removal
    rows = open(f"/proc/sysvipc/{kind}").read().splitlines()[1:]
    check(f"no SysV {kind} object is left, also one handed to uid 0 (proves the server's removal)", not rows, rows[:3])
for what in ("environ", "mem", "maps"):
    try:
        open(f"/proc/{os.getppid()}/{what}", "rb").read(16)
        check(f"the server's /proc/<pid>/{what} is unreadable", False)
    except OSError:
        check(f"the server's /proc/<pid>/{what} is unreadable", True)
print(json.dumps(results))
'''

PROBE_MEMORY = r'''
chunks = []
while True:
    chunks.append(bytearray(256 * 1024 * 1024))
'''

# Started inside the rag container: posts one frame to the sandbox as rag.sandbox does, prints the reply.
CLIENT = r'''
import json, sys, urllib.error, urllib.request
from rag_common.sandbox import MEDIA_TYPE, encode
job = json.loads(sys.stdin.readline())
body = b"".join(encode(job["code"], job["timeout_s"], [(p, d.encode()) for p, d in job.get("files", {}).items()]))
req = urllib.request.Request("http://sandbox:8008/v1/execute", data=body, headers={"content-type": MEDIA_TYPE})
try:
    print(urllib.request.urlopen(req, timeout=job["timeout_s"] + 600).read().decode())
except urllib.error.HTTPError as e:
    print(json.dumps({"http_status": e.code, "detail": e.read().decode()[:500]}))
'''


def probe_results(reply: dict) -> list[Check]:
    """The checks a probe printed as its last line of output; a failed probe is one failed check."""
    lines = (reply.get("stdout") or "").strip().splitlines()
    try:
        return [Check(name, ok, detail) for name, ok, detail in json.loads(lines[-1])]
    except (IndexError, ValueError, TypeError):
        detail = reply.get("detail") or reply.get("error") or (reply.get("stderr") or "")[-300:] or str(reply)[:300]
        return [Check("probe ran", False, detail)]


def services(network: dict | None, project: str) -> set[str]:
    """The compose services with a container on `network` (containers are named <project>-<service>-<n>)."""
    out = set()
    for c in ((network or {}).get("Containers") or {}).values():
        name = c.get("Name") or ""
        out.add(name[len(project) + 1:].rsplit("-", 1)[0] if name.startswith(project + "-") else name)
    return out


def leftover_cwd(reply: dict) -> str | None:
    match = re.search(r"^CWD=(.+)$", reply.get("stdout") or "", re.M)
    return match.group(1).strip() if match else None


# --- running ------------------------------------------------------------------------------------------------------


class Stack:
    def __init__(self, compose_dir: str):
        self.compose = ["docker", "compose", "--project-directory", compose_dir]

    def run(self, args: list[str], stdin: str | None = None, timeout: float = 900) -> subprocess.CompletedProcess:
        return subprocess.run(args, input=stdin, capture_output=True, text=True, timeout=timeout, check=False)

    def container_id(self, service: str) -> str:
        out = self.run([*self.compose, "ps", "-q", service]).stdout.strip()
        if not out:
            raise SystemExit(f"{service} is not running; start the stack with the sandbox profile first "
                             "(COMPOSE_PROFILES=sandbox docker compose up -d)")
        return out.splitlines()[0]

    def inspect(self, kind: str, name: str) -> dict:
        proc = self.run(["docker", kind, "inspect", name] if kind == "network" else ["docker", "inspect", name])
        if proc.returncode:
            raise SystemExit(f"docker inspect {name} failed: {proc.stderr.strip()}")
        return json.loads(proc.stdout)[0]

    def execute(self, code: str, timeout_s: float = 60, files: dict[str, str] | None = None) -> dict:
        job = json.dumps({"code": code, "timeout_s": timeout_s, "files": files or {}})
        proc = self.run([*self.compose, "exec", "-T", "rag", "python", "-c", CLIENT], stdin=job + "\n",
                        timeout=timeout_s + 660)
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            return {"detail": (proc.stderr or proc.stdout).strip()[-500:]}

    def host_uid_check(self, container_id: str, uid: int = 10101) -> Check:
        """No process outside the sandbox container runs as `uid` on this host (Linux hosts only)."""
        name = f"no process outside the sandbox runs as uid {uid} on the host"
        if not sys.platform.startswith("linux"):
            return Check(name, True, "skipped: not a Linux Docker host (Docker Desktop runs containers in a VM)",
                         skipped=True)
        ps = self.run(["ps", "-e", "-o", "uid=,pid="], timeout=60)
        top = self.run(["docker", "top", container_id, "-eo", "pid"], timeout=60)
        if ps.returncode or top.returncode:
            return Check(name, False, (ps.stderr or top.stderr).strip()[:200])
        mine = {int(x) for x in top.stdout.split()[1:] if x.isdigit()}
        others = [int(p) for u, p in (line.split() for line in ps.stdout.splitlines() if line.strip())
                  if int(u) == uid and int(p) not in mine]
        return Check(name, not others, f"pids {others[:10]}")

    def reaches_sandbox(self, service: str) -> bool | None:
        code = ("import socket\ntry:\n    socket.create_connection(('sandbox', 8008), timeout=3).close()\n"
                "    print('yes')\nexcept OSError:\n    print('no')")
        proc = self.run([*self.compose, "exec", "-T", service, "python", "-c", code], timeout=60)
        out = proc.stdout.strip()
        return None if out not in ("yes", "no") else out == "yes"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--compose-dir", default=".", help="the folder with compose.yaml (default: .)")
    parser.add_argument("--skip-memory", action="store_true", help="leave out the memory limit check")
    parser.add_argument("--skip-port-scan", action="store_true",
                        help="leave out the scan of every TCP port on the host's address (about a minute)")
    args = parser.parse_args(argv)
    stack = Stack(args.compose_dir)
    checks: list[Check] = []
    try:
        container = stack.inspect("container", stack.container_id(SERVICE))
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"cannot inspect the stack: {exc}", file=sys.stderr)
        return 2
    nets = list(((container.get("NetworkSettings") or {}).get("Networks") or {}))
    network = stack.inspect("network", nets[0]) if nets else None
    project = ((container.get("Config") or {}).get("Labels") or {}).get("com.docker.compose.project", "local-rag")
    rag = stack.inspect("container", stack.container_id("rag"))
    checks += static_checks(container, network, services(network, project), (rag.get("Config") or {}).get("Env"))
    members = sorted(c["IPv4Address"].split("/")[0] for c in ((network or {}).get("Containers") or {}).values()
                     if c.get("IPv4Address"))
    probe = (PROBE_CONTAINMENT.replace("__OTHERS__", repr(OTHERS)).replace("__SCAN__", repr(not args.skip_port_scan))
             .replace("__MEMBERS__", repr(members)))
    checks += probe_results(stack.execute(probe, timeout_s=590))  # within SANDBOX_MAX_TIMEOUT_S (600)
    left = stack.execute(PROBE_LEAVE, timeout_s=5)
    checks.append(Check("a run past its time limit is stopped", bool(left.get("timed_out")), str(left.get("error"))))
    cwd = leftover_cwd(left)
    checks.append(Check("the previous run printed its folder", cwd is not None, (left.get("stdout") or "")[:200]))
    out = left.get("stdout") or ""
    made = all(re.search(rf"^{k}=\d+ given=0$", out, re.M) for k in ("SHM", "MSG", "SEM"))
    checks.append(Check("the previous run made SysV objects and handed them to uid 0 (removed before the next)",
                        made, out[:300]))
    checks += probe_results(stack.execute(PROBE_AFTER.replace("__CWD__", repr(cwd or "/nonexistent")), timeout_s=60))
    if not args.skip_memory:
        memory = stack.execute(PROBE_MEMORY, timeout_s=120)
        checks.append(Check("a run past the memory limit is killed", memory.get("signal") == "SIGKILL"
                            or "MemoryError" in (memory.get("stderr") or ""),
                            str(memory.get("error") or memory.get("detail") or memory.get("stderr", "")[-200:])))
        after = stack.execute("print('still here')", timeout_s=30)
        checks.append(Check("the sandbox works after that", (after.get("stdout") or "").strip() == "still here",
                            str(after)[:200]))
    checks.append(stack.host_uid_check(container["Id"]))
    for service in ("sql", "threads", "ingest-api"):
        reached = stack.reaches_sandbox(service)
        checks.append(Check(f"{service} cannot reach the sandbox", reached is False,
                            "could not run the check" if reached is None else ""))
    width = max(len(c.name) for c in checks)
    for c in checks:
        label = "skip" if c.skipped else "ok  " if c.ok else "FAIL"
        print(f"{label}  {c.name.ljust(width)}  {c.detail if not c.ok or c.skipped else ''}".rstrip())
    failed = [c for c in checks if not c.ok]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed" + (": ALL PASSED" if not failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
