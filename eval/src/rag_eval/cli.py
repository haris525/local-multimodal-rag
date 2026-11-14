"""rag-eval: generate a golden set, run retrieval ablations, judge answer quality; and benchmark runs scored
without a judge (bench, compare, bench-setup, bench-clean, fetch-public)."""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from rag_eval import bench, environment, public, variants
from rag_eval.client import Gateway, Stack
from rag_eval.generate import generate
from rag_eval.paths import NAMED_SETS, REPO_ROOT, eval_dir, public_dir, reports_dir
from rag_eval.runs import CONFIGS, answers, check_labels, load, retrieval

REPORTS = reports_dir()
DEFAULT_REQUEST_TIMEOUT = 900.0  # seconds, per request in flight: Gateway's timeout before --concurrency
DOCMAP = eval_dir("datasets") / "bench-docmap.json"  # ignored by Git: doc ids are per account


def _fmt(ci: tuple[float, float, float]) -> str:
    mean, lo, hi = ci
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def _write(kind: str, payload: dict, table: list[str]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = REPORTS / f"{stamp}-{kind}"
    base.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    base.with_suffix(".md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return base.with_suffix(".md")


def cmd_generate(args, stack: Stack) -> None:
    n = generate(stack, Path(args.out), args.n, args.unanswerable, args.docs, args.seed)
    print(f"wrote {n} questions to {args.out}; review them before relying on the scores")


def cmd_retrieval(args, stack: Stack) -> None:
    items = load(Path(args.dataset))
    stale = check_labels(stack, items)
    if stale and not args.allow_stale:
        raise SystemExit("labels point at chunks whose text changed (re-ingested or re-chunked):\n  "
                         + "\n  ".join(stale[:20]) + "\nregenerate the set, or pass --allow-stale and read location-level metrics only")
    result = retrieval(stack, items, args.configs)
    metrics = list(next(iter(result["summary"].values())).keys())
    table = [f"# Retrieval evaluation ({result['queries']} answerable queries, mean [95% CI])", "",
             "| configuration | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    table += [f"| {name} | " + " | ".join(_fmt(s[m]) for m in metrics) + " |" for name, s in result["summary"].items()]
    if result["delta_vs_production"]:
        table += ["", "## Difference from production (paired bootstrap)", "",
                  "| configuration | strict MRR@10 | location MRR@10 |", "|---|---|---|"]
        table += [f"| {n} | {_fmt(d['strict MRR@10'])} | {_fmt(d['location MRR@10'])} |"
                  for n, d in result["delta_vs_production"].items()]
    result["stale_labels"] = stale
    print("\n".join(table))
    print("report:", _write("retrieval", result, table))


def cmd_answers(args, stack: Stack) -> None:
    items = load(Path(args.dataset))[: args.limit] if args.limit else load(Path(args.dataset))
    result = answers(stack, items, args.modes)
    table = [f"# Answer quality ({len(items)} questions, mean [95% CI]; judge = same local model)", ""]
    for mode, summary in result["summary"].items():
        table += [f"## mode: {mode}", "", "| metric | value |", "|---|---|"]
        table += [f"| {k} | {_fmt(v)} |" for k, v in summary.items()] + [""]
    print("\n".join(table))
    print("report:", _write("answers", result, table))


# --- benchmark ------------------------------------------------------------------------------------------


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _sets(args) -> list[str]:
    """The sets to run: --set as a comma list; under --variants the default includes the held-out public sets."""
    spec = args.set or ("local,hotpotqa,musique" if getattr(args, "variants", None) else "local")
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if len(names) != len(set(names)):
        raise SystemExit(f"--set lists a set twice: {spec}")
    if args.data and (len(names) != 1 or names[0] not in public.SOURCES):
        raise SystemExit("--data needs --set with exactly one public set")
    return names


def _public_items(name: str, args) -> tuple[list[bench.Item], dict[str, bytes]]:
    path = Path(args.data) if args.data else public_dir() / public.SOURCES[name]["file"]
    if not path.exists():
        raise SystemExit(f"{path} not found: run `python -m rag_eval fetch-public {name}` first "
                         "(or leave the public sets out with --set local)")
    cases = public.sample(public.load_cases(name, path), args.limit or 30, args.seed)
    return public.build(name, cases, args.corpus)


def _local_items(name: str) -> list[bench.Item]:
    path = NAMED_SETS.get(name) or Path(name)
    if not path.exists():
        raise SystemExit(f"benchmark set {path} not found (pass --set PATH)")
    return bench.load_items(path)


def resolve(gateway: Gateway, names: set[str]) -> dict[str, str]:
    """{file name: doc_id} of the benchmark account's ready documents; SystemExit naming any missing."""
    ready: dict[str, dict] = {}
    for d in gateway.documents():
        if d["status"] == "ready" and d["name"] in names:
            if d["name"] not in ready or d.get("updated_at", "") > ready[d["name"]].get("updated_at", ""):
                ready[d["name"]] = d
    missing = sorted(names - set(ready))
    if missing:
        raise SystemExit("these files are not ready documents of the token's user:\n  " + "\n  ".join(missing)
                         + "\nrun `python -m rag_eval bench-setup` first (with the same token)")
    return {name: d["doc_id"] for name, d in ready.items()}


def _load(args, gateway: Gateway) -> tuple[list[bench.Item], dict[str, str], dict]:
    """Items, {file name: doc_id} and set metadata for every set in --set; public documents are ensured first.
    With several sets, local items get the tag of their set ("hard" for the hard set, "local" for the local set
    or a path), so the per-tag tables and verdicts can tell them apart."""
    names = _sets(args)
    items: list[bench.Item] = []
    doc_ids: dict[str, str] = {}
    info: dict = {"set": ",".join(names)}
    for name in names:
        if name in public.SOURCES:
            got, files = _public_items(name, args)
            got = bench.select(got, args.tags, args.skip_tags, args.ids)
            needed = sorted({n for it in got for n in it.docs})
            doc_ids |= gateway.ensure({n: (lambda data=files[n]: data) for n in needed}, args.ingest_budget,
                                      progress=_say)
            info |= {"corpus": args.corpus, "seed": args.seed, "limit": args.limit or 30}
        else:
            got = bench.select(_local_items(name), args.tags, args.skip_tags, args.ids, args.limit)
            doc_ids |= resolve(gateway, {n for it in got for n in it.docs})
            if len(names) > 1:
                for it in got:
                    tag = name if name in NAMED_SETS else "local"
                    it.tags = [*it.tags, tag] if tag not in it.tags else it.tags
        items += got
    if not items:
        raise SystemExit("no questions left after the filters")
    return items, doc_ids, info


def cmd_bench(args) -> None:
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    timeout = args.request_timeout if args.request_timeout is not None else DEFAULT_REQUEST_TIMEOUT * args.concurrency
    if timeout <= 0:
        raise SystemExit("--request-timeout must be more than 0 seconds")
    if args.variants:
        if args.think:
            raise SystemExit("--variants keeps answer thinking off for every variant, so only deep search differs; "
                             "drop --think")
        if args.agent or args.deep_think or args.variant:
            raise SystemExit("--variants sets each variant's agent, deep_think, fanout, research_calls and python "
                             "(AGENT with +think, +fanout, +calls<N>, +python); drop --agent, --deep-think and "
                             "--variant")
        try:
            chosen = variants.parse_variants(args.variants)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    gateway = Gateway(url=args.url, timeout=timeout)
    items, doc_ids, info = _load(args, gateway)
    started = bench.now()
    meta = {**info, "mode": args.mode, "think": args.think, "deep_think": args.deep_think, "repeats": args.repeats,
            "agent": args.agent, "variant": args.variant or args.agent or "default", "url": str(gateway.api.base_url),
            "set_digest": bench.set_digest(items), "started": started.isoformat(timespec="seconds"),
            "stamp": started.strftime("%Y%m%d-%H%M%S"), "documents": doc_ids, "questions": len(items),
            "environment": environment.detect(args.compose_project, args.app_commit)}
    if args.concurrency > 1:  # absent at 1, so a sequential run's reports stay as they were
        meta["concurrency"] = args.concurrency
    if args.concurrency > 1 or args.request_timeout is not None:
        meta["request_timeout_s"] = timeout
    reports = Path(args.reports) if args.reports else reports_dir()
    reports.mkdir(parents=True, exist_ok=True)
    label = "combined" if args.variants else bench.safe_label(meta["variant"])
    stream = reports / f"bench-{label}-{meta['stamp']}.runs.jsonl"
    _say(f"rag image {meta['environment'].get('rag_image') or 'unknown'}, model {meta['environment'].get('model') or 'unknown'}, "
         f"LLM_CTX {meta['environment'].get('llm_ctx') or 'unknown'}; run records stream to {stream}")
    if args.concurrency > 1:
        _say(f"{args.concurrency} requests in flight at a time, sent in the run order; records arrive as they finish "
             f"(their `order` field is the run order); each request may wait {timeout:.0f} s for the next bytes of "
             "its reply")
        slots = (meta["environment"].get("rag_settings") or {}).get("RAG_LLM_PARALLEL")
        if slots and slots.isdigit() and args.concurrency > int(slots):
            _say(f"note: rag sets RAG_LLM_PARALLEL={slots}; requests beyond the llama slots wait in llama-server's "
                 "queue, which adds latency and can reach rag's LLM timeouts")

    def sink(rec: dict) -> None:
        with stream.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    if args.variants:
        _bench_variants(args, gateway, items, doc_ids, meta, chosen, reports, sink)
        return
    _say(f"{len(items)} questions x {args.repeats} repeats, mode {args.mode}, agent {args.agent or '(not sent)'}, "
         f"deep_think {'on' if args.deep_think else 'off'}")
    runs: list[dict] = []
    try:
        bench.run(gateway, items, doc_ids, args.mode, args.think, args.repeats, args.agent,
                  sink=lambda rec: (runs.append(rec), sink(rec)), progress=_say, deep_think=args.deep_think,
                  concurrency=args.concurrency)
    except KeyboardInterrupt:
        meta["incomplete"] = True
        _say("interrupted: writing the report for the runs so far" + _abandoned(args.concurrency))
    meta["finished"] = bench.now().isoformat(timespec="seconds")
    if not runs:
        raise SystemExit("no run finished")
    runs.sort(key=lambda rec: rec.get("order", 0))  # run order; a no-op for a sequential run
    asked = [it for it in items if any(r["id"] == it.id for r in runs)]
    js, md = bench.write_report(bench.build_report(meta, asked, runs), asked, reports)
    print(md.read_text(encoding="utf-8"))
    print(f"report: {js}\n        {md}")
    if meta.get("incomplete"):
        raise SystemExit(130)


def _abandoned(concurrency: int) -> str:
    return "; requests still in flight are abandoned and left out" if concurrency > 1 else ""


def _bench_variants(args, gateway: Gateway, items: list[bench.Item], doc_ids: dict[str, str], meta: dict,
                    chosen: list, reports: Path, sink) -> None:
    labels = [v.label for v in chosen]
    total = len(items) * args.repeats * len(chosen)
    _say(f"{len(items)} questions x {args.repeats} repeats x {len(chosen)} variants ({', '.join(labels)}) = {total} runs, "
         f"mode {args.mode}, interleaved in a random order (seed {args.order_seed}), answer thinking off")
    collected: dict[str, list[dict]] = {label: [] for label in labels}
    meta = meta | {"variants": labels, "schedule_seed": args.order_seed, "variant": "combined", "agent": None,
                   "deep_think": False, "think": False,
                   "schedule": "each repeat shuffles the questions; each question runs every variant in a shuffled order"}
    try:
        variants.run(gateway, items, doc_ids, args.mode, args.repeats, chosen, args.order_seed,
                     sink=lambda rec: (collected[rec["variant"]].append(rec), sink(rec)), progress=_say,
                     concurrency=args.concurrency)
    except KeyboardInterrupt:
        meta["incomplete"] = True
        _say("interrupted: writing the reports for the runs so far" + _abandoned(args.concurrency))
    meta["finished"] = bench.now().isoformat(timespec="seconds")
    for runs in collected.values():
        runs.sort(key=lambda rec: rec["order"])  # run order; a no-op for a sequential run
    meta["runs"] = sum(len(rs) for rs in collected.values())
    if not meta["runs"]:
        raise SystemExit("no run finished")
    paths: dict[str, str] = {}
    for v in chosen:
        runs = collected[v.label]
        if not runs:
            continue
        asked = [it for it in items if any(r["id"] == it.id for r in runs)]
        vmeta = meta | {"variant": v.label, "agent": v.agent, "deep_think": v.deep_think, "fanout": v.fanout,
                        "research_calls": v.research_calls, "python": v.python, "think": False,
                        "combined_report": f"bench-combined-{meta['stamp']}.json"}
        vmeta.pop("runs")
        js, _ = bench.write_report(bench.build_report(vmeta, asked, runs), asked, reports)
        paths[v.label] = str(js)
    report = variants.combined(meta, items, {k: v for k, v in collected.items() if v}, chosen, paths)
    js, md = variants.write_combined(report, reports)
    print(md.read_text(encoding="utf-8"))
    print(f"combined report: {js}\n                 {md}")
    for label, path in paths.items():
        print(f"{label}: {path}")
    if meta.get("incomplete"):
        raise SystemExit(130)


def cmd_compare(args) -> None:
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.a, args.b))
    for path, rep in ((args.a, a), (args.b, b)):
        if rep.get("kind") != "bench":
            raise SystemExit(f"{path} is not a bench report")
    result = bench.compare(a, b, args.metric)
    text = bench.compare_markdown(result)
    print(text)
    reports = Path(args.reports) if args.reports else reports_dir()
    reports.mkdir(parents=True, exist_ok=True)
    base = reports / (f"compare-{bench.safe_label(result['a'])}-vs-{bench.safe_label(result['b'])}-"
                      f"{datetime.now(UTC):%Y%m%d-%H%M%S}")
    base.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    base.with_suffix(".md").write_text(text, encoding="utf-8")
    print(f"report: {base.with_suffix('.md')}")


def _find_files(names: set[str], folders: list[Path]) -> tuple[dict[str, Path], list[str]]:
    found: dict[str, Path] = {}
    for folder in folders:
        if folder.is_dir():
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.name in names and path.name not in found:
                    found[path.name] = path
    return found, sorted(names - set(found))


def cmd_setup(args) -> None:
    gateway = Gateway(url=args.url)
    for name in _sets(args):
        if name in public.SOURCES:
            items, files = _public_items(name, args)
            doc_ids = gateway.ensure({n: (lambda data=data: data) for n, data in files.items()}, args.ingest_budget,
                                     refresh=args.refresh, progress=_say)
            print(f"{len(doc_ids)} documents ready for {len(items)} {name} questions ({args.corpus})")
            continue
        items = bench.select(_local_items(name), args.tags, args.skip_tags)
        names = {n for it in items for n in it.docs}
        folders = [Path(f) for f in (args.docs, args.samples) if f]
        paths, missing = _find_files(names, folders)
        if missing:
            hint = ("\nthe smoke-* files come from `uv run --with fpdf2 --with openpyxl --with pillow "
                    "scripts/make_samples.py samples`" if any(m.startswith("smoke-") for m in missing) else "")
            raise SystemExit(f"not found under {', '.join(map(str, folders))}:\n  " + "\n  ".join(missing) + hint)
        _say(f"{len(paths)} files for {len(items)} questions; PDFs take longest (the vision model reads every figure)")
        doc_ids = gateway.ensure({n: (lambda f=f: f.read_bytes()) for n, f in sorted(paths.items())}, args.ingest_budget,
                                 refresh=args.refresh, progress=_say)
        DOCMAP.parent.mkdir(parents=True, exist_ok=True)
        DOCMAP.write_text(json.dumps(dict(sorted(doc_ids.items())), indent=2) + "\n", encoding="utf-8")
        print(f"{len(doc_ids)} documents ready; file name -> doc_id map written to {DOCMAP}")


def cmd_clean(args) -> None:
    gateway = Gateway(url=args.url)
    doomed: list[dict] = []
    for name in _sets(args):
        if name in public.SOURCES:
            doomed += [d for d in gateway.documents() if d["name"].startswith(f"bench-{name}-")]
        else:
            names = {n for it in _local_items(name) for n in it.docs}
            doomed += [d for d in gateway.documents() if d["name"] in names]
    for d in doomed:
        print(f"{'delete' if args.yes else 'would delete'} {d['name']} ({d['doc_id']})")
        if args.yes:
            gateway.delete(d["doc_id"])
    if not args.yes and doomed:
        print("dry run: add --yes to delete")
    print(f"{len(doomed)} documents{' deleted' if args.yes else ''}")


def cmd_fetch(args) -> None:
    unknown = sorted(set(args.names) - set(public.SOURCES))
    if unknown:
        raise SystemExit(f"unknown datasets {unknown}; choose from {sorted(public.SOURCES)}")
    public.fetch(args.names or list(public.SOURCES), Path(args.dest) if args.dest else public_dir(), force=args.force)


# --- argument parsing -------------------------------------------------------------------------------------


def _set_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", default=None,
                   help="comma list of sets: local (eval/benchmarks/local_samples.jsonl), hard "
                        "(eval/benchmarks/hard_samples.jsonl), global (eval/benchmarks/global_samples.jsonl), forms "
                        "(eval/benchmarks/forms_samples.jsonl), totals (eval/benchmarks/totals_samples.jsonl), "
                        "hotpotqa, musique, or the path of a JSONL set "
                        "(default local; "
                        "local,hotpotqa,musique with --variants)")
    p.add_argument("--url", help="gateway API base (default $EVAL_URL, else http://localhost:8000/api)")
    p.add_argument("--tags", nargs="*", default=[], help="keep questions with any of these tags")
    p.add_argument("--skip-tags", nargs="*", default=[], help="drop questions with any of these tags")
    p.add_argument("--limit", type=int, help="local: the first N questions; public: a random subset of N (default 30)")
    p.add_argument("--seed", type=int, default=7, help="public sets: the seed of the random subset")
    p.add_argument("--corpus", choices=["per-case", "shared"], default="per-case",
                   help="public sets: each question's own paragraphs, or one shared corpus of every sampled case")
    p.add_argument("--data", help="public sets: the dataset file (default eval/datasets/public/<file>)")
    p.add_argument("--ingest-budget", type=int, default=4 * 3600, help="seconds to wait for documents to be ready")


def _default_repeats() -> int:
    raw = os.getenv("EVAL_REPEATS", "3").strip() or "3"
    if not raw.isdigit() or int(raw) < 1:
        raise SystemExit(f"EVAL_REPEATS must be a whole number of at least 1, not {raw!r}")
    return int(raw)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="rag-eval")
    sub = root.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="bootstrap a golden set from indexed chunks")
    g.add_argument("--out", default="/eval/datasets/golden.jsonl")
    g.add_argument("--n", type=int, default=120)
    g.add_argument("--unanswerable", type=float, default=0.1)
    g.add_argument("--docs", nargs="*")
    g.add_argument("--seed", type=int, default=7)
    r = sub.add_parser("retrieval", help="retriever ablations: MRR, Recall, nDCG with bootstrap CIs")
    r.add_argument("--dataset", default="/eval/datasets/golden.jsonl")
    r.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    r.add_argument("--allow-stale", action="store_true")
    a = sub.add_parser("answers", help="faithfulness, correctness, abstention, citation validity")
    a.add_argument("--dataset", default="/eval/datasets/golden.jsonl")
    a.add_argument("--modes", nargs="*", default=["single", "deep"], choices=["single", "deep"])
    a.add_argument("--limit", type=int)

    b = sub.add_parser("bench", help="run a benchmark set through the gateway, scored without a judge (RAG_TOKEN)")
    _set_args(b)
    b.add_argument("--mode", choices=["single", "deep", "exhaustive"], default="deep",
                   help="the chat request's mode (default deep); exhaustive reads every section of the selected documents")
    b.add_argument("--think", action="store_true", help="ask with Think on (thinking in the answer)")
    b.add_argument("--deep-think", action="store_true",
                   help="send deep_think: true (thinking in deep search's plan and verify steps; separate from --think)")
    b.add_argument("--repeats", type=int, default=_default_repeats(),
                   help="times each question is asked (default $EVAL_REPEATS, else 3)")
    b.add_argument("--variant", help="label for the report and its file name (default: --agent, else 'default')")
    b.add_argument("--agent", help="the chat request's agent field: a deep-search arm the rag registers (langgraph, "
                                   "deepagents, deepagents_stock, or an optional one; rag answers an unknown name with "
                                   "a 422 that lists them), passed through as given; sent only when given")
    b.add_argument("--ids", nargs="*", default=[], help="only these question ids")
    b.add_argument("--variants", help="several variants in one fair, interleaved run: a comma list of AGENT with "
                                      "optional suffixes in any order, +think (deep_think on), +fanout (fanout on), "
                                      "+calls<N> (research_calls N) and +python (the run_python tool; the stack needs "
                                      "RAG_DEEP_PYTHON=request and the compose profile sandbox), e.g. "
                                      "langgraph,langgraph+think,deepagents,"
                                      "deepagents+fanout,deepagents+fanout+calls20; any arm the rag registers; "
                                      "answer thinking stays off")
    b.add_argument("--order-seed", type=int, default=7, help="--variants: the seed of the randomized run order")
    b.add_argument("--concurrency", type=int, default=1,
                   help="requests in flight at a time, sent in the run order (default 1; 2 uses both slots of the "
                        "default llama server; latencies are then measured under load, which the reports say)")
    b.add_argument("--request-timeout", type=float, default=None,
                   help=f"seconds a request may wait for the next bytes of its reply before it is recorded as "
                        f"failed (default {DEFAULT_REQUEST_TIMEOUT:.0f} x --concurrency)")
    b.add_argument("--app-commit", help="the git commit the running app was built from, recorded in the reports")
    b.add_argument("--compose-project", default="local-rag",
                   help="compose project whose rag and llm containers are inspected (read-only) for the reports")
    b.add_argument("--reports", help="report folder (default eval/reports)")
    c = sub.add_parser("compare", help="paired comparison of two bench reports")
    c.add_argument("a", help="report A (.json)")
    c.add_argument("b", help="report B (.json)")
    c.add_argument("--metric", default="correct", help="per-question metric to compare (default correct; also "
                                                       "correct_cited, citations_ok, f1, em, contains)")
    c.add_argument("--reports", help="report folder (default eval/reports)")
    s = sub.add_parser("bench-setup", help="upload a set's files to the benchmark account and wait until ready")
    _set_args(s)
    s.add_argument("--docs", default=str(REPO_ROOT / "examples" / "documents") if REPO_ROOT else None,
                   help="folder with the sample documents (default examples/documents)")
    s.add_argument("--samples", default=str(REPO_ROOT / "samples") if REPO_ROOT else None,
                   help="folder with the smoke-* files from scripts/make_samples.py (default samples)")
    s.add_argument("--refresh", action="store_true", help="upload again even when a ready document has the name")
    k = sub.add_parser("bench-clean", help="delete a set's documents from the benchmark account (dry run without --yes)")
    _set_args(k)
    k.add_argument("--yes", action="store_true")
    f = sub.add_parser("fetch-public", help="download HotpotQA and MuSiQue dev files once into eval/datasets/public")
    f.add_argument("names", nargs="*", default=[], help=f"datasets to fetch: {', '.join(public.SOURCES)} (default: both)")
    f.add_argument("--dest", help="folder (default eval/datasets/public)")
    f.add_argument("--force", action="store_true", help="download again even if the file is there")
    return root


JUDGED = {"generate": cmd_generate, "retrieval": cmd_retrieval, "answers": cmd_answers}  # need EVAL_TOKEN and llm
PLAIN = {"bench": cmd_bench, "compare": cmd_compare, "bench-setup": cmd_setup, "bench-clean": cmd_clean,
         "fetch-public": cmd_fetch}


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        if args.cmd in PLAIN:
            PLAIN[args.cmd](args)
        else:
            JUDGED[args.cmd](args, Stack())
    except httpx.HTTPStatusError as exc:
        raise SystemExit(explain(exc)) from exc


def explain(exc: httpx.HTTPStatusError) -> str:
    """Readable messages for the errors a token-scoped run hits."""
    status, text = exc.response.status_code, exc.response.text
    if status == 401:
        return ("the gateway refused the API token (RAG_TOKEN or EVAL_TOKEN: expired, revoked or mistyped): "
                "create a new one in the user menu")
    if status == 403:
        return f"the gateway refused the request (403): {text[:200]}"
    if status == 404 and ("unknown documents" in text or "document" in text and "not found" in text):
        return "the golden set references documents this token's user does not own (or that were deleted)"
    return f"{exc.request.method} {exc.request.url.path}: HTTP {status}: {text[:300]}"


if __name__ == "__main__":
    main()
