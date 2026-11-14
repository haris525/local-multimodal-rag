"""Public multi-hop benchmarks, read from local copies: HotpotQA (distractor setting, dev) and MuSiQue
(answerable, dev). The repository ships none of their data.

`python -m rag_eval fetch-public` downloads each file once from its official source into
eval/datasets/public/ (ignored by Git), checks its SHA-256 and writes SOURCES.txt next to it:

- HotpotQA: hotpot_dev_distractor_v1.json, CC BY-SA 4.0 (https://hotpotqa.github.io/). The official host
  stopped resolving in 2026, so the Internet Archive's copy of that same URL is the fallback; the hash makes
  sure both give the same bytes.
- MuSiQue: musique_ans_v1.0_dev.jsonl from the data zip, CC BY 4.0
  (https://github.com/StonyBrookNLP/musique; the zip is the Google Drive file its download_data.sh fetches).

A case becomes small .txt documents (the paragraph title on the first line, then its text), a question,
gold answers (the answer and its aliases) and the documents that hold its supporting paragraphs. Corpus modes:

- per-case: a question selects only its own paragraphs (HotpotQA's 10, one per file; MuSiQue's 20, two
  per file, to stay within the 10 documents a question may select);
- shared: the paragraphs of every sampled case, deduplicated, packed into 10 shared files that every
  question selects, so the other cases' paragraphs are distractors as well.
"""

import hashlib
import json
import random
import re
import sys
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx

from rag_common.contracts import MAX_SELECTED_DOCS
from rag_eval.bench import Item


@dataclass
class Paragraph:
    title: str
    text: str
    supporting: bool = False

    def render(self) -> str:
        return f"{self.title}\n\n{self.text}\n"


@dataclass
class Case:
    id: str
    question: str
    answers: list[str]
    paragraphs: list[Paragraph] = field(default_factory=list)
    kind: str = ""  # HotpotQA: bridge / comparison; MuSiQue: 2hop, 3hop or 4hop


# --- loaders ------------------------------------------------------------------------------------------


def _answers(*values) -> list[str]:
    out: list[str] = []
    for value in values:
        for v in [value] if isinstance(value, str) else (value or []):
            v = str(v).strip()
            if v and v not in out:
                out.append(v)
    return out


def hotpotqa_cases(records: Iterable[dict]) -> list[Case]:
    """HotpotQA records: the official JSON ({"_id", "context": [[title, [sentences]]], "supporting_facts":
    [[title, sentence_id]]}) or the Hugging Face export (context and supporting_facts as dicts of lists)."""
    cases = []
    for i, r in enumerate(records):
        ctx, sup = r.get("context"), r.get("supporting_facts") or []
        if isinstance(ctx, dict):
            pairs = list(zip(ctx.get("title") or [], ctx.get("sentences") or []))
        else:
            pairs = [(c[0], c[1]) for c in ctx or []]
        support = set(sup.get("title") or []) if isinstance(sup, dict) else {s[0] for s in sup}
        paragraphs = [Paragraph(str(t), " ".join(str(s).strip() for s in sents if str(s).strip()), str(t) in support)
                      for t, sents in pairs]
        answers = _answers(r.get("answer"))
        if r.get("question") and answers and paragraphs:
            cases.append(Case(str(r.get("_id") or r.get("id") or f"hotpotqa-{i}"), r["question"], answers,
                              paragraphs, str(r.get("type") or "")))
    return cases


def musique_cases(records: Iterable[dict]) -> list[Case]:
    """MuSiQue JSONL records ({"id", "question", "answer", "answer_aliases", "answerable", "paragraphs": [{"title",
    "paragraph_text", "is_supporting"}]}); unanswerable records are skipped."""
    cases = []
    for i, r in enumerate(records):
        if r.get("answerable") is False:
            continue
        answers = _answers(r.get("answer"), r.get("answer_aliases"))
        paragraphs = [Paragraph(str(p.get("title") or ""), str(p.get("paragraph_text") or "").strip(),
                                bool(p.get("is_supporting"))) for p in r.get("paragraphs") or []]
        if r.get("question") and answers and paragraphs:
            cid = str(r.get("id") or f"musique-{i}")
            hops = re.match(r"\d+hop", cid)  # 2hop__..., 3hop1__...: the number of hops
            cases.append(Case(cid, r["question"], answers, paragraphs, hops.group(0) if hops else ""))
    return cases


def load_cases(name: str, path: Path) -> list[Case]:
    if name == "hotpotqa":
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        return hotpotqa_cases(raw.get("data", raw) if isinstance(raw, dict) else raw)
    if name == "musique":
        with path.open(encoding="utf-8") as f:
            return musique_cases(json.loads(line) for line in f if line.strip())
    raise ValueError(f"unknown public benchmark {name!r}")


def sample(cases: list[Case], limit: int | None, seed: int) -> list[Case]:
    """A fixed random subset: the same file, limit and seed always give the same cases, in the same order."""
    if not limit or limit >= len(cases):
        return list(cases)
    return random.Random(seed).sample(cases, limit)


# --- documents and items ------------------------------------------------------------------------------


def _key(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _pack(paragraphs: list[Paragraph], files: int) -> list[list[Paragraph]]:
    """Split paragraphs into at most `files` consecutive groups of near-equal size."""
    files = max(1, min(files, len(paragraphs)))
    size, extra = divmod(len(paragraphs), files)
    groups, start = [], 0
    for i in range(files):
        end = start + size + (1 if i < extra else 0)
        groups.append(paragraphs[start:end])
        start = end
    return groups


def _text(group: list[Paragraph]) -> bytes:
    return "\n".join(p.render() for p in group).encode("utf-8")


def build(name: str, cases: list[Case], corpus: str = "per-case") -> tuple[list[Item], dict[str, bytes]]:
    """(benchmark items, {file name: content}) for these cases. File names start with bench-<name>- so they
    are easy to find and delete, and are derived from the content's identity, so a second run reuses them."""
    files: dict[str, bytes] = {}
    items: list[Item] = []
    if corpus == "per-case":
        for case in cases:
            names, expected = [], []
            for j, group in enumerate(_pack(case.paragraphs, MAX_SELECTED_DOCS), 1):
                fname = f"bench-{name}-{_key(case.id)}-{j:02d}.txt"
                files[fname] = _text(group)
                names.append(fname)
                if any(p.supporting for p in group):
                    expected.append(fname)
            items.append(_item(name, case, names, expected))
    elif corpus == "shared":
        unique: dict[tuple[str, str], Paragraph] = {}
        for case in cases:
            for p in case.paragraphs:
                unique.setdefault((p.title, p.text), Paragraph(p.title, p.text))
        ordered = sorted(unique.values(), key=lambda p: _key(p.title + "\n" + p.text, 16))
        corpus_id = _key("\n".join(sorted(c.id for c in cases)), 8)
        where: dict[tuple[str, str], str] = {}
        names = []
        for j, group in enumerate(_pack(ordered, MAX_SELECTED_DOCS), 1):
            fname = f"bench-{name}-shared-{corpus_id}-{j:02d}.txt"
            files[fname] = _text(group)
            names.append(fname)
            where.update({(p.title, p.text): fname for p in group})
        for case in cases:
            expected = sorted({where[(p.title, p.text)] for p in case.paragraphs if p.supporting})
            items.append(_item(name, case, names, expected))
    else:
        raise ValueError(f"corpus is per-case or shared, not {corpus!r}")
    return items, files


def _item(name: str, case: Case, docs: list[str], expected: list[str]) -> Item:
    return Item(id=f"{name}-{case.id}", question=case.question, docs=docs, expected_docs=expected or list(docs),
                tags=[name, case.kind] if case.kind else [name], reference=case.answers[0], answers=case.answers,
                correct_by="contains", origin=name)


# --- download once --------------------------------------------------------------------------------------

DRIVE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"  # the zip StonyBrookNLP/musique download_data.sh fetches (272 MB)
SOURCES: dict[str, dict] = {
    "hotpotqa": {
        "file": "hotpot_dev_distractor_v1.json",
        "license": "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)",
        "home": "https://hotpotqa.github.io/",
        "urls": ["http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
                 "https://web.archive.org/web/20260310132809id_/"
                 "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"],
        "sha256": "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea",
        "citation": "Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov and Manning (2018). HotpotQA: A Dataset for "
                    "Diverse, Explainable Multi-hop Question Answering. EMNLP.",
    },
    "musique": {
        "file": "musique_ans_v1.0_dev.jsonl",
        "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "home": "https://github.com/StonyBrookNLP/musique",
        "urls": [f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t"],
        "archive": "zip",
        "sha256": "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",  # the extracted file
        "citation": "Trivedi, Balasubramanian, Khot and Sabharwal (2022). MuSiQue: Multihop Questions via "
                    "Single-hop Question Composition. TACL.",
    },
}
MAX_DOWNLOAD = 600 * 2**20  # bytes; the files are smaller (46 MB, and a 272 MB zip)


def _download(url: str, target: Path, client: httpx.Client) -> str:
    """Stream url into target; returns the SHA-256 of what was written."""
    digest, size = hashlib.sha256(), 0
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with target.open("wb") as out:
            for chunk in response.iter_bytes(1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise ValueError(f"{url} is larger than {MAX_DOWNLOAD >> 20} MB")
                digest.update(chunk)
                out.write(chunk)
    return digest.hexdigest()


def _extract(archive: Path, member_name: str, target: Path) -> str:
    """Copy the one member called member_name (at any depth) out of the zip; never writes other paths."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if Path(m.filename).name == member_name and "__MACOSX" not in m.filename]
        if not members:
            raise ValueError(f"{member_name} is not in {archive.name}")
        with z.open(members[0]) as src, target.open("wb") as out:
            while chunk := src.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
    return digest.hexdigest()


def fetch(names: Iterable[str], folder: Path, client: httpx.Client | None = None, force: bool = False) -> list[Path]:
    """Download each named dataset into folder unless it is already there; verify its SHA-256; record the
    source and license in SOURCES.txt. Returns the dataset files."""
    folder.mkdir(parents=True, exist_ok=True)
    client = client or httpx.Client(timeout=httpx.Timeout(60, read=300), follow_redirects=True,
                                    headers={"User-Agent": "rag-eval-benchmark-fetch"})
    out = []
    for name in names:
        src = SOURCES[name]
        target = folder / src["file"]
        if target.exists() and not force:
            print(f"{name}: {target} already present, not downloading again")
            out.append(target)
            continue
        part = target.with_name(target.name + ".part")
        errors = []
        for url in src["urls"]:
            print(f"{name}: downloading {url}", flush=True)
            try:
                if src.get("archive") == "zip":
                    archive = target.with_name(target.name + ".zip.part")
                    try:
                        _download(url, archive, client)
                        got = _extract(archive, src["file"], part)
                    finally:
                        archive.unlink(missing_ok=True)
                else:
                    got = _download(url, part, client)
            except (httpx.HTTPError, ValueError, zipfile.BadZipFile, OSError) as exc:
                part.unlink(missing_ok=True)
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
                print(f"{name}: failed ({type(exc).__name__}); trying the next source", file=sys.stderr)
                continue
            if src.get("sha256") and got != src["sha256"]:
                part.unlink(missing_ok=True)
                errors.append(f"{url}: SHA-256 {got} is not the recorded {src['sha256']}")
                continue
            part.replace(target)
            _record(folder, name, url, got)
            print(f"{name}: saved {target} ({target.stat().st_size:,} bytes, sha256 {got[:12]}...)")
            out.append(target)
            break
        else:
            raise SystemExit(f"{name}: every source failed:\n  " + "\n  ".join(errors))
    return out


def _record(folder: Path, name: str, url: str, sha256: str) -> None:
    src = SOURCES[name]
    line = (f"{src['file']}\n  dataset: {name}\n  license: {src['license']}\n  home: {src['home']}\n"
            f"  downloaded from: {url}\n  sha256: {sha256}\n  date: {datetime.now(UTC):%Y-%m-%d}\n"
            f"  cite: {src['citation']}\n\n")
    with (folder / "SOURCES.txt").open("a", encoding="utf-8") as f:
        f.write(line)
