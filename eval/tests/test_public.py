import hashlib
import io
import json
import zipfile

import httpx
import pytest

from rag_eval import public
from rag_eval.public import build, hotpotqa_cases, load_cases, musique_cases, sample

# One HotpotQA dev question, shortened (Yang et al., 2018; CC BY-SA 4.0; see THIRD_PARTY_NOTICES.md).
HOTPOT = {"_id": "5a8b57f25542995d1e6f1371", "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
          "answer": "yes", "type": "comparison", "level": "hard",
          "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
          "context": [["Ed Wood (film)", [" Ed Wood is a 1994 film.", " It stars Johnny Depp."]],
                      ["Scott Derrickson", [" Scott Derrickson is an American director."]],
                      ["Ed Wood", [" Edward Wood Jr. was an American filmmaker."]]]}
MUSIQUE = {"id": "3hop1__1_2_3", "question": "Who owns the label?", "answer": "Warner Music Group",
           "answer_aliases": ["Warner Music", "Warner Music Group"], "answerable": True,
           "paragraphs": [{"idx": i, "title": f"T{i}", "paragraph_text": f"Paragraph {i}.", "is_supporting": i in (3, 17)}
                          for i in range(20)]}


def test_hotpotqa_official_and_hugging_face_shapes_give_the_same_case():
    official = hotpotqa_cases([HOTPOT])[0]
    hf = hotpotqa_cases([{"id": HOTPOT["_id"], "question": HOTPOT["question"], "answer": "yes", "type": "comparison",
                          "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
                          "context": {"title": [c[0] for c in HOTPOT["context"]],
                                      "sentences": [c[1] for c in HOTPOT["context"]]}}])[0]
    assert official == hf
    assert official.answers == ["yes"] and official.kind == "comparison"
    assert [p.supporting for p in official.paragraphs] == [False, True, True]
    assert official.paragraphs[0].text == "Ed Wood is a 1994 film. It stars Johnny Depp."


def test_musique_aliases_hops_and_unanswerable_records():
    cases = musique_cases([MUSIQUE, {**MUSIQUE, "id": "2hop__9_9", "answerable": False}])
    assert len(cases) == 1
    case = cases[0]
    assert case.answers == ["Warner Music Group", "Warner Music"] and case.kind == "3hop"
    assert sum(p.supporting for p in case.paragraphs) == 2


def test_the_subset_is_fixed_by_its_seed():
    cases = musique_cases([{**MUSIQUE, "id": f"2hop__{i}"} for i in range(50)])
    first, again = sample(cases, 30, 7), sample(cases, 30, 7)
    assert [c.id for c in first] == [c.id for c in again] and len(first) == 30
    assert [c.id for c in sample(cases, 30, 8)] != [c.id for c in first]
    assert len(sample(cases, 100, 7)) == 50


def test_per_case_documents_stay_within_ten_and_name_the_supporting_files():
    items, files = build("musique", musique_cases([MUSIQUE]), "per-case")
    item = items[0]
    assert len(item.docs) == 10 and len(files) == 10  # 20 paragraphs, two per file
    assert all(name.startswith("bench-musique-") and name.endswith(".txt") for name in files)
    assert len(item.expected_docs) == 2 and set(item.expected_docs) <= set(item.docs)
    assert files[item.docs[0]].decode() == "T0\n\nParagraph 0.\n\nT1\n\nParagraph 1.\n"
    assert item.correct_by == "contains" and item.answers[0] == "Warner Music Group" and item.tags == ["musique", "3hop"]
    again, _ = build("musique", musique_cases([MUSIQUE]), "per-case")
    assert again[0].docs == item.docs  # the same case always gets the same file names


def test_shared_corpus_deduplicates_and_every_question_selects_it():
    second = {**HOTPOT, "_id": "other", "question": "Second?", "answer": "Johnny Depp",
              "supporting_facts": [["Ed Wood (film)", 1]],
              "context": HOTPOT["context"][:1] + [["New", [" A new paragraph."]]]}
    items, files = build("hotpotqa", hotpotqa_cases([HOTPOT, second]), "shared")
    assert len(files) == 4  # 4 distinct paragraphs, fewer than 10 files
    assert items[0].docs == items[1].docs == sorted(files)
    blob = b"".join(files.values())
    assert blob.count(b"Ed Wood (film)") == 1
    film = next(name for name, data in files.items() if b"Ed Wood (film)" in data)
    assert items[1].expected_docs == [film]
    with pytest.raises(ValueError):
        build("hotpotqa", [], "everything")


def test_load_cases_reads_both_files(tmp_path):
    (tmp_path / "h.json").write_text(json.dumps([HOTPOT]), encoding="utf-8")
    (tmp_path / "m.jsonl").write_text(json.dumps(MUSIQUE) + "\n\n", encoding="utf-8")
    assert load_cases("hotpotqa", tmp_path / "h.json")[0].id == HOTPOT["_id"]
    assert load_cases("musique", tmp_path / "m.jsonl")[0].id == MUSIQUE["id"]


def _zip(name: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("__MACOSX/data/._" + name, b"junk")
        z.writestr("data/" + name, data)
    return buf.getvalue()


def test_fetch_falls_back_verifies_extracts_records_and_never_downloads_twice(tmp_path, monkeypatch):
    hot, mus = b'[{"_id": "x"}]', b'{"id": "2hop__1"}\n'
    monkeypatch.setitem(public.SOURCES, "hotpotqa", {**public.SOURCES["hotpotqa"], "urls": ["http://dead.example/h.json",
                        "http://mirror.example/h.json"], "sha256": hashlib.sha256(hot).hexdigest()})
    monkeypatch.setitem(public.SOURCES, "musique", {**public.SOURCES["musique"], "urls": ["http://drive.example/m.zip"],
                        "sha256": hashlib.sha256(mus).hexdigest()})
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "dead.example":
            raise httpx.ConnectError("name does not resolve")
        if request.url.host == "mirror.example":
            return httpx.Response(200, content=hot)
        return httpx.Response(200, content=_zip("musique_ans_v1.0_dev.jsonl", mus))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    paths = public.fetch(["hotpotqa", "musique"], tmp_path, client=client)
    assert [p.read_bytes() for p in paths] == [hot, mus]
    assert calls == ["dead.example", "mirror.example", "drive.example"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["SOURCES.txt", "hotpot_dev_distractor_v1.json",
                                                          "musique_ans_v1.0_dev.jsonl"]  # no zip or .part left
    record = (tmp_path / "SOURCES.txt").read_text(encoding="utf-8")
    assert "CC BY-SA 4.0" in record and "CC BY 4.0" in record and "http://mirror.example/h.json" in record
    public.fetch(["hotpotqa", "musique"], tmp_path, client=client)
    assert len(calls) == 3  # already present: nothing downloaded


def test_fetch_refuses_a_file_whose_hash_differs(tmp_path, monkeypatch):
    monkeypatch.setitem(public.SOURCES, "hotpotqa", {**public.SOURCES["hotpotqa"], "urls": ["http://x.example/h.json"],
                        "sha256": "0" * 64})
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"tampered")))
    with pytest.raises(SystemExit, match="SHA-256"):
        public.fetch(["hotpotqa"], tmp_path, client=client)
    assert not any(tmp_path.glob("hotpot*"))


def test_recorded_sources_and_licenses():
    assert public.SOURCES["hotpotqa"]["license"].startswith("CC BY-SA 4.0")
    assert public.SOURCES["musique"]["license"].startswith("CC BY 4.0")
    assert public.SOURCES["hotpotqa"]["urls"][0] == "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    assert all(len(s["sha256"]) == 64 for s in public.SOURCES.values())
