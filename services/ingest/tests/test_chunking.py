import pandas as pd

from ingest.chunking import ChunkingConfig, prose_chunks, row_chunks, split_sentences, table_chunks
from ingest.elements import Element, Sheet

CFG = ChunkingConfig(text_target=40, text_max=60, min_tail=10, table_max=60)


def test_split_sentences_respects_abbreviations_initials_and_decimals():
    text = "Dr. Smith measured 3.5 kg, e.g. in Fig. 2. J. Doe agreed. Results follow!"
    assert split_sentences(text) == ["Dr. Smith measured 3.5 kg, e.g. in Fig. 2.", "J. Doe agreed.", "Results follow!"]


def test_prose_chunks_stay_under_cap_end_on_sentences_and_merge_short_tail(counter):
    sentence = "The committee reviewed the budget and approved every line item in full."
    paragraphs = [Element(kind="text", text=" ".join([sentence] * 3), section="A", page=1),
                  Element(kind="text", text=" ".join([sentence] * 4), section="A", page=2),
                  Element(kind="text", text="Short end.", section="A", page=2)]
    chunks = prose_chunks(paragraphs, "doc.pdf", counter, CFG)
    assert all(c.n_tokens <= CFG.text_max for c in chunks)
    assert all(c.text.rstrip().endswith(".") for c in chunks)
    assert chunks[-1].text.endswith("Short end.") and chunks[-1].n_tokens > CFG.min_tail
    assert chunks[0].search_text.startswith("Document: doc.pdf | Section: A\n")
    assert (chunks[0].page, chunks[-1].page_end) == (1, 2)


def test_table_windows_repeat_header_and_cover_every_row_once(counter):
    frame = pd.DataFrame({"Region": [f"R{i}" for i in range(30)], "Revenue": range(30)},
                         index=pd.Index(range(5, 35), name="row"))
    chunks = table_chunks(frame, "Document: d", "Table 1: revenue", counter, CFG, page=3)
    assert len(chunks) > 1
    assert all(c.text.startswith("Table 1: revenue\n| Region | Revenue |") for c in chunks)
    spans = [(c.row_start, c.row_end) for c in chunks]
    assert spans[0][0] == 5 and spans[-1][1] == 34
    assert all(b[0] == a[1] + 1 for a, b in zip(spans, spans[1:]))


def test_row_with_long_text_cell_is_split_and_each_piece_keeps_the_key_fields(counter):
    note = " ".join(["The battery overheated during charging and the unit shut down."] * 12)
    sheet = Sheet("Tickets", pd.DataFrame({"Ticket": ["T-1"], "Status": ["open"], "Notes": [note]},
                                          index=pd.Index([7], name="row")))
    chunks = row_chunks(sheet, "tickets", "t.xlsx", counter, ChunkingConfig(text_max=60, long_cell_tokens=20))
    assert len(chunks) > 1
    assert all(c.text.startswith("Ticket: T-1\nStatus: open\nNotes: ") for c in chunks)
    assert {(c.row_start, c.table) for c in chunks} == {(7, "tickets")}


def test_row_over_the_cap_without_long_cells_is_split_never_dropped(counter):
    """The census sheet's rows vanished from the index: long column names, short numbers, no 'long cell'."""
    long = "Population Estimate as of July 1 of the year for the area named in the first column"
    frame = pd.DataFrame({"Area": [".Ohio"], **{f"{long} {y}": [11_000_000 + y] for y in range(2018, 2026)}},
                         index=pd.Index([40], name="row"))
    chunks = row_chunks(Sheet("Pop", frame), "pop", "census.xlsx", counter, CFG)
    assert len(chunks) > 1 and all(c.n_tokens <= CFG.text_max for c in chunks)
    assert all(c.text.startswith("Area: .Ohio") and c.row_start == 40 for c in chunks)
    text = "\n".join(c.text for c in chunks)
    assert all(str(11_000_000 + y) in text for y in range(2018, 2026))


def test_text_after_a_figure_or_table_in_the_same_section_starts_its_own_run():
    """A paragraph after a figure joined the figure's run, and the chunker reads a run by its first element:
    every paragraph after the first figure or table of a section was dropped (the Mistral paper lost its
    sliding window and rolling buffer paragraphs)."""
    from ingest.chunking import group_by_section

    els = [Element(kind="text", text="a", section="S"), Element(kind="image", text="cap", section="S"),
           Element(kind="text", text="b", section="S"), Element(kind="text", text="c", section="S"),
           Element(kind="table", text="t", section="S"), Element(kind="text", text="d", section="S"),
           Element(kind="text", text="e", section="T")]
    assert [[e.text for e in run] for run in group_by_section(els)] == [["a"], ["cap"], ["b", "c"], ["t"], ["d"], ["e"]]


def test_the_pipeline_indexes_every_paragraph_around_a_figure(tmp_path, counter):
    from types import SimpleNamespace

    from ingest.config import Settings
    from ingest.elements import ParsedDocument
    from ingest.pipeline import Pipeline

    parsed = ParsedDocument(elements=[
        Element(kind="text", text="Before the figure.", section="2 Architecture", page=2),
        Element(kind="image", text="Figure 1: Sliding window attention.", section="2 Architecture", page=2,
                image=b"png"),
        Element(kind="text", text="The theoretical attention span is about 131K tokens.", section="2 Architecture",
                page=2),
        Element(kind="text", text="The cache uses 8x less memory.", section="2 Architecture", page=2),
    ])
    pipe = Pipeline(Settings(data_dir=tmp_path, audit_url=None, docling_artifacts=None), None, None, None, counter, None)
    chunks = pipe.chunk(SimpleNamespace(doc_id="d" * 32, name="mistral.pdf"), parsed, [])
    text = "\n".join(c.text for c in chunks)
    assert "131K tokens" in text and "8x less memory" in text and "Before the figure" in text
    assert [c.kind for c in chunks].count("image") == 1
