"""The sheet-layout regression corpus (tests/fixtures/layouts): every adversarial sheet and the real sample
workbooks read exactly as expected.json records (columns, row counts, SQL types, which reader), and on every
sheet that the simple reader reads right (simple_right.txt, labelled by hand), the layout reader gives the same
columns and rows."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "layouts"))
from corpus import EXPECTED, HERE, SAMPLES, simple_of, simple_right, tables_of  # noqa: E402

EXP = json.loads(EXPECTED.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(EXP["corpus"]))
def test_every_corpus_sheet_reads_as_expected(name):
    assert tables_of(HERE / name) == EXP["corpus"][name]


@pytest.mark.parametrize("name", sorted(EXP["samples"]))
def test_the_real_samples_read_as_expected(name):
    path = SAMPLES / name
    if not path.exists():
        pytest.skip(f"sample {name} not in this checkout")
    assert tables_of(path) == EXP["samples"][name]


@pytest.mark.parametrize("name", simple_right())
def test_the_layout_reader_reads_like_main_where_main_reads_right(name):
    new = [(t["columns"], t["rows"]) for t in tables_of(HERE / name)]
    main = [(t["columns"], t["rows"]) for t in simple_of(HERE / name)]
    assert new == main


def test_the_corpus_covers_every_group_and_the_parity_list_is_labelled():
    names = set(EXP["corpus"])
    prefixes = ("r1-", "r2-", "r3-", "r4-", "r5-", "r6-", "r7-", "r8-", "first-", "supp-")
    counts = {p: sum(n.startswith(p) for n in names) for p in prefixes}
    assert counts == {"r1-": 42, "r2-": 7, "r3-": 21, "r4-": 23, "r5-": 2, "r6-": 29, "r7-": 18, "r8-": 7, "first-": 6,
                      "supp-": 51}
    assert {p.name for p in HERE.iterdir() if p.suffix in (".csv", ".xlsx")} == names
    listed = simple_right()
    assert len(listed) == len(set(listed)) and set(listed) <= names and len(listed) >= 152
    assert "r3-transposed_kv.csv" not in listed and "r1-two_tables_side_by_side.xlsx" not in listed
