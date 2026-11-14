from rag_common.index import ChunkPayload

from rag.fusion import Candidate
from rag.prompts import build_sources, location, parse_citations
from rag.terms import exact_terms


def payload(idx: int, text: str, tokens: int, **kw) -> ChunkPayload:
    return ChunkPayload(doc_id="d", doc_name="r.pdf", doc_type="pdf", kind="text", chunk_index=idx, text=text,
                        search_text=text, text_sha="x", n_tokens=tokens, **kw)


def test_exact_terms_keeps_codes_numbers_quotes_and_acronyms_only():
    q = 'How many units of INV-2025-0042 cost $1,250 in Q3 per the "net revenue" line of the SKU table?'
    assert exact_terms(q) == ["net revenue", "INV-2025-0042", "$1,250", "Q3", "SKU"]
    assert exact_terms("what is the vacation policy for new staff") == []


def test_parse_citations_splits_groups_and_flags_unknown_numbers():
    assert parse_citations("A [1][3]. B [2, 5]. C [x]", [1, 2, 3]) == ([1, 2, 3], [5])
    assert parse_citations("A [2] B [4]", [1, 4, 7]) == ([4], [2])  # sparse numbers after the budget cut


def test_location_labels():
    assert location(payload(0, "t", 1, page=3, page_end=4, section="Results")) == "pp. 3-4, Results"
    assert location(payload(0, "t", 1, sheet="Orders", row_start=17, row_end=17)) == 'sheet "Orders", row 17'


def test_build_sources_widens_with_neighbours_only_within_budget():
    cands = [Candidate("c1", payload(5, "middle", 100, page=2)), Candidate("c2", payload(9, "other", 100, page=4))]
    prev, nxt = payload(4, "before", 50, page=1), payload(6, "after", 50, page=2)
    sources = build_sources(cands, {"c1": (prev, nxt)}, fixed=[], budget=330)
    assert [s.n for s in sources] == [1, 2]
    assert sources[0].text == "before\n\nmiddle" and sources[0].location == "pp. 1-2"  # 'after' did not fit
