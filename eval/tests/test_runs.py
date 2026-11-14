import pytest

from rag_eval.client import ChatOutcome
from rag_eval.generate import generate
from rag_eval.runs import answers


class FakeStack:
    """The faithfulness judge returns broken JSON once; everything else answers normally."""

    def __init__(self):
        self.calls = 0

    def chat(self, question, doc_ids, mode):
        return ChatOutcome(answer="North had 120 k$ [1].", sources=[{"n": 1, "text": "North: 120 k$"}], citations=[1])

    def llm_json(self, prompt, max_tokens=800, temperature=0.0):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("Expecting ',' delimiter")
        return {"claims": [{"supported": True}], "score": 1, "abstained": True}

    def documents(self):
        return [{"doc_id": "d" * 32, "name": "a.txt", "status": "ready"}]

    def chunks(self, doc_id):
        return [{"doc_id": doc_id, "kind": "text", "n_tokens": 3, "text": "too short"}]


def test_a_malformed_judge_answer_costs_one_metric_not_the_run():
    items = [{"id": f"q{i}", "question": "North Q3?", "reference_answer": "120 k$", "doc_ids": ["d" * 32],
              "answerable": True} for i in range(2)]
    summary = answers(FakeStack(), items, ["single"])["summary"]["single"]
    means = {metric: interval[0] for metric, interval in summary.items()}  # bootstrap_ci gives (mean, low, high)
    assert means["judge failures"] == 0.25  # 1 of 4 judge calls (2 items x faithfulness + correctness)
    assert means["correctness"] == 1.0 and means["faithfulness"] == 1.0  # both items scored; one lost faithfulness


def test_generation_stops_cleanly_when_no_chunk_is_long_enough(tmp_path):
    with pytest.raises(SystemExit, match="no chunks"):
        generate(FakeStack(), tmp_path / "q.jsonl", n=5, unanswerable=0.2, doc_ids=None, seed=1)
