"""Model code for the embedding benchmark: embedders, exact dense search and the reranker, with the production
settings of services/models (BF16, sentence-transformers, the same prompts, max_seq_length 1024, text batch 32,
rerank batch 16, sigmoid rerank scores). Needs torch and a GPU: run it in the local-rag/models image."""

import gc
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

# the defaults of services/models/src/models_svc/config.py (a test keeps them equal). The query one is also General's
# default since search tasks; the earlier one, "Given a question, retrieve passages that answer it", is measured with
# --query-instruction-text.
QUERY_INSTRUCTION = "Given a question about the user's documents, retrieve passages with the facts needed to answer it"
RERANK_INSTRUCTION = "Given a question about the user's documents, judge whether the passage helps answer it"
MAX_SEQ_TOKENS = 1024
TEXT_BATCH = 32
RERANK_BATCH = 16
GIB = 2**30
QUERY_MODES = ("app", "model-default", "none")  # the query instruction: the app's, the model's own, or none


@dataclass(frozen=True)
class Arm:
    name: str
    path: Path
    kind: str  # "text": Qwen3-Embedding (Instruct/Query prompt); "vl": Qwen3-VL-Embedding (instruction as prompt)


def gpu_used_gib() -> float | None:
    """Whole-card memory in use (nvidia-smi), or None where nvidia-smi is not available."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        return round(float(out.split()[0]) / 1024, 2)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class Embedder:
    def __init__(self, arm: Arm, device: str = "cuda", query_mode: str = "app", query_text: str | None = None):
        """query_text, when given, is the instruction itself (empty: none) and overrides query_mode."""
        if query_mode not in QUERY_MODES:
            raise ValueError(f"query_mode must be one of {QUERY_MODES}")
        self.arm = arm
        self.query_mode = query_mode
        self.query_text = query_text
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        t = time.perf_counter()
        self.model = SentenceTransformer(str(arm.path), device=device, model_kwargs={"dtype": torch.bfloat16})
        self.model.max_seq_length = MAX_SEQ_TOKENS
        self.load_s = time.perf_counter() - t
        self.weights_gib = (torch.cuda.memory_allocated() - before) / GIB
        self.dim = self.model.get_sentence_embedding_dimension()

    def _query_kwargs(self) -> dict:
        """app: as services/models (text: "Instruct: ...\\nQuery:"; VL: the instruction as its prompt);
        model-default: the model's own query prompt (text: its "query" prompt; VL: its default prompt);
        none: no instruction (text: an empty prompt; VL: its default prompt, which it always applies).
        A query_text replaces all three: "Instruct: <text>\\nQuery:" (text) or the text as the prompt (VL)."""
        if self.query_text is not None:
            if self.arm.kind == "vl":
                return {"prompt": self.query_text} if self.query_text else {}
            return {"prompt": f"Instruct: {self.query_text}\nQuery:" if self.query_text else ""}
        if self.query_mode == "app":
            return {"prompt": f"Instruct: {QUERY_INSTRUCTION}\nQuery:" if self.arm.kind == "text" else QUERY_INSTRUCTION}
        if self.arm.kind == "vl":
            return {}
        return {"prompt_name": "query"} if self.query_mode == "model-default" else {"prompt": ""}

    def passages(self, texts: list[str]) -> np.ndarray:
        # Passages get no prompt (text) or the model's default prompt (VL), as at ingest time.
        kwargs = {"prompt": ""} if self.arm.kind == "text" else {}
        v = self.model.encode(texts, batch_size=TEXT_BATCH, normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False, **kwargs)
        return v.astype(np.float32)

    def queries(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, batch_size=TEXT_BATCH, normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False, **self._query_kwargs())
        return v.astype(np.float32)

    def query_latency_ms(self, texts: list[str]) -> dict[str, float]:
        """One question at a time, as a chat request embeds it (after one warm-up call)."""
        self.queries(texts[:1])
        times = []
        for text in texts:
            torch.cuda.synchronize()
            t = time.perf_counter()
            self.queries([text])
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
        return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95))}

    def peak_gib(self) -> float:
        return torch.cuda.max_memory_allocated() / GIB

    def close(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def dense_top(passages: np.ndarray, queries: np.ndarray, k: int, allowed: list[np.ndarray | None],
              device: str = "cuda", batch: int = 256) -> list[list[int]]:
    """Exact cosine search (vectors are normalized), best first; `allowed[i]` masks query i's passages."""
    p = torch.from_numpy(passages).to(device)
    out: list[list[int]] = []
    for start in range(0, len(queries), batch):
        q = torch.from_numpy(queries[start:start + batch]).to(device)
        s = q @ p.T
        for j in range(s.shape[0]):
            mask = allowed[start + j]
            row = s[j]
            if mask is not None:
                row = row.masked_fill(~torch.from_numpy(mask).to(device), float("-inf"))
            kk = min(k, int(mask.sum()) if mask is not None else row.numel())
            vals, idx = torch.topk(row, kk)
            out.append([int(i) for i, v in zip(idx.tolist(), vals.tolist(), strict=True) if v != float("-inf")])
    del p
    torch.cuda.empty_cache()
    return out


class Reranker:
    def __init__(self, path: Path, device: str = "cuda", instruction: str = RERANK_INSTRUCTION):
        self.model = CrossEncoder(str(path), device=device, model_kwargs={"dtype": torch.bfloat16},
                                  prompts={"rag": instruction}, default_prompt_name="rag",
                                  max_length=MAX_SEQ_TOKENS)

    def scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        s = self.model.predict(pairs, batch_size=RERANK_BATCH, activation_fn=torch.nn.Sigmoid(),
                               show_progress_bar=False)
        return [float(x) for x in s]
