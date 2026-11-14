"""Where eval keeps its files. In a source checkout (running on the host with `uv run --directory eval`) that is
the repository's eval/ folder; in the eval container it is /eval (the compose file mounts datasets and reports
there). EVAL_REPORTS and EVAL_PUBLIC_DIR override the two folders a run writes or reads."""

import os
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKOUT = (EVAL_ROOT / "pyproject.toml").is_file() and (EVAL_ROOT / "src" / "rag_eval").is_dir()
REPO_ROOT = EVAL_ROOT.parent if SOURCE_CHECKOUT else None


def eval_dir(sub: str) -> Path:
    return (EVAL_ROOT if SOURCE_CHECKOUT else Path("/eval")) / sub


def reports_dir() -> Path:
    return Path(os.getenv("EVAL_REPORTS") or eval_dir("reports"))


def public_dir() -> Path:
    return Path(os.getenv("EVAL_PUBLIC_DIR") or eval_dir("datasets") / "public")


LOCAL_SET = eval_dir("benchmarks") / "local_samples.jsonl"
HARD_SET = eval_dir("benchmarks") / "hard_samples.jsonl"  # multi-step and cross-document questions
GLOBAL_SET = eval_dir("benchmarks") / "global_samples.jsonl"  # whole-document questions (exhaustive mode)
FORMS_SET = eval_dir("benchmarks") / "forms_samples.jsonl"  # questions where names are written in more than one form
TOTALS_SET = eval_dir("benchmarks") / "totals_samples.jsonl"  # questions where a total row changes the answer
NAMED_SETS = {"local": LOCAL_SET, "hard": HARD_SET, "global": GLOBAL_SET,
              "forms": FORMS_SET, "totals": TOTALS_SET}  # --set names for the sets in eval/benchmarks
