"""Fill the models volume: the Qwen3.8-27B GGUF and vision projector, and three Hugging Face models.

Runs inside the `fetch-models` compose service (the only component that downloads anything):
    docker compose --profile setup run --rm fetch-models              # download from Hugging Face
    docker compose --profile setup run --rm fetch-models --from-dir /seed   # copy from MODELS_SEED_DIR
`--from-dir` expects the same layout as /models (gguf/ and hf/<name>/), e.g. an earlier download.
"""

import argparse
import shutil
import sys
from pathlib import Path

GGUF_REPO = "unsloth/Qwen3.8-27B-GGUF"
GGUF_FILES = ["Qwen3.8-27B-UD-Q4_K_XL.gguf", "mmproj-BF16.gguf"]
HF_MODELS = ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-VL-Embedding-2B", "Qwen/Qwen3-Reranker-0.6B"]


def copy_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        if path.is_file() and ".cache" not in path.parts:
            target = dst / path.relative_to(src)
            if target.exists() and target.stat().st_size == path.stat().st_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"copy {path.relative_to(src)}", flush=True)
            shutil.copyfile(path, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="/models")
    parser.add_argument("--from-dir", help="copy from a local folder instead of downloading")
    args = parser.parse_args()
    dest = Path(args.dest)

    if args.from_dir:
        src = Path(args.from_dir)
        for sub in ("gguf", "hf"):
            if not (src / sub).is_dir():
                sys.exit(f"{src / sub} not found; expected gguf/ and hf/ folders")
            copy_tree(src / sub, dest / sub)
    else:
        from huggingface_hub import hf_hub_download, snapshot_download  # only needed to download

        for name in GGUF_FILES:
            print(f"download {GGUF_REPO}/{name}", flush=True)
            hf_hub_download(GGUF_REPO, name, local_dir=dest / "gguf")
        for repo in HF_MODELS:
            print(f"download {repo}", flush=True)
            snapshot_download(repo, local_dir=dest / "hf" / repo.split("/")[1])
    print("models ready:", *sorted(str(p.relative_to(dest)) for p in dest.glob("*/*")), sep="\n  ")


if __name__ == "__main__":
    main()
