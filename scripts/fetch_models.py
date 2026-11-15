"""Fill the models volume: the Qwen3.8-27B GGUF and vision projector, and three Hugging Face models.

Runs inside the `fetch-models` compose service (the only component that downloads anything):
    docker compose --profile setup run --rm fetch-models              # download from Hugging Face
    docker compose --profile setup run --rm fetch-models --from-dir /seed   # copy from MODELS_SEED_DIR
`--from-dir` expects the same layout as /models (gguf/ and hf/<name>/), e.g. an earlier download.
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

GGUF_REPO = "unsloth/Qwen3.8-27B-GGUF"
GGUF_FILES = ["Qwen3.8-27B-UD-Q4_K_XL.gguf", "mmproj-BF16.gguf"]
HF_MODELS = ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-VL-Embedding-2B", "Qwen/Qwen3-Reranker-0.6B"]


def copy_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        relative = path.relative_to(src)
        if path.is_file() and ".cache" not in relative.parts:
            target = dst / relative
            if target.exists() and target.stat().st_size == path.stat().st_size:
                target.chmod(0o644)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"copy {relative}", flush=True)
            # An interrupted copy must not leave a partial weight file that looks installed.
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                temporary = Path(out.name)
            try:
                shutil.copyfile(path, temporary)
                temporary.chmod(0o644)  # read-only bind mounts must be readable by non-root containers
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="/models")
    parser.add_argument("--from-dir", help="copy from a local folder instead of downloading")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="reuse Hugging Face's host cache for HF models, then copy into --dest")
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
            target = dest / "hf" / repo.split("/")[1]
            if args.reuse_cache:
                cached = snapshot_download(repo)
                copy_tree(Path(cached), target)
            else:
                snapshot_download(repo, local_dir=target)
    print("models ready:", *sorted(str(p.relative_to(dest)) for p in dest.glob("*/*")), sep="\n  ")


if __name__ == "__main__":
    main()
