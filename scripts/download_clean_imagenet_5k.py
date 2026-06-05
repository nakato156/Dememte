#!/usr/bin/env python3
"""Download a clean ImageNet-1K 5k subset from the gated Hugging Face mirror.

Prerequisites:
  1. Accept access to https://huggingface.co/datasets/ILSVRC/imagenet-1k
  2. Export a token with that access:
       export HF_TOKEN=hf_...

The output is ImageFolder-compatible:

    experiments/data/imagenet-clean-5k/train/<wnid>/*.JPEG
    experiments/data/imagenet-clean-5k/val/<wnid>/*.JPEG

This subset is clean ImageNet, not ImageNet-C. Training still uses DeMemte's
on-the-fly clean+corrupt augmentation inside ``run_epoch_vqsa``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from PIL import Image


REPO_ID = "ILSVRC/imagenet-1k"
TRAIN_PREFIX = "data/train-"
VAL_PREFIX = "data/validation-"


def load_label_to_wnid(class_index_path: Path | None = None) -> dict[int, str]:
    if class_index_path is None:
        class_index_path = Path.home() / ".cache" / "dememte" / "imagenet_class_index.json"
    if not class_index_path.exists():
        raise FileNotFoundError(
            f"Missing ImageNet class index: {class_index_path}. "
            "Run any DeMemte ImageNet loader once or provide --class-index-path."
        )
    raw = json.loads(class_index_path.read_text(encoding="utf-8"))
    return {int(idx): value[0] for idx, value in raw.items()}


def list_split_files(token: str, prefix: str) -> list[str]:
    api = HfApi(token=token)
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    return sorted(path for path in files if path.startswith(prefix) and path.endswith(".parquet"))


def image_bytes(value) -> bytes:
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is not None:
            return data
    if hasattr(value, "as_py"):
        return image_bytes(value.as_py())
    raise ValueError(f"Unsupported image payload type: {type(value)!r}")


def save_image(payload: bytes, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(payload)) as img:
        img.convert("RGB").save(dest, format="JPEG", quality=95)


def extract_split(
    split: str,
    files: list[str],
    per_class: int,
    out_root: Path,
    token: str,
    label_to_wnid: dict[int, str],
    cache_dir: Path,
) -> dict[str, int]:
    counts = {wnid: 0 for wnid in label_to_wnid.values()}
    total_needed = len(counts) * per_class
    total_written = 0
    for shard_idx, filename in enumerate(files):
        if total_written >= total_needed:
            break
        local = hf_hub_download(
            REPO_ID,
            filename,
            repo_type="dataset",
            token=token,
            cache_dir=str(cache_dir),
        )
        table = pq.read_table(local, columns=["image", "label"])
        images = table["image"]
        labels = table["label"].to_pylist()
        for row_idx, label in enumerate(labels):
            wnid = label_to_wnid[int(label)]
            if counts[wnid] >= per_class:
                continue
            dest = out_root / split / wnid / f"{split}_{shard_idx:03d}_{row_idx:06d}.JPEG"
            save_image(image_bytes(images[row_idx]), dest)
            counts[wnid] += 1
            total_written += 1
            if total_written >= total_needed:
                break
        done_classes = sum(1 for value in counts.values() if value >= per_class)
        print(f"{split}: shard={filename} written={total_written}/{total_needed} done_classes={done_classes}/1000")
    missing = {wnid: count for wnid, count in counts.items() if count < per_class}
    if missing:
        raise RuntimeError(f"{split} incomplete: {len(missing)} classes missing samples. Example: {list(missing.items())[:5]}")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default="experiments/data/imagenet-clean-5k")
    parser.add_argument("--cache-dir", default="experiments/data/downloads/hf-imagenet")
    parser.add_argument("--class-index-path", default=None)
    parser.add_argument("--train-per-class", type=int, default=5)
    parser.add_argument("--val-per-class", type=int, default=1)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.token:
        raise SystemExit(
            "Missing HF_TOKEN. Accept https://huggingface.co/datasets/ILSVRC/imagenet-1k "
            "and run: export HF_TOKEN=hf_..."
        )
    out_root = Path(args.out_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    label_to_wnid = load_label_to_wnid(Path(args.class_index_path).expanduser() if args.class_index_path else None)
    try:
        train_files = list_split_files(args.token, TRAIN_PREFIX)
        val_files = list_split_files(args.token, VAL_PREFIX)
        train_counts = extract_split("train", train_files, args.train_per_class, out_root, args.token, label_to_wnid, cache_dir)
        val_counts = extract_split("val", val_files, args.val_per_class, out_root, args.token, label_to_wnid, cache_dir)
    except GatedRepoError as exc:
        raise SystemExit(
            "ImageNet-1K is gated. Accept the terms on Hugging Face with the same account as HF_TOKEN."
        ) from exc
    except HfHubHTTPError as exc:
        raise SystemExit(f"Hugging Face download failed: {exc}") from exc

    manifest = {
        "source": f"huggingface:{REPO_ID}",
        "source_type": "clean_imagenet",
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "train_images": sum(train_counts.values()),
        "val_images": sum(val_counts.values()),
        "train_classes": len(train_counts),
        "val_classes": len(val_counts),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
