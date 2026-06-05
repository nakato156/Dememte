#!/usr/bin/env python3
"""Create a clean ImageNet-style 5k subset from a local clean ImageNet source.

Expected source layout:

    source/train/<wnid>/*.JPEG
    source/val/<wnid>/*.JPEG

If ``source/val`` is missing, the script can split each class from
``source/train`` using ``--split-val-from-train``. The output uses symlinks by
default, so it is cheap in disk space and keeps the provenance explicit.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def link_or_copy(src: Path, dest: Path, copy: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy:
        shutil.copy2(src, dest)
    else:
        dest.symlink_to(src.resolve())


def sample_class_files(class_dir: Path, n: int, rng: random.Random) -> list[Path]:
    files = image_files(class_dir)
    if len(files) <= n:
        return files
    return sorted(rng.sample(files, n))


def write_split(
    source_split: Path,
    out_split: Path,
    per_class: int,
    rng: random.Random,
    copy: bool,
    max_classes: int | None,
    skip_by_class: dict[str, set[Path]] | None = None,
) -> dict[str, int]:
    class_dirs = sorted(p for p in source_split.iterdir() if p.is_dir())
    if max_classes is not None:
        class_dirs = class_dirs[:max_classes]
    counts = {}
    for class_dir in class_dirs:
        skip = skip_by_class.get(class_dir.name, set()) if skip_by_class else set()
        candidates = [p for p in image_files(class_dir) if p not in skip]
        if len(candidates) <= per_class:
            files = candidates
        else:
            files = sorted(rng.sample(candidates, per_class))
        for idx, src in enumerate(files):
            dest = out_split / class_dir.name / f"{src.stem}_{idx}{src.suffix}"
            link_or_copy(src, dest, copy=copy)
        counts[class_dir.name] = len(files)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="experiments/data/imagenet")
    parser.add_argument("--out-root", default="experiments/data/imagenet-clean-5k")
    parser.add_argument("--train-per-class", type=int, default=5)
    parser.add_argument("--val-per-class", type=int, default=1)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--split-val-from-train", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    train_root = source_root / "train"
    val_root = source_root / "val"
    if not train_root.exists():
        raise FileNotFoundError(f"Missing clean ImageNet train split: {train_root}")
    if not val_root.exists() and not args.split_val_from_train:
        raise FileNotFoundError(
            f"Missing clean ImageNet val split: {val_root}. "
            "Pass --split-val-from-train only for local debugging from a clean train source."
        )
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    selected_train_by_class: dict[str, set[Path]] = {}
    class_dirs = sorted(p for p in train_root.iterdir() if p.is_dir())
    if args.max_classes is not None:
        class_dirs = class_dirs[:args.max_classes]
    for class_dir in class_dirs:
        files = sample_class_files(class_dir, args.train_per_class, rng)
        selected_train_by_class[class_dir.name] = set(files)
        for idx, src in enumerate(files):
            dest = out_root / "train" / class_dir.name / f"{src.stem}_{idx}{src.suffix}"
            link_or_copy(src, dest, copy=args.copy)

    if val_root.exists():
        val_counts = write_split(
            val_root,
            out_root / "val",
            args.val_per_class,
            rng,
            copy=args.copy,
            max_classes=args.max_classes,
        )
    else:
        val_counts = write_split(
            train_root,
            out_root / "val",
            args.val_per_class,
            rng,
            copy=args.copy,
            max_classes=args.max_classes,
            skip_by_class=selected_train_by_class,
        )

    train_counts = {wnid: len(files) for wnid, files in selected_train_by_class.items()}
    manifest = {
        "source_root": str(source_root),
        "out_root": str(out_root),
        "source_type": "clean_imagenet",
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "max_classes": args.max_classes,
        "seed": args.seed,
        "copy": bool(args.copy),
        "split_val_from_train": bool(args.split_val_from_train and not val_root.exists()),
        "train_classes": len(train_counts),
        "val_classes": len(val_counts),
        "train_images": sum(train_counts.values()),
        "val_images": sum(val_counts.values()),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
