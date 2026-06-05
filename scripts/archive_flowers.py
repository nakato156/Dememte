#!/usr/bin/env python3
"""Archive the legacy Flowers102 dataset and optionally delete the source."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def make_archive(source: Path, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("zstd"):
        archive = archive_dir / f"{source.name}.tar.zst"
        run(["tar", "--zstd", "-cf", str(archive), "-C", str(source.parent), source.name])
    else:
        archive = archive_dir / f"{source.name}.tar.gz"
        run(["tar", "-czf", str(archive), "-C", str(source.parent), source.name])
    return archive


def verify_archive(archive: Path, expected_root: str) -> None:
    listing = subprocess.run(["tar", "-tf", str(archive)], check=True, text=True, capture_output=True)
    if not any(line.split("/", 1)[0] == expected_root for line in listing.stdout.splitlines()):
        raise RuntimeError(f"Archive {archive} does not contain expected root {expected_root!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="experiments/data/flowers-102")
    parser.add_argument("--archive-dir", default="experiments/data_archives")
    parser.add_argument("--delete-source", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    archive_dir = Path(args.archive_dir).expanduser().resolve()
    if not source.exists():
        print(f"source_missing={source}")
        return
    archive = make_archive(source, archive_dir)
    verify_archive(archive, source.name)
    digest = sha256_file(archive)
    digest_path = archive.with_suffix(archive.suffix + ".sha256")
    digest_path.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"archive={archive}")
    print(f"sha256={digest}")
    if args.delete_source:
        shutil.rmtree(source)
        print(f"deleted_source={source}")


if __name__ == "__main__":
    main()
