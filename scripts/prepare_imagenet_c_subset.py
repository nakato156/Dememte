#!/usr/bin/env python3
"""Download and extract a disk-conscious ImageNet-C subset from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path


ZENODO_RECORD = "https://zenodo.org/records/2235448/files"
TARBALLS = {
    "noise.tar": {
        "md5": "e80562d7f6c3f8834afb1ecf27252745",
        "corruptions": ("gaussian_noise",),
    },
    "blur.tar": {
        "md5": "2d8e81fdd8e07fef67b9334fa635e45c",
        "corruptions": ("motion_blur",),
    },
    "digital.tar": {
        "md5": "89157860d7b10d5797849337ca2e5c03",
        "corruptions": ("pixelate", "jpeg_compression"),
    },
}
DEFAULT_SEVERITIES = ("3", "5")


def file_md5(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    mode = "wb"
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
        mode = "ab"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response, tmp.open(mode) as f:
        shutil.copyfileobj(response, f, length=1024 * 1024)
    tmp.rename(dest)


def is_safe_member(member: tarfile.TarInfo) -> bool:
    path = Path(member.name)
    return not path.is_absolute() and ".." not in path.parts


def wanted_member(member_name: str, corruptions: set[str], severities: set[str]) -> bool:
    parts = Path(member_name).parts
    if len(parts) < 3:
        return False
    return parts[0] in corruptions and parts[1] in severities


def extract_subset(tar_path: Path, out_root: Path, corruptions: set[str], severities: set[str]) -> int:
    count = 0
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not is_safe_member(member):
                raise RuntimeError(f"Unsafe tar member: {member.name}")
            if wanted_member(member.name, corruptions, severities):
                tar.extract(member, out_root)
                if member.isfile():
                    count += 1
    return count


def count_condition_files(root: Path, corruptions: set[str], severities: set[str]) -> dict[str, int]:
    counts = {}
    for corruption in sorted(corruptions):
        for severity in sorted(severities, key=int):
            condition = root / corruption / severity
            counts[f"{corruption}/{severity}"] = sum(1 for p in condition.rglob("*") if p.is_file()) if condition.exists() else 0
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default="experiments/data/imagenet-c-subset")
    parser.add_argument("--download-dir", default="experiments/data/downloads/imagenet-c")
    parser.add_argument("--severities", nargs="+", default=list(DEFAULT_SEVERITIES))
    parser.add_argument("--keep-tars", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-md5", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    download_dir = Path(args.download_dir).expanduser().resolve()
    severities = {str(s) for s in args.severities}
    out_root.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    all_corruptions: set[str] = set()
    for tar_name, meta in TARBALLS.items():
        corruptions = set(meta["corruptions"])
        all_corruptions.update(corruptions)
        tar_path = download_dir / tar_name
        url = f"{ZENODO_RECORD}/{tar_name}?download=1"
        if not args.skip_download and not tar_path.exists():
            print(f"download={url}")
            download(url, tar_path)
        if not tar_path.exists():
            raise FileNotFoundError(f"Missing tarball: {tar_path}")
        if not args.skip_md5:
            actual = file_md5(tar_path)
            if actual != meta["md5"]:
                raise RuntimeError(f"MD5 mismatch for {tar_name}: expected {meta['md5']}, got {actual}")
        extracted = extract_subset(tar_path, out_root, corruptions, severities)
        print(f"extracted_files[{tar_name}]={extracted}")
        if not args.keep_tars:
            os.remove(tar_path)
            print(f"removed_tar={tar_path}")

    counts = count_condition_files(out_root, all_corruptions, severities)
    manifest = out_root / "subset_manifest.txt"
    manifest.write_text("\n".join(f"{k} {v}" for k, v in sorted(counts.items())) + "\n", encoding="utf-8")
    print(f"manifest={manifest}")
    for key, value in sorted(counts.items()):
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
