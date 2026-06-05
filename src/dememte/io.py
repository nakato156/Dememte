"""Checkpoint and metrics I/O — small wrappers used by every notebook."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import torch


def _serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    if is_dataclass(value):
        return _serializable(asdict(value))
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(model, path, extra: dict = None) -> None:
    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(_serializable(extra))
    ensure_dir(Path(path).parent)
    torch.save(payload, path)


def load_checkpoint(model, path, device: str = "cuda", strict: bool = False) -> dict:
    payload = torch.load(path, map_location=device)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return payload


def write_json(obj, path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serializable(obj), f, indent=2)


def write_csv(rows: Iterable[dict], path) -> None:
    rows = list(rows)
    ensure_dir(Path(path).parent)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _serializable(row.get(k)) for k in fieldnames})
