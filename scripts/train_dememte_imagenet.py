#!/usr/bin/env python3
"""Train a DeMemte-ImageNet checkpoint with a frozen ResNet-50 backbone."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dememte.config import E6Config  # noqa: E402
from dememte.data import build_imagenet_loaders, seed_everything  # noqa: E402
from dememte.io import save_checkpoint, write_json  # noqa: E402
from dememte.models.dememte import make_dememte_variant  # noqa: E402
from dememte.training import initialize_vqsa_codebook, make_optimizer_vqsa, run_epoch_vqsa  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="experiments/data/imagenet-clean-5k")
    parser.add_argument("--out-dir", default="experiments/imagenet_dememte/out")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-samples-per-class", type=int, default=None)
    parser.add_argument("--max-val-samples-per-class", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--lr-vq", type=float, default=3e-4)
    parser.add_argument("--lr-cls", type=float, default=1e-4)
    parser.add_argument("--vq-weight", type=float, default=1.0)
    parser.add_argument("--train-corrupt-prob", type=float, default=0.7)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--no-kmeans-init", action="store_true")
    parser.add_argument("--no-dead-restart", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = E6Config(
        dataset="imagenet",
        data_dir=args.data_root,
        num_classes=1000,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        backbone_name="resnet50",
        backbone_pretrained=True,
        backbone_out_channels=2048,
        quantizer_type="ema_vq",
        vq_kmeans_init=not args.no_kmeans_init,
        vq_kmeans_steps=10,
        dead_code_restart=not args.no_dead_restart,
        dead_code_restart_after_epoch=1,
        vqsa_train_backbone=bool(args.train_backbone),
        epochs_vqsa_max=args.epochs,
        lr_vq=args.lr_vq,
        lr_cls=args.lr_cls,
        vq_weight=args.vq_weight,
        train_corrupt_prob=args.train_corrupt_prob,
        out_dir=str(out_dir),
        seed=args.seed,
    )

    train_loader, val_loader, data_meta = build_imagenet_loaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples_per_class=args.max_train_samples_per_class,
        max_val_samples_per_class=args.max_val_samples_per_class,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        seed=args.seed,
    )
    print(json.dumps(data_meta, indent=2))

    model = make_dememte_variant(cfg, device=args.device)
    criterion = nn.CrossEntropyLoss()
    initialized = initialize_vqsa_codebook(model, train_loader, cfg, args.device)
    optimizer = make_optimizer_vqsa(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )

    best_acc = -1.0
    best_path = out_dir / "dememte_imagenet_resnet50_vqsa_best.pt"
    history = []
    no_imp = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch_vqsa(model, train_loader, optimizer, True, cfg, args.device, criterion, epoch=epoch)
        val_metrics = run_epoch_vqsa(model, val_loader, optimizer, False, cfg, args.device, criterion, epoch=epoch)
        scheduler.step(val_metrics["acc"])
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"[imagenet-vqsa {epoch:02d}] "
            f"tr_loss={train_metrics['loss']:.4f} tr_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_usage={val_metrics['hard_usage']:.4f} val_dead={val_metrics['dead_code_fraction']:.4f} "
            f"val_hard_ppl={val_metrics['hard_perplexity']:.2f}"
        )
        if val_metrics["acc"] > best_acc + cfg.early_stop_min_delta:
            best_acc = val_metrics["acc"]
            no_imp = 0
            save_checkpoint(
                model,
                best_path,
                extra={
                    "config": asdict(cfg),
                    "data_meta": data_meta,
                    "epoch": epoch,
                    "best_val_acc": best_acc,
                    "codebook_initialized": initialized,
                },
            )
        else:
            no_imp += 1
            if no_imp >= cfg.early_stop_patience:
                print(f"early_stop_epoch={epoch}")
                break

    write_json({"history": history, "best_val_acc": best_acc, "checkpoint": str(best_path)}, out_dir / "train_history.json")
    write_json({"config": asdict(cfg), "data_meta": data_meta}, out_dir / "train_config.json")
    print(f"best_checkpoint={best_path}")
    print(f"best_val_acc={best_acc:.6f}")


if __name__ == "__main__":
    main()
