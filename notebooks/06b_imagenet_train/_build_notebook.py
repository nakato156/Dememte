"""Build the DeMemte-ImageNet training notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


OUT = Path(__file__).with_name("e6_imagenet_train.ipynb")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


nb = nbf.v4.new_notebook()
nb["cells"] = [
    md(
        """
        # E6 ImageNet Train

        Entrena un checkpoint `dememte_imagenet_resnet50_vqsa_best.pt` usando
        ImageNet limpio. Este checkpoint es el que E10b debe cargar antes de
        evaluar memoria sobre ImageNet-C.
        """
    ),
    code(
        """
        import sys
        from pathlib import Path

        import torch
        import torch.nn as nn

        ROOT = Path.cwd()
        while ROOT.name != "Dememte" and ROOT.parent != ROOT:
            ROOT = ROOT.parent
        sys.path.insert(0, str(ROOT / "src"))

        from dememte.config import E6Config
        from dememte.data import build_imagenet_loaders, seed_everything
        from dememte.io import save_checkpoint, write_json
        from dememte.models.dememte import make_dememte_variant
        from dememte.training import initialize_vqsa_codebook, make_optimizer_vqsa, run_epoch_vqsa
        """
    ),
    code(
        """
        DATA_ROOT = ROOT / "experiments/data/imagenet-clean-5k"
        OUT_DIR = ROOT / "experiments/imagenet_dememte/out"
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        SEED = 42
        BATCH_SIZE = 64
        NUM_WORKERS = 4
        EPOCHS = 10

        # Para una prueba rápida antes del entrenamiento completo:
        MAX_TRAIN_SAMPLES_PER_CLASS = None
        MAX_VAL_SAMPLES_PER_CLASS = None
        """
    ),
    code(
        """
        seed_everything(SEED)

        cfg = E6Config(
            dataset="imagenet",
            data_dir=str(DATA_ROOT),
            num_classes=1000,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            device=DEVICE,
            backbone_name="resnet50",
            backbone_pretrained=True,
            backbone_out_channels=2048,
            quantizer_type="ema_vq",
            vq_kmeans_init=True,
            vq_kmeans_steps=10,
            dead_code_restart=True,
            dead_code_restart_after_epoch=1,
            vqsa_train_backbone=False,
            epochs_vqsa_max=EPOCHS,
            lr_vq=3e-4,
            lr_cls=1e-4,
            vq_weight=1.0,
            train_corrupt_prob=0.7,
            out_dir=str(OUT_DIR),
            seed=SEED,
        )
        """
    ),
    code(
        """
        train_loader, val_loader, data_meta = build_imagenet_loaders(
            DATA_ROOT,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            max_train_samples_per_class=MAX_TRAIN_SAMPLES_PER_CLASS,
            max_val_samples_per_class=MAX_VAL_SAMPLES_PER_CLASS,
            seed=SEED,
        )
        data_meta
        """
    ),
    code(
        """
        model = make_dememte_variant(cfg, device=DEVICE)
        criterion = nn.CrossEntropyLoss()
        codebook_initialized = initialize_vqsa_codebook(model, train_loader, cfg, DEVICE)
        optimizer = make_optimizer_vqsa(model, cfg)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )
        codebook_initialized
        """
    ),
    code(
        """
        best_acc = -1.0
        no_imp = 0
        history = []
        ckpt_path = OUT_DIR / "dememte_imagenet_resnet50_vqsa_best.pt"

        for epoch in range(1, EPOCHS + 1):
            train_metrics = run_epoch_vqsa(model, train_loader, optimizer, True, cfg, DEVICE, criterion, epoch=epoch)
            val_metrics = run_epoch_vqsa(model, val_loader, optimizer, False, cfg, DEVICE, criterion, epoch=epoch)
            scheduler.step(val_metrics["acc"])
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            print(
                f"[imagenet-vqsa {epoch:02d}] "
                f"tr_loss={train_metrics['loss']:.4f} tr_acc={train_metrics['acc']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                f"val_usage={val_metrics['hard_usage']:.4f} "
                f"val_dead={val_metrics['dead_code_fraction']:.4f} "
                f"val_hard_ppl={val_metrics['hard_perplexity']:.2f}"
            )
            if val_metrics["acc"] > best_acc + cfg.early_stop_min_delta:
                best_acc = val_metrics["acc"]
                no_imp = 0
                save_checkpoint(
                    model,
                    ckpt_path,
                    extra={
                        "config": cfg,
                        "data_meta": data_meta,
                        "epoch": epoch,
                        "best_val_acc": best_acc,
                        "codebook_initialized": codebook_initialized,
                    },
                )
            else:
                no_imp += 1
                if no_imp >= cfg.early_stop_patience:
                    print(f"early_stop_epoch={epoch}")
                    break

        write_json({"history": history, "best_val_acc": best_acc, "checkpoint": str(ckpt_path)}, OUT_DIR / "train_history.json")
        write_json({"config": cfg, "data_meta": data_meta}, OUT_DIR / "train_config.json")
        best_acc, ckpt_path
        """
    ),
]


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(OUT)
