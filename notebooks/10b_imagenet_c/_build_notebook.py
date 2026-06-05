"""Build the E10 ImageNet-C migration notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


OUT = Path(__file__).with_name("e10_imagenet_c.ipynb")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


nb = nbf.v4.new_notebook()
nb["cells"] = [
    md(
        """
        # E10 ImageNet-C

        Migración de DeMemte desde Flowers-102 hacia ImageNet/ImageNet-C.
        `source_resnet50_imagenet` es el baseline de clasificación real; las
        variantes DeMemte/E10 usan `num_classes=1000` y el backbone ImageNet.
        """
    ),
    code(
        """
        import json
        import sys
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import torch

        ROOT = Path.cwd()
        while ROOT.name != "Dememte" and ROOT.parent != ROOT:
            ROOT = ROOT.parent
        sys.path.insert(0, str(ROOT / "src"))

        from dememte.config import E6Config
        from dememte.data import build_imagenet_c_loader
        from dememte.evaluation import evaluate_baseline, evaluate_dememte, evaluate_dememte_tta
        from dememte.io import load_checkpoint
        from dememte.memory import HippocampalConfig, HippocampalMemoryAdapter
        from dememte.models import make_imagenet_resnet50
        from dememte.models.dememte import make_dememte_variant
        """
    ),
    code(
        """
        DATA_ROOT = ROOT / "experiments/data/imagenet-c-subset"
        OUT_DIR = ROOT / "notebooks/10b_imagenet_c/out"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        DEMEMTE_CHECKPOINT = ROOT / "experiments/imagenet_dememte/out/dememte_imagenet_resnet50_vqsa_best.pt"

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        BATCH_SIZE = 64
        NUM_WORKERS = 4
        CORRUPTIONS = ["gaussian_noise", "motion_blur", "pixelate", "jpeg_compression"]
        SEVERITIES = [3, 5]
        MAX_SAMPLES_PER_CLASS = None
        RETURN_PREDICTIONS = True
        """
    ),
    code(
        """
        def scalar_row(record, variant, corruption, severity):
            row = {"variant": variant, "corruption": corruption, "severity": severity}
            for key, value in record.items():
                if isinstance(value, (int, float, bool, np.floating)):
                    row[key] = value
            return row


        def condition_loaders():
            for corruption in CORRUPTIONS:
                for severity in SEVERITIES:
                    loader, meta = build_imagenet_c_loader(
                        DATA_ROOT,
                        corruption,
                        severity,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,
                        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
                        seed=42,
                    )
                    yield corruption, severity, loader, meta
        """
    ),
    code(
        """
        resnet50 = make_imagenet_resnet50(device=DEVICE).eval()
        dememte_cfg = E6Config(
            dataset="imagenet_c",
            data_dir=str(DATA_ROOT),
            num_classes=1000,
            backbone_name="resnet50",
            backbone_out_channels=2048,
            quantizer_type="ema_vq",
            vq_kmeans_init=False,
            dead_code_restart=False,
        )
        dememte = make_dememte_variant(dememte_cfg, device=DEVICE).eval()
        if DEMEMTE_CHECKPOINT.exists():
            payload = load_checkpoint(dememte, DEMEMTE_CHECKPOINT, device=DEVICE, strict=True)
            print(f"Loaded DeMemte-ImageNet checkpoint: {DEMEMTE_CHECKPOINT}")
            print({k: payload.get(k) for k in ("epoch", "best_val_acc", "codebook_initialized")})
        else:
            print(f"WARNING: missing DeMemte-ImageNet checkpoint: {DEMEMTE_CHECKPOINT}")
            print("Train on clean ImageNet first; E10 variants would otherwise use a random head/codebook.")
        dememte.eval()
        """
    ),
    code(
        """
        variants = {
            "dememte_imagenet_source": None,
            "e10_assoc_recall_imagenet_c": HippocampalConfig(
                recall_sem=True, recall_epi=False, T=1, beta=1.0, lambda_max=0.1, gate_mode="const"
            ),
            "e10_episodic_imagenet_c": HippocampalConfig(
                recall_sem=False, recall_epi=True, T=1, beta=0.0, lambda_max=0.1, gate_mode="const"
            ),
            "e10_dual_memory_imagenet_c": HippocampalConfig(
                recall_sem=True, recall_epi=True, T=1, beta=0.5, lambda_max=0.1, gate_mode="unfamiliarity"
            ),
        }
        """
    ),
    code(
        """
        rows = []
        curve_rows = []
        prediction_rows = []

        for corruption, severity, loader, meta in condition_loaders():
            print(f"Evaluating {corruption}/{severity} ({meta['size']} samples)")
            source = evaluate_baseline(
                resnet50,
                loader,
                device=DEVICE,
                return_predictions=RETURN_PREDICTIONS,
            )
            rows.append(scalar_row(source, "source_resnet50_imagenet", corruption, severity))
            curve_rows.append(scalar_row(source, "source_resnet50_imagenet", corruption, severity))
            if RETURN_PREDICTIONS:
                for item in source.pop("predictions", []):
                    prediction_rows.append({"variant": "source_resnet50_imagenet", **item})

            dememte_source = evaluate_dememte(
                dememte,
                loader,
                device=DEVICE,
                return_predictions=RETURN_PREDICTIONS,
            )
            rows.append(scalar_row(dememte_source, "dememte_imagenet_source", corruption, severity))
            curve_rows.append(scalar_row(dememte_source, "dememte_imagenet_source", corruption, severity))
            if RETURN_PREDICTIONS:
                for item in dememte_source.pop("predictions", []):
                    prediction_rows.append({"variant": "dememte_imagenet_source", **item})

            for name, cfg in variants.items():
                if cfg is None:
                    continue
                adapter = HippocampalMemoryAdapter(dememte, cfg)
                record = evaluate_dememte_tta(
                    adapter,
                    loader,
                    device=DEVICE,
                    return_predictions=RETURN_PREDICTIONS,
                    tta_method="e10_hippocampal_memory",
                    tta_base_variant=name,
                )
                rows.append(scalar_row(record, name, corruption, severity))
                curve_rows.append(scalar_row(record, name, corruption, severity))
                if RETURN_PREDICTIONS:
                    for item in record.pop("predictions", []):
                        prediction_rows.append({"variant": name, **item})
        """
    ),
    code(
        """
        results = pd.DataFrame(rows)
        curves = pd.DataFrame(curve_rows)
        preds = pd.DataFrame(prediction_rows)

        results.to_csv(OUT_DIR / "e10_imagenet_c_results.csv", index=False)
        curves.to_csv(OUT_DIR / "e10_imagenet_c_curves.csv", index=False)
        if len(preds):
            preds.to_csv(OUT_DIR / "e10_imagenet_c_predictions.csv", index=False)

        summary = []
        summary.append("# E10 ImageNet-C Summary\\n")
        summary.append(f"- conditions: {len(CORRUPTIONS) * len(SEVERITIES)}")
        summary.append(f"- device: {DEVICE}")
        summary.append(f"- max_samples_per_class: {MAX_SAMPLES_PER_CLASS}")
        summary.append("\\n## Corrupt Accuracy Avg\\n")
        if "acc" in results:
            summary.append(results.groupby("variant")["acc"].mean().sort_values(ascending=False).to_markdown())
        summary.append("\\n## Codebook / Memory Signals\\n")
        signal_cols = [
            "hard_usage_mean", "dead_code_fraction_mean", "hard_perplexity_mean",
            "dq_mean_mean", "assignment_entropy_mean", "codebook_perplexity_mean",
            "recall_sharpness_mean", "completion_amount_mean", "g_mean_mean",
            "episodic_buffer_churn_mean",
        ]
        available = [c for c in signal_cols if c in results.columns]
        if available:
            summary.append(results.groupby("variant")[available].mean().to_markdown())
        (OUT_DIR / "e10_imagenet_c_summary.md").write_text("\\n".join(summary), encoding="utf-8")
        results
        """
    ),
]


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(OUT)
