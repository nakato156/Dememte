"""Generate e10_memory.ipynb from this Python script.

Run once:
    python notebooks/10_memory_hippocampal/_build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in dedent(src).strip("\n").splitlines()]}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in dedent(src).strip("\n").splitlines()],
    }


cells = [
    md(r"""
    # 10 — E10 Memoria asociativa biológica (TTA-only)

    Operacionaliza tres mecanismos biológicos sobre los checkpoints E6 sin
    reentrenar:

    1. **Recuperación asociativa** del codebook como Modern Hopfield
       (Ramsauer 2021; Millidge 2022).
    2. **Pattern completion** iterativo en `z_pool` con gate de familiaridad
       / unfamiliaridad (Tyulmankov 2022; Krotov 2021).
    3. **Doble vía CLS**: codebook semántico + buffer episódico EMA
       (Sun-Saxe-Fitzgerald 2023; Spens & Burgess 2024).

    La integración es una **mezcla suave en `zq_pool`** con `λ_max ≤ 0.1` para
    preservar la calibración del clasificador downstream (filosofía α-mix de
    Lim 2023 TTN trasladada del espacio BN al espacio de tokens).

    **Phase 0** corre tres pre-flight checks: si cualquiera falla, las
    variantes asociadas se descartan o se aborta el experimento.

    Bases: `e6_ema_kmeans_restart` y `e6_simvq_linear` (Phase 0 restringe la
    segunda a variantes `episodic_only` si su utilización efectiva del
    codebook es < 10%).
    """),

    code(r"""
    import sys
    from pathlib import Path

    ROOT = Path.cwd()
    while ROOT != ROOT.parent and not (ROOT / 'src' / 'dememte').exists():
        ROOT = ROOT.parent
    if str(ROOT / 'src') not in sys.path:
        sys.path.insert(0, str(ROOT / 'src'))
    print('repo root:', ROOT)
    """),

    code(r"""
    import math
    import json

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from dememte.config import e6_config
    from dememte.corruptions import STRICT_SUITE, apply_eval_corruption
    from dememte.data import build_loaders, seed_everything
    from dememte.evaluation import evaluate_dememte_suite, evaluate_dememte_tta_suite, signal_curve_rows
    from dememte.io import ensure_dir, load_checkpoint, write_csv, write_json
    from dememte.memory import (
        EpisodicBuffer,
        HippocampalConfig,
        HippocampalMemoryAdapter,
        associative_recall,
        effective_codebook,
        familiarity_gate,
    )
    from dememte.models import make_dememte_e6

    BASES = ['e6_ema_kmeans_restart', 'e6_simvq_linear']

    OUT = ensure_dir(ROOT / 'notebooks' / '10_memory_hippocampal' / 'out')
    E6_OUT = ROOT / 'notebooks' / '06_e6_zq_alignment' / 'out'

    # E10 hyperparameters — defaults grounded in the plan's literature table.
    LAMBDA_MAX = 0.1     # Lim 2023 TTN typical α-mix
    TAU = 1.0            # match vq_temperature (Ramsauer 2021 controls capacity)
    TAU_EPI = 1.0
    BETA_SEM = 1.0       # pure semantic
    BETA_MIX = 0.5       # semantic + episodic mix
    ALPHA_W = 0.1        # Sun-Saxe-Fitzgerald 2023 fast plasticity
    ALPHA_S = 0.001      # Spens & Burgess 2024 slow consolidation
    EPI_SIZE = 256       # Chandra 2023 capacity scaling

    CLEAN_FLOOR_TOL = 0.005   # Wang 2022 CoTTA / Song 2023 EcoTTA forgetting threshold

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    print('bases:', BASES)
    """),

    md(r"""
    ## Data
    """),

    code(r"""
    # All E6 variants share the same data config (seed, splits, batch_size).
    # We build loaders once from the first base.
    base_cfg = e6_config(BASES[0])
    for candidate in [ROOT / 'experiments' / 'data', ROOT / 'data', Path(base_cfg.data_dir).expanduser()]:
        candidate = candidate.resolve()
        if (candidate / 'flowers-102').exists() or candidate.name == 'flowers-102':
            base_cfg.data_dir = str(candidate)
            break
    seed_everything(base_cfg.seed)

    tr_loader, va_loader, te_loader, meta = build_loaders(
        data_dir=base_cfg.data_dir,
        batch_size=base_cfg.batch_size,
        num_workers=base_cfg.num_workers,
        val_ratio=base_cfg.val_ratio,
        split_seed=base_cfg.split_seed,
        protocol=base_cfg.benchmark_protocol,
    )
    print(meta)
    """),

    md(r"""
    ## Loaders y constructores por base
    """),

    code(r"""
    def cfg_for_base(base):
        cfg = e6_config(base)
        cfg.data_dir = base_cfg.data_dir
        return cfg


    def ckpt_for_base(base):
        return E6_OUT / base / 'best.pt'


    def load_base_model(base):
        cfg = cfg_for_base(base)
        model = make_dememte_e6(cfg, device=device)
        load_checkpoint(model, ckpt_for_base(base), device=device, strict=True)
        return model


    def write_markdown_table(rows, path):
        path = Path(path)
        ensure_dir(path.parent)
        if not rows:
            path.write_text('', encoding='utf-8')
            return
        df = pd.DataFrame(rows)
        cols = list(df.columns)
        header = '| ' + ' | '.join(str(c) for c in cols) + ' |'
        sep = '| ' + ' | '.join('---' for _ in cols) + ' |'
        body = []
        for _, row in df.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                cells.append(f'{v:.4f}' if isinstance(v, float) else str(v))
            body.append('| ' + ' | '.join(cells) + ' |')
        path.write_text('\n'.join([header, sep, *body]), encoding='utf-8')
    """),

    md(r"""
    ## Phase 0 — Pre-flight checks (gates duros antes del eval suite)

    Cualquier gate fallido restringe o aborta variantes. Resultados se
    persisten a `out/e10_phase0.json` para auditoría.
    """),

    code(r"""
    # Verificar checkpoints existen antes de Phase 0.
    missing = [b for b in BASES if not ckpt_for_base(b).exists()]
    if missing:
        raise FileNotFoundError(
            f'Faltan checkpoints E6: {missing}. Correr notebook 06 primero.'
        )
    print('checkpoints OK:', {b: str(ckpt_for_base(b)) for b in BASES})
    """),

    code(r"""
    # ---------------- P0.1 — Audit del gate de familiaridad ----------------
    # Calibrar sigma para mediana(g_clean) en [0.3, 0.7] sobre cada base, y
    # reportar mediana(g) en gaussian_noise sigma=1.5 y pixel_mask 0.75.
    # Decision: si gate familiarity se inerta en corrupt, sólo unfamiliarity
    # y const sobreviven.

    @torch.no_grad()
    def collect_min_dists(model, loader, codebook, corruption=None, severity=0.0, max_batches=8):
        model.eval()
        g_state = torch.Generator(device=device)
        corr_offset = 0 if corruption is None else sum(ord(c) for c in corruption)
        g_state.manual_seed(base_cfg.seed + int(1000 * severity) + corr_offset)
        rows = []
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            x_eval = apply_eval_corruption(x, corruption, severity, g_state)
            feats = model.backbone(x_eval)
            z = model.vqsa.projector(feats)
            z_pool = model.vqsa.pool(z).flatten(1)
            d2 = (z_pool.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(-1)
            min_d2 = d2.min(dim=-1).values
            rows.append(min_d2.detach().cpu())
        return torch.cat(rows) if rows else torch.empty(0)


    phase0 = {}
    sigma_by_base = {}

    for base in BASES:
        model = load_base_model(base).eval()
        model.requires_grad_(False)
        cb = effective_codebook(model.vq)
        if cb is None:
            print(f'[{base}] no codebook (FSQ?), skipping P0.1')
            continue
        d_clean = collect_min_dists(model, te_loader, cb)
        d_noise = collect_min_dists(model, te_loader, cb, 'gaussian_noise', 1.5)
        d_mask  = collect_min_dists(model, te_loader, cb, 'pixel_mask', 0.75)

        # Calibrate sigma so median(g_clean) ≈ 0.5.
        med_d_clean = float(d_clean.median().item())
        sigma2 = med_d_clean / math.log(2.0)
        sigma = math.sqrt(max(sigma2, 1e-8))
        sigma_by_base[base] = sigma

        def g_of(d2):
            return torch.exp(-d2 / (sigma ** 2))

        g_clean = g_of(d_clean)
        g_noise = g_of(d_noise)
        g_mask  = g_of(d_mask)

        med_g_clean = float(g_clean.median().item())
        med_g_noise = float(g_noise.median().item())
        med_g_mask  = float(g_mask.median().item())
        med_g_corrupt = min(med_g_noise, med_g_mask)

        # Decision rule from the plan.
        if med_g_corrupt > 0.05:
            gate_decision = 'familiarity viable'
        elif (1.0 - med_g_corrupt) > 0.95:
            gate_decision = 'familiarity inert in corrupt — use unfamiliarity or const'
        else:
            gate_decision = 'ambiguous — fallback to const gate'

        phase0[f'P0.1::{base}'] = dict(
            sigma=sigma,
            median_min_dist_clean=med_d_clean,
            median_g_clean=med_g_clean,
            median_g_gaussian_noise_1p5=med_g_noise,
            median_g_pixel_mask_0p75=med_g_mask,
            gate_decision=gate_decision,
        )
        print(f'[{base}] sigma={sigma:.4f}  g_clean={med_g_clean:.3f}  '
              f'g_noise={med_g_noise:.3f}  g_mask={med_g_mask:.3f}  -> {gate_decision}')
    """),

    code(r"""
    # ---------------- P0.2 — Audit del codebook (Hopfield capacity) -------
    # hard_usage < 10% ⇒ restringir esa base a variantes episodic_only.
    # Ramsauer 2021 Theorem 3: la capacidad efectiva del codebook depende del
    # número de prototipos efectivamente almacenados (no del tamaño nominal).

    HARD_USAGE_THRESHOLD = 0.10

    @torch.no_grad()
    def hard_usage_on_clean(model, loader, max_batches=20):
        model.eval()
        counts = None
        K = 0
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            _, _, dbg = model(x, return_debug=True)
            idx = dbg.get('encoding_indices')
            K = int(dbg.get('num_embeddings', 0) or 0)
            if idx is None or K == 0:
                return None
            bc = torch.bincount(idx.reshape(-1).long().cpu(), minlength=K).float()
            counts = bc if counts is None else counts + bc
        if counts is None or counts.sum().item() == 0:
            return None
        used = (counts > 0).float().mean().item()
        return float(used), K


    base_restrictions = {}
    for base in BASES:
        model = load_base_model(base).eval()
        result = hard_usage_on_clean(model, te_loader)
        if result is None:
            base_restrictions[base] = 'episodic_only_lookup_free'
            phase0[f'P0.2::{base}'] = dict(hard_usage=None, decision='lookup_free → episodic_only')
            print(f'[{base}] lookup-free quantizer → restrict to episodic_only')
            continue
        usage, K = result
        if usage < HARD_USAGE_THRESHOLD:
            base_restrictions[base] = 'episodic_only_low_usage'
            decision = f'hard_usage={usage:.3f} < {HARD_USAGE_THRESHOLD} → episodic_only'
        else:
            base_restrictions[base] = 'all_variants'
            decision = f'hard_usage={usage:.3f} ≥ {HARD_USAGE_THRESHOLD} → all variants'
        phase0[f'P0.2::{base}'] = dict(hard_usage=usage, num_embeddings=K, decision=decision)
        print(f'[{base}] {decision}')
    """),

    code(r"""
    # ---------------- P0.3 — Clean accuracy floor con assoc_recall_const ---
    # Catastrophic-forgetting gate: clean_acc(adapter, lambda=0.1) debe estar
    # a ≤ 0.5 pp por debajo de source.clean_acc. Wang 2022 / Song 2023.

    @torch.no_grad()
    def clean_acc_of_adapter(adapter, loader):
        adapter.model.eval()
        total = correct = 0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = adapter(x)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        return correct / max(1, total)


    @torch.no_grad()
    def clean_acc_of_model(model, loader):
        model.eval()
        total = correct = 0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            pred = model(x).argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        return correct / max(1, total)


    p03_results = {}
    for base in BASES:
        sigma = sigma_by_base.get(base, 1.0)
        model = load_base_model(base)
        src_acc = clean_acc_of_model(model, te_loader)

        if base_restrictions[base] == 'all_variants':
            cfg = HippocampalConfig(
                recall_sem=True, recall_epi=False, T=1,
                gate_mode='const', lambda_max=LAMBDA_MAX, tau=TAU, sigma=sigma,
            )
            adapter = HippocampalMemoryAdapter(load_base_model(base), cfg)
            adapter_acc = clean_acc_of_adapter(adapter, te_loader)
        else:
            cfg = HippocampalConfig(
                recall_sem=False, recall_epi=True, T=1,
                gate_mode='const', lambda_max=LAMBDA_MAX, tau=TAU,
                tau_epi=TAU_EPI, sigma=sigma,
            )
            adapter = HippocampalMemoryAdapter(load_base_model(base), cfg)
            adapter_acc = clean_acc_of_adapter(adapter, te_loader)

        delta = adapter_acc - src_acc
        passes = delta >= -CLEAN_FLOOR_TOL
        p03_results[base] = dict(source_clean=src_acc, adapter_clean=adapter_acc, delta=delta, passes=bool(passes))
        phase0[f'P0.3::{base}'] = p03_results[base]
        print(f'[{base}] src_clean={src_acc:.4f}  adapter_clean={adapter_acc:.4f}  '
              f'delta={delta:+.4f}  passes_floor={passes}')

    # Persist Phase 0 outputs.
    write_json(phase0, OUT / 'e10_phase0.json')
    print('Phase 0 results written to', OUT / 'e10_phase0.json')

    # Hard abort gate: if any base fails P0.3, mark it and skip.
    bases_to_skip = [b for b, r in p03_results.items() if not r['passes']]
    if bases_to_skip:
        print(f'WARNING: bases failing clean-acc floor → skipped: {bases_to_skip}')
    """),

    md(r"""
    ## Definición de variantes (post-Phase 0)
    """),

    code(r"""
    def variants_for_base(base):
        # Return list of (variant_name, HippocampalConfig) tuples for a base.
        sigma = sigma_by_base.get(base, 1.0)
        restr = base_restrictions[base]

        # Decide gate mode per base from P0.1 decision.
        decision = phase0.get(f'P0.1::{base}', {}).get('gate_decision', '')
        if 'familiarity viable' in decision:
            best_gate = 'familiarity'
        elif 'unfamiliarity' in decision:
            best_gate = 'unfamiliarity'
        else:
            best_gate = 'const'

        base_kwargs = dict(
            lambda_max=LAMBDA_MAX, tau=TAU, tau_epi=TAU_EPI, sigma=sigma,
            alpha_w=ALPHA_W, episodic_size=EPI_SIZE,
        )

        out = []

        # episodic_only is always valid (no semantic dependence).
        out.append((
            'episodic_only',
            HippocampalConfig(
                recall_sem=False, recall_epi=True, T=1, gate_mode='const',
                beta=0.0, episodic_init_from_codebook=(restr == 'all_variants'),
                **base_kwargs,
            ),
        ))

        if restr != 'all_variants':
            # P0.2 restricted this base; only episodic_only is reportable.
            return out

        # Semantic-dependent variants.
        out += [
            ('assoc_recall_const', HippocampalConfig(
                recall_sem=True, recall_epi=False, T=1, gate_mode='const',
                beta=1.0, **base_kwargs)),
            ('assoc_recall_familiarity', HippocampalConfig(
                recall_sem=True, recall_epi=False, T=1, gate_mode='familiarity',
                beta=1.0, **base_kwargs)),
            ('assoc_recall_unfamiliarity', HippocampalConfig(
                recall_sem=True, recall_epi=False, T=1, gate_mode='unfamiliarity',
                beta=1.0, **base_kwargs)),
            ('completion_T3_best_gate', HippocampalConfig(
                recall_sem=True, recall_epi=False, T=3, gate_mode=best_gate,
                beta=1.0, **base_kwargs)),
            ('hippocampal_full', HippocampalConfig(
                recall_sem=True, recall_epi=True, T=3, gate_mode=best_gate,
                beta=BETA_MIX, episodic_init_from_codebook=True, **base_kwargs)),
            ('consolidation_slow', HippocampalConfig(
                recall_sem=True, recall_epi=True, T=3, gate_mode=best_gate,
                beta=BETA_MIX, episodic_init_from_codebook=True,
                alpha_s=ALPHA_S, consolidation_every=50, **base_kwargs)),
        ]
        return out
    """),

    md(r"""
    ## Run E10 — eval suite por base × variante
    """),

    code(r"""
    all_summaries = []
    all_curves = []

    for base in BASES:
        if base in bases_to_skip:
            print(f'=== {base} skipped (P0.3 failed) ===')
            continue
        print(f'=== BASE: {base} ===')
        teacher = load_base_model(base).eval()
        teacher.requires_grad_(False)
        cfg = cfg_for_base(base)
        ckpt = ckpt_for_base(base)

        # Source baseline.
        src_metrics = evaluate_dememte_suite(load_base_model(base), te_loader, device=device)

        for variant_name, hp_cfg in variants_for_base(base):
            print(f'  -- {base} :: {variant_name}')

            def factory(hp_cfg=hp_cfg):
                model = load_base_model(base)
                return HippocampalMemoryAdapter(model, hp_cfg)

            metrics = evaluate_dememte_tta_suite(
                factory,
                te_loader,
                device=device,
                tta_method=variant_name,
                tta_base_variant=base,
                teacher_model=teacher,
            )
            clean_record = metrics.pop('clean_record')
            corrupt_records = metrics.pop('corruption_records')
            label = f'{base}::{variant_name}'
            curve_rows = signal_curve_rows(variant_name, label, clean_record, corrupt_records)

            summary = {k: v for k, v in metrics.items()
                       if isinstance(v, (int, float, bool, str, np.floating))}
            summary.update({
                'variant': variant_name,
                'label': label,
                'base_variant': base,
                'base_checkpoint': str(ckpt),
                'protocol': meta['protocol'],
                'split_seed': meta['split_seed'],
                'quantizer_type': cfg.quantizer_type,
                'delta_clean_vs_source': metrics['clean_acc'] - src_metrics['clean_acc'],
                'delta_corrupt_vs_source': metrics['corrupt_acc_avg'] - src_metrics['corrupt_acc_avg'],
                'passes_clean_floor': bool(metrics['clean_acc'] >= src_metrics['clean_acc'] - CLEAN_FLOOR_TOL),
            })
            all_summaries.append(summary)
            all_curves.extend(curve_rows)

            method_dir = ensure_dir(OUT / base / variant_name)
            write_json(summary, method_dir / 'metrics.json')
            write_csv(curve_rows, method_dir / 'signal_curves.csv')

            report_keys = ['clean_acc', 'corrupt_acc_avg', 'delta_clean_vs_source',
                           'delta_corrupt_vs_source', 'completion_amount_corrupt_avg',
                           'recall_sharpness_corrupt_avg', 'g_mean_corrupt_avg']
            print('     ', {k: round(float(summary[k]), 4) for k in report_keys if k in summary})

        # Also add source row to summaries for the base.
        src_row = {k: v for k, v in src_metrics.items()
                   if isinstance(v, (int, float, bool, str, np.floating))}
        src_row.update({
            'variant': 'source', 'label': f'{base}::source', 'base_variant': base,
            'base_checkpoint': str(ckpt), 'protocol': meta['protocol'],
            'split_seed': meta['split_seed'], 'quantizer_type': cfg.quantizer_type,
            'delta_clean_vs_source': 0.0, 'delta_corrupt_vs_source': 0.0,
            'passes_clean_floor': True,
        })
        all_summaries.append(src_row)

    write_csv(all_summaries, OUT / 'e10_results.csv')
    write_csv(all_curves, OUT / 'e10_curves.csv')

    ranked = sorted(all_summaries, key=lambda r: r.get('corrupt_acc_avg', 0.0), reverse=True)
    write_markdown_table(ranked, OUT / 'e10_summary.md')
    pd.DataFrame(ranked).head(20)
    """),

    md(r"""
    ## Lectura

    **Hito mecánico (criterio principal, no accuracy).** Antes de leer
    accuracy, verificar:

    1. `completion_amount_corrupt_avg > 0` para variantes con `T ≥ 1` →
       confirma que la inyección estructural muerde (Ramsauer 2021 Eq. 7).
    2. `episodic_buffer_churn_corrupt_avg > 0` para variantes con
       `recall_epi=True` → confirma que el buffer episódico se escribe
       (Sun-Saxe-Fitzgerald 2023 plasticidad rápida).
    3. `traj_max_step_corrupt_avg` finito y no creciente con `T` → confirma
       que el loop converge (Kim 2021).

    **Floor de clean accuracy (gate duro).** `passes_clean_floor=True` para
    cada variante reportada en el ranking final. Si una variante regresiona
    clean acc, se reporta separadamente (Wang 2022 CoTTA, Song 2023 EcoTTA).

    **Comparativa de aislamiento.** `delta_corrupt_vs_source` cuantifica la
    ganancia neta; comparar con `assoc_recall_const` para aislar si la
    ganancia viene del gate biológico / iteración / episódico, o sólo de
    "softear" el `argmin` del VQ.
    """),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "e10_memory.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
