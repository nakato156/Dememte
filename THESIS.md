# THESIS.md — DeMemte claims ledger

> Fuente única de la narrativa. beads = tracking del trabajo; `insights.md` = detalle por
> experimento; este archivo = qué afirmamos, con qué evidencia y en qué estado.
> `inform` (Typst) se DERIVA de aquí, no al revés, y jala números con `#csv()` directo de
> los CSV. `HANDOFF.md` es scratchpad de sesión, no contrato.
>
> **Regla absoluta: cero números en este archivo.** Ninguna celda lleva un valor, delta,
> pp ni porcentaje — sólo la ruta al CSV/JSON que lo contiene. Si un número cambia, cambia
> el CSV; este ledger no se toca. Sin excepciones, sin "ilustrativos".

## Spine (un párrafo)

DeMemte = backbone congelado + VQSA como sustrato. La "memoria" útil NO es el codebook
paramétrico (zq) sino **retrieval episódico no-paramétrico que vota en el logit**. Claim
central: ese retrieval rompe el trade-off clean↔corrupt (sube ambos a la vez), con coste de
calibración acotado. El codebook se reposiciona como memoria paramétrica que se ablaciona y
resulta inerte/aliasada — sobrevive como resultado negativo, no como el mecanismo.

## Vocabulario de estado (campo `estado`)

- `sostenido`     — evidencia directa en ≥1 dominio, reproducible desde el CSV citado.
- `parcial`       — señal positiva con deuda abierta (dominio único, coste no acotado, base débil).
- `negativo-útil` — el claim ES un resultado negativo bien diagnosticado; aporta a la tesis.
- `abierto`       — hipótesis sin evidencia suficiente todavía (o evidencia que no la aísla).

(No hay estado `refutado`: ningún claim actual está refutado en puro — los negativos son
`negativo-útil` y las hipótesis sin aislar son `abierto`.)

## Claims ledger

`verificado` = fecha del último cotejo fila ↔ CSV. Antes de confiar en una fila para escribir
el `inform`, releer su(s) CSV si la fecha es vieja.

| id | claim | estado | evidencia (ruta) | dominios | verificado | tensión / deuda | issue |
|----|-------|--------|------------------|----------|------------|-----------------|-------|
| C1 | aug+FT compra corrupt a costa de clean (existe el trade-off) | `abierto` | `notebooks/01_baseline/out/baseline_curves.csv`; `notebooks/04_finetune_vs_frozen/out/comparison_curves.csv` | Flowers | 2026-06-05 | nb01/nb04 solo tienen frozen vs FT-aug, y FT-aug domina en clean Y corrupt → el trade-off NO está aislado. Falta baseline FT-clean (alto clean/bajo corrupt) para contrastar. Sin él, la premisa de la tesis no está demostrada en Flowers | — |
| C2 | memoria latente conservadora (λ≤0.1, LN affine, memreg que pina drift) es inerte en accuracy | `negativo-útil` | `notebooks/08_e7b_tta/out/e7b_results.csv`; `notebooks/09_e7c_codebook/out/e7c_results.csv`; `notebooks/10_memory_hippocampal/out/e10_results.csv`; `notebooks/10b_imagenet_c/out/e10_imagenet_c_results.csv` | Flowers, ImageNet-C | 2026-06-05 | causa diagnosticada: clasificador frozen + downstream; LN gradiente estructuralmente nulo (E7b); soft-mix se lava antes del logit (E10, ambos dominios) | wtq |
| C3 | retrieval no-paramétrico que vota en el logit (`logits_base + α·logits_cache`) SÍ mueve la predicción | `sostenido` | `notebooks/11_retrieval_memory/out/e11_results.csv` | ImageNet-C | 2026-06-05 | — | wtq.1 |
| C4 | ese retrieval rompe el trade-off (sube clean Y corrupt simultáneo) | `parcial` | `notebooks/11_retrieval_memory/out/e11_results.csv` (variante `source_cache_z_pool_fixed_alpha`) | ImageNet-C | 2026-06-05 | (a) un solo dominio; falta curva de severidad. (b) **mejora relativa al sustrato congelado (`dememte_imagenet_source`), no a ResNet50 plano**, que rinde muy por encima en ImageNet-C (`notebooks/10b_imagenet_c/out/e10_imagenet_c_results.csv`, fila `source_resnet50_imagenet`). Reportar el claim como mejora sobre el sustrato frozen, no como SOTA | — |
| C5 | la representación útil para recuperar vecinos es `z_pool` (continuo, pre-cuantización), no `zq_pool` | `sostenido` | `notebooks/11_retrieval_memory/out/e11_results.csv`; `notebooks/11_retrieval_memory/insights.md` (Insight 2) | ImageNet-C | 2026-06-05 | — | — |
| C6 | el codebook (`zq_pool`) como clave de memoria está aliasado: vota con fuerza pero sin precisión | `negativo-útil` | `notebooks/11_retrieval_memory/out/e11_results.csv` (variante `source_cache_zq_pool_fixed_alpha`) | ImageNet-C | 2026-06-05 | reframe del pivote: zq deja de ser "la memoria" y pasa a ablación negativa que justifica el uso de z_pool. Gate de unfamiliarity lo rescata parcialmente pero queda lejos de z_pool | — |
| C7 | memoria episódica online con pseudo-labels se contamina (rompe más de lo que repara) | `negativo-útil` | `notebooks/11_retrieval_memory/out/e11_results.csv` (variantes `episodic_cache_zq_pool`, `dual_cache_zq_pool`) | ImageNet-C | 2026-06-05 | lección para el claim biológico: plasticidad episódica sin filtro de verdad = auto-refuerzo, no memoria útil | — |
| C8 | la ganancia de retrieval tiene coste de calibración (ECE/NLL corrupto empeoran) | `parcial` | `notebooks/11_retrieval_memory/out/e11_results.csv` (cols `ece_corrupt_avg`, `nll_corrupt_avg`) | ImageNet-C | 2026-06-05 | DEUDA central del epic: el criterio de éxito exige acotar ECE/NLL, no accuracy a secas. Candidatos: temperatura/escala separada de `logits_cache`, gate por margen/acuerdo | wtq |

## Cobertura OOD (qué shift, dónde, estado del experimento)

Sin números — solo rol, eje de shift y estado de ejecución. El slate objetivo del pivote a
retrieval es Flowers + ImageNet (C y R) + CIFAR-C.

| dataset / backbone | rol en la tesis | tipo de shift | granularidad | severidad | estado |
|--------------------|-----------------|---------------|--------------|-----------|--------|
| Flowers-102 / RN18 | control fino + baselines del trade-off + ablaciones de codebook | sintético (gaussian_noise, pixel_mask, cutout, blur) | fino | grid 3 niveles | baselines hechos; retrieval E11 NO portado aún |
| ImageNet-C / RN50 | escala | sintético (gaussian_noise, motion_blur, pixelate, jpeg) | medio | sev 3, 5 | E10 + E11 hechos |
| ImageNet-R o Sketch / RN50 | refutar "solo es denoising de ruido sintético" | **natural / semántico** (renditions/sketch) | medio | n/a | PENDIENTE |
| CIFAR-10-C / CIFAR-100-C | curva de severidad barata | sintético (15 corrupciones) | grueso / medio | **1–5** | PENDIENTE (se ordena después) |

## Tensiones abiertas (resumen vivo)

- **Premisa del trade-off sin aislar (C1):** falta el baseline FT-clean en Flowers. Mientras
  no exista, la motivación "frozen rompe un trade-off" no está demostrada con los CSV en disco;
  lo que hay muestra FT-aug dominando a frozen.
- **Base ImageNet débil (C4):** el checkpoint `dememte_imagenet_resnet50_vqsa` rinde muy por
  debajo de ResNet50 plano en ImageNet-C. El positivo E11 es sobre esa base, no sobre SOTA —
  decidir si se reporta como "mejora relativa al sustrato congelado" o se fortalece la base.
- **Dominio del positivo (C3/C4):** sostenido solo en ImageNet-C. Portar E11 a Flowers y añadir
  shift natural (ImageNet-R) y curva (CIFAR-C) para sacar C4 de `parcial`.
- **Calibración (C8):** criterio de éxito del epic = ganancia robusta CON ECE/NLL acotado.
- **Pivote narrativo consumado:** zq→z_pool (C5/C6); el codebook es ahora ablación, no mecanismo.

## Cómo se actualiza

1. Cada experimento nuevo → fila(s) de claim + puntero a su CSV + (si aplica) issue beads + `verificado` con la fecha de hoy.
2. El estado cambia solo con evidencia en `out/`. `parcial` → `sostenido` exige 2º dominio o curva de severidad.
3. `verificado` se actualiza cada vez que se relee el CSV de la fila; si está viejo, releer antes de citar en el `inform`.
4. Antes de tocar el `inform` (Typst), este ledger debe estar al día — es el contrato.
