# THESIS.md — DeMemte claims ledger

> Fuente única de la narrativa. beads = tracking del trabajo; `insights.md` = detalle por
> experimento; este archivo = qué afirmamos, con qué evidencia y en qué estado.
> El `inform` (hoy `inform.tex` en LaTeX; objetivo: migrar a Typst y jalar números con
> `#csv()` directo de los CSV) se DERIVA de aquí, no al revés. `HANDOFF.md` es scratchpad
> de sesión, no contrato.
>
> **Regla absoluta: cero números de resultados en este archivo.** Ninguna celda lleva un
> valor de métrica (accuracy, delta, pp, ECE/NLL, porcentaje) — sólo la ruta al CSV/JSON
> que lo contiene. Sí están permitidos los que NO son resultados: fechas de cotejo, ids de
> claim (C1…), nombres de dataset/backbone e hiperparámetros que *definen* un claim (λ≤0.1,
> sev 1–5). Si un número de resultado cambia, cambia el CSV; este ledger no se toca.

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
| C1 | aug+FT compra corrupt a costa de clean (existe el trade-off) | `sostenido` | `notebooks/01_baseline/out/baseline_summary.csv`; `notebooks/12_flowers_retrieval/out/ft_clean_baseline.csv` | Flowers | 2026-06-06 | trade-off AISLADO con la pata FT-clean que faltaba: FT-clean (alto clean/bajo corrupt) vs FT-aug (menor clean/mayor corrupt) muestra que la augmentation compra corrupt a costa de clean. Reproducible desde los CSV citados (perfiles enfrentados) | wtq.5 |
| C2 | memoria latente conservadora (λ≤0.1, LN affine, memreg que pina drift) es inerte en accuracy | `negativo-útil` | `notebooks/08_e7b_tta/out/e7b_results.csv`; `notebooks/09_e7c_codebook/out/e7c_results.csv`; `notebooks/10_memory_hippocampal/out/e10_results.csv`; `notebooks/10b_imagenet_c/out/e10_imagenet_c_results.csv` | Flowers, ImageNet-C | 2026-06-05 | causa diagnosticada: clasificador frozen + downstream; LN gradiente estructuralmente nulo (E7b); soft-mix se lava antes del logit (E10, ambos dominios) | wtq |
| C3 | retrieval no-paramétrico que vota en el logit (`logits_base + α·logits_cache`) SÍ mueve la predicción | `sostenido` | `notebooks/11_retrieval_memory/out/e11_results.csv`; `notebooks/13_imagenet_r/out/e13_imagenet_r.csv` | ImageNet-C, ImageNet-R | 2026-06-07 | reforzado con shift **natural** (ImageNet-R): el voto episódico mueve la predicción aun sin ruido sintético que quitar | wtq.1, wtq.6 |
| C4 | ese retrieval rompe el trade-off (sube clean Y corrupt simultáneo) | `parcial` | `notebooks/11_retrieval_memory/out/e11_results.csv` (variante `source_cache_z_pool_fixed_alpha`); `notebooks/12_flowers_retrieval/out/e11_retrieval_flowers.csv` (barrido α); `notebooks/13_imagenet_r/out/e13_imagenet_r.csv` (shift natural, barrido α) | ImageNet-C, Flowers, ImageNet-R | 2026-06-07 | (a) **CERRADA**: replicado en 3 dominios (ImageNet-C sintético, Flowers sintético-benigno, ImageNet-R **natural**); el retrieval sube clean Y shift sobre el source, coste calibración acotado a α≈1.0 (gate C8/wtq.8). El 3er dominio (R) es de naturaleza distinta → relaja la tensión "solo dominios benignos/sintéticos". (b) **abierta**: sigue siendo mejora relativa al sustrato congelado, no SOTA — FT-aug domina en Flowers; RN50 plano supera al sustrato en ImageNet-C y en R los absolutos son bajos (base VQSA débil). Reportar como mejora sobre el sustrato frozen | wtq.5, wtq.6 |
| C5 | la representación útil para recuperar vecinos es `z_pool` (continuo, pre-cuantización), no `zq_pool` | `sostenido` | `notebooks/11_retrieval_memory/out/e11_results.csv`; `notebooks/11_retrieval_memory/insights.md` (Insight 2) | ImageNet-C | 2026-06-05 | — | — |
| C6 | el codebook (`zq_pool`) como clave de memoria está aliasado: vota con fuerza pero sin precisión | `negativo-útil` | `notebooks/11_retrieval_memory/out/e11_results.csv` (variante `source_cache_zq_pool_fixed_alpha`) | ImageNet-C | 2026-06-05 | reframe del pivote: zq deja de ser "la memoria" y pasa a ablación negativa que justifica el uso de z_pool. Gate de unfamiliarity lo rescata parcialmente pero queda lejos de z_pool | — |
| C7 | memoria episódica online con pseudo-labels se contamina (rompe más de lo que repara) | `negativo-útil` | `notebooks/11_retrieval_memory/out/e11_results.csv` (variantes `episodic_cache_zq_pool`, `dual_cache_zq_pool`) | ImageNet-C | 2026-06-05 | lección para el claim biológico: plasticidad episódica sin filtro de verdad = auto-refuerzo, no memoria útil | — |
| C8 | la ganancia de retrieval tiene coste de calibración (ECE/NLL corrupto empeoran) | `parcial` | `notebooks/11_retrieval_memory/out/e11_results.csv` (cols `ece_corrupt_avg`, `nll_corrupt_avg`); `notebooks/12_flowers_retrieval/out/e11_retrieval_flowers.csv`; `notebooks/13_imagenet_r/out/e13_imagenet_r.csv` (cols `ece`, `nll`, `brier`) | ImageNet-C, Flowers, ImageNet-R | 2026-06-07 | DEUDA central del epic: el criterio de éxito exige acotar ECE/NLL, no accuracy a secas. Patrón confirmado en 3 dominios: ganancia con coste acotado a α≈1.0 (en R Brier incluso mejora, NLL ~plano hasta α=1.0). Candidatos: temperatura/escala separada de `logits_cache`, gate por margen/acuerdo | wtq.8 |

## Cobertura OOD (qué shift, dónde, estado del experimento)

Sin métricas — solo rol, eje de shift y estado de ejecución. El slate objetivo del pivote a
retrieval es Flowers + ImageNet (C y R) + CIFAR-C.

| dataset / backbone | rol en la tesis | tipo de shift | granularidad | severidad | estado | issue |
|--------------------|-----------------|---------------|--------------|-----------|--------|-------|
| Flowers-102 / RN18 | control de dominio + baselines del trade-off + ablaciones de codebook + 2º dominio para C4 — **NO evidencia de robustez fuerte** (dominio benigno: shift sintético sobre clases que el backbone ya separa) | sintético (gaussian_noise, pixel_mask, cutout, blur) | fino | grid 3 niveles | retrieval E11 portado (z_pool, barrido α) + FT-clean baseline hechos | wtq.5 |
| ImageNet-C / RN50 | escala | sintético (gaussian_noise, motion_blur, pixelate, jpeg) | medio | sev 3, 5 | E10 + E11 hechos | wtq.1 |
| ImageNet-R / RN50 | refutar "solo es denoising de ruido sintético" | **natural / semántico** (renditions) | medio | n/a | retrieval E11 portado (z_pool, máscara-200 Hendrycks, barrido α); el efecto SOBREVIVE al shift natural — sube clean Y R, coste calibración acotado a α≈1.0 → `notebooks/13_imagenet_r/out/e13_imagenet_r.csv` | wtq.6 |
| CIFAR-10-C / CIFAR-100-C | curva de severidad barata | sintético (15 corrupciones) | grueso / medio | **1–5** | PENDIENTE (se ordena después) | wtq.7 |

## Tensiones abiertas (resumen vivo)

- **Premisa del trade-off sin aislar (C1):** falta el baseline FT-clean en Flowers. Mientras
  no exista, la motivación "frozen rompe un trade-off" no está demostrada con los CSV en disco;
  lo que hay muestra FT-aug dominando a frozen.
- **Base ImageNet débil (C4):** el checkpoint `dememte_imagenet_resnet50_vqsa` rinde muy por
  debajo de ResNet50 plano en ImageNet-C. El positivo E11 es sobre esa base, no sobre SOTA —
  decidir si se reporta como "mejora relativa al sustrato congelado" o se fortalece la base.
- **Dominio del positivo (C3/C4):** ya en 3 dominios (ImageNet-C + Flowers + ImageNet-R, wtq.6).
  Falta solo la curva de severidad (CIFAR-C, wtq.7); la deuda que mantiene C4 en `parcial` es la
  (b) no-SOTA, no el dominio ni la naturaleza del shift.
- **Riesgo de robustez solo en dominios benignos (C4): RELAJADO (wtq.6).** El blindaje que
  cargaba ImageNet-R ya está: el efecto SOBREVIVE a un shift natural/semántico (renditions, sin
  ruido sintético que quitar) — sube clean Y R con coste de calibración acotado a α≈1.0
  (`notebooks/13_imagenet_r/out/e13_imagenet_r.csv`). La objeción "solo denoising de ruido
  sintético" ya no queda en pie. Queda el matiz honesto: absolutos bajos en R por la base RN50
  VQSA débil (deuda C4-b), el claim vive en el delta source→retrieval, no en el absoluto.
- **Calibración transversal (C8):** no es un paso final sino una condición que se re-mide en
  cada dominio del slate (wtq.8 atraviesa wtq.5/.6/.7). C4 no sube a `sostenido` sin ECE/NLL
  acotado, no solo accuracy.
- **Calibración (C8):** criterio de éxito del epic = ganancia robusta CON ECE/NLL acotado.
- **Pivote narrativo consumado:** zq→z_pool (C5/C6); el codebook es ahora ablación, no mecanismo.

## Cómo se actualiza

1. Cada experimento nuevo → fila(s) de claim + puntero a su CSV + (si aplica) issue beads + `verificado` con la fecha de hoy.
2. El estado cambia solo con evidencia en `out/`. `parcial` → `sostenido` exige 2º dominio o curva de severidad.
3. `verificado` se actualiza cada vez que se relee el CSV de la fila; si está viejo, releer antes de citar en el `inform`.
4. Antes de tocar el `inform` (Typst), este ledger debe estar al día — es el contrato.
