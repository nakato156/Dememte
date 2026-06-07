# E12 — Flowers retrieval + FT-clean baseline (wtq.5)

Dos outputs independientes, dos CSVs, que alimentan **C1** y **C4** por separado.
Números detallados aquí; el ledger (`THESIS.md`) solo cita las rutas de CSV.

Reproducir: `PYTHONPATH=src uv run python notebooks/12_flowers_retrieval/e12_flowers_retrieval.py`
(test set Flowers-102, `historical_trainval_resplit`, seed=42, `STRICT_SUITE`).

## Output 1 — FT-clean baseline → `out/ft_clean_baseline.csv` (C1)

`FT-clean` = full fine-tune de ResNet18 en imágenes **limpias** (`train_corrupt_prob=0.0`),
misma receta `train_baseline_phased` que el FT-aug de nb01/nb04 pero sin corrupción.

Tabla de baselines Flowers (clean / corrupt_avg), citando los CSVs de cada notebook:

| modelo | clean | corrupt | fuente |
|---|---:|---:|---|
| RN18 frozen linear-probe | 0.562 | 0.292 | `notebooks/01_baseline/out/baseline_summary.csv` |
| **RN18 FT-clean** | **0.916** | 0.453 | `out/ft_clean_baseline.csv` |
| RN18 FT-aug | 0.879 | 0.578 | `notebooks/01_baseline/out/baseline_summary.csv` |
| DeMemte VQSA frozen (E6 source) | 0.752 | 0.502 | `out/e11_retrieval_flowers.csv` (fila `source`) |

**Insight C1 (el trade-off existe y queda aislado).** FT-clean es el régimen de máxima
accuracy limpia (0.916, la más alta) y mínima robustez (0.453). Contra FT-aug, la
augmentation compra **+12.5pp de corrupt** (0.453→0.578) **a costa de −3.7pp de clean**
(0.916→0.879). Antes, nb01/nb04 solo tenían frozen-linear (débil en todo) y FT-aug, y FT-aug
parecía dominar — sin la pata FT-clean el trade-off no se veía. Ahora sí: **C1 deja de estar
`abierto`**.

## Output 2 — E11 retrieval en Flowers → `out/e11_retrieval_flowers.csv` (C4)

Checkpoint VQSA Flowers E6 (`e6_ema_kmeans_restart`, cargado `strict=True`), source cache de
1632 keys desde el train limpio, `key_space="z_pool"` (C5), `alpha_mode="fixed"`, source-only
(`episodic_size=0`). Barrido de α:

| variante | clean | corrupt | ece_clean | ece_corrupt |
|---|---:|---:|---:|---:|
| source (sin retrieval) | 0.752 | 0.502 | 0.058 | 0.091 |
| α=0.25 | 0.771 | 0.514 | 0.066 | 0.101 |
| α=0.5 | 0.784 | 0.522 | 0.074 | 0.112 |
| **α=1.0** | **0.797** | **0.533** | 0.089 | 0.133 |
| α=2.0 | 0.806 | 0.539 | 0.111 | 0.174 |
| α=4.0 | 0.809 | 0.537 | 0.137 | 0.243 |

**Insight C4 (rompe el trade-off en 2º dominio).** El retrieval sube clean **y** corrupt
simultáneamente sobre el source en todo el rango de α — el efecto que en ImageNet-C sostenía
C4, replicado en Flowers. clean crece monótono; corrupt pico en α≈2.0 (+3.7pp). Esto cierra la
deuda (a) de C4 (dominio único) — ahora 2 dominios.

**Insight C8 (coste de calibración).** ECE empeora monótono con α (corrupt 0.091→0.243). Hay
un régimen de ganancia con coste acotado: a **α=1.0**, clean +4.5pp / corrupt +3.1pp con
ece_corrupt 0.133 (vs 0.091 source). α≥2.0 ya paga calibración cara. El α elegido para reportar
C4 debe respetar el gate de C8/wtq.8 (ganancia CON ECE acotado) → α≈1.0.

**Matiz honesto (deuda C4-b, sin cerrar).** La ganancia es **relativa al sustrato frozen**, no
absoluta: FT-aug (0.879/0.578) sigue por encima de DeMemte+retrieval (0.797/0.533 @α=1.0) en
Flowers. Coherente con la tensión de dominio benigno ya anotada en el ledger — el peso de
robustez fuerte lo carga ImageNet-R (wtq.6), no Flowers.

## Estado de claims (propuesto; el cambio de estado lo valida el dueño del ledger)

- **C1**: `abierto` → `sostenido` (trade-off aislado en Flowers vía FT-clean vs FT-aug).
- **C4**: se mantiene `parcial` con deuda (a) cerrada (2º dominio) y (b) explícita (no SOTA);
  candidato a `sostenido` si el criterio acepta "mejora sobre sustrato frozen con ECE acotado".
- **Matriz OOD, fila Flowers**: estado → "retrieval E11 portado; FT-clean + barrido α hechos".
