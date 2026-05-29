# DeMemte Handoff

## Estado actual

DeMemte esta implementado como un modelo VQSA estricto para Flowers-102:

```text
image -> ResNet18 backbone -> 1x1 projector -> VQ codebook -> GAP(z/zq)
      -> self-attention fusion -> MLP classifier
```

La linea activa del repo ya no usa attractor/gate. Los experimentos relevantes son:

- `notebooks/06_e6_zq_alignment/`: variantes anti-colapso de cuantizador.
- `notebooks/07_e7_tta/`: primer intento de Test-Time Adaptation (E7 v1 — colapso).
- `notebooks/08_e7b_tta/`: TTA conservador con LayerNorm + preservacion de memoria
  latente (E7b — resultado negativo util, ejecutado).
- (pendiente) `notebooks/09_e7c_codebook/`: plasticidad del codebook (E7c — siguiente
  paso, ver roadmap abajo).

El checkpoint base usado para E7/E7b/E7c es:

```text
notebooks/06_e6_zq_alignment/out/e6_ema_kmeans_restart/best.pt
```

## Que se ha intentado

### E6: variantes de cuantizador

Se probaron:

- `e6_paper_faithful`
- `e6_zq_align_mse`
- `e6_ema_kmeans_restart`
- `e6_simvq_linear`
- `e6_fsq`

Resultados principales de `notebooks/06_e6_zq_alignment/out/e6_results.csv`:

| variante | clean_acc | corrupt_acc_avg | nota |
| --- | ---: | ---: | --- |
| `e6_fsq` | 0.7413 | 0.5039 | mejor corrupto por margen minimo, codebook lookup-free |
| `e6_ema_kmeans_restart` | 0.7523 | 0.5030 | mejor balance clean/corrupt y mejor uso real del codebook |
| `e6_paper_faithful` | 0.7361 | 0.4957 | colapso fuerte de uso de codebook |
| `e6_simvq_linear` | 0.7317 | 0.4807 | calibracion buena, robustez peor |
| `e6_zq_align_mse` | 0.7161 | 0.4739 | alineacion zq empeoro el resultado |

Decision actual: usar `e6_ema_kmeans_restart` como base principal porque conserva el
mejor `clean_acc`, empata practicamente en robustez corrupta y tiene mejor diagnostico
de codebook.

### E7 v1: TENT/EATA paper-style sobre BatchNorm affine

Se implemento:

- `src/dememte/tta.py`
  - `TentAdapter`, `EATALiteAdapter`, `softmax_entropy`
  - `collect_tta_bn_params`, `configure_tta_model`
- evaluacion TTA en `src/dememte/evaluation.py`
- notebook `notebooks/07_e7_tta/e7_tta.ipynb`
- tests en `tests/test_vqsa.py`

Metodos: `source`, `tent_bn`, `eata_lite_d005`, `eata_lite_d04` (EATA-lite sin Fisher).

Resultados (`notebooks/07_e7_tta/out/e7_results.csv`):

| metodo | clean_acc | corrupt_acc_avg | ece_corrupt_avg | nll_corrupt_avg |
| --- | ---: | ---: | ---: | ---: |
| `source` | 0.7523 | 0.5030 | 0.0903 | 2.0224 |
| `tent_bn` | 0.0319 | 0.0286 | 0.5090 | 8.9881 |
| `eata_lite_d005` | 0.0262 | 0.0220 | 0.5128 | 9.2272 |
| `eata_lite_d04` | 0.0311 | 0.0276 | 0.5359 | 9.3464 |

Conclusion: E7 v1 colapsa. El sanity check aislo la causa: forzar batch-stats de BN
(`track_running_stats=False`, `running_mean/var=None`) sobre `batch_size=16` con loader
ordenado destruye la representacion *antes del primer gradiente*. No es un problema de
TENT/EATA ni de `lr`.

```text
source normal:                   acc=0.8125
backbone BN con batch-stats:     acc=0.0
projector BN con batch-stats:    acc=0.0
backbone+projector batch-stats:  acc=0.0
```

### E7b: TTA conservador con LayerNorm + preservacion de memoria latente

Diseñado a partir de las respuestas fundamentadas de los autores de los papers de TTA
(ver `RESPONSES.md`), con tres correcciones explicitas al "E7b" especulativo del
handoff anterior:

- Q1/Q3 (SAR, [arXiv:2302.12400](https://arxiv.org/abs/2302.12400)): BatchNorm es el
  factor que desestabiliza TTA; usar normas agnosticas al batch (GroupNorm/LayerNorm).
- Q2 (CoTTA/EcoTTA): descontaminar el filtro EATA con logits de un teacher source
  congelado.
- Q4 (SoTTA, EcoTTA): el "BN Stats" no-update NO invalida el experimento — es un
  baseline a *reportar* junto a source, no una compuerta.
- Q5: el aporte de DeMemte es la preservacion de memoria latente (pattern completion),
  no la pura minimizacion de entropia.

Implementado en `src/dememte/tta.py`:

- `configure_tta_layernorm(model)` — congela todo salvo `nn.LayerNorm`; **nunca** toca
  `track_running_stats` ni `running_mean/var` (BN se queda en source stats).
- `collect_tta_ln_params(model)` — analogo a `collect_tta_bn_params`.
- `NoUpdateAdapter` — forward bajo `no_grad`, para baseline `bn_stats_no_update`.
- `latent_memory_loss(student_dbg, teacher_dbg, w_z, w_zq, w_assign)` —
  `MSE(z) + MSE(zq) + KL(soft_assign_src ‖ soft_assign)` contra teacher congelado.
  Inspirado en la auto-destilacion de [EcoTTA](https://arxiv.org/abs/2303.01904)
  (`‖x̃ₖ−xₖ‖₁` contra la red source congelada). Salta el KL si `soft_assign is None`
  (FSQ).
- `MemoryTentAdapter` — TENT-LN + memoria contra teacher source.
- `SourceFilterEATAAdapter` — filtro de fiabilidad/diversidad sobre **logits del
  teacher** (descontaminado); `memory_weights` opcional anade el regularizador.

Notebook: `notebooks/08_e7b_tta/e7b_tta.ipynb`. Tests nuevos en `tests/test_vqsa.py`
(total: 23 passed).

Variantes corridas y resultados (`notebooks/08_e7b_tta/out/e7b_results.csv`):

| variante | clean | corrupt avg | ECE corrupt | NLL corrupt | Δcorrupt vs source |
|---|---:|---:|---:|---:|---:|
| `source` | 0.7523 | 0.5030 | 0.0903 | 2.022 | — |
| `bn_stats_no_update` | 0.0317 | 0.0284 | 0.4896 | 8.740 | **−0.475 (colapso)** |
| `tent_ln` | 0.7520 | 0.5021 | 0.0948 | 2.033 | −0.0009 |
| `eata_ln` | 0.7517 | **0.5037** | 0.0912 | 2.021 | +0.0007 |
| `eata_ln_srcfilter` | 0.7520 | 0.5036 | 0.0913 | 2.021 | +0.0006 |
| `tent_ln_memreg` | 0.7520 | 0.5021 | 0.0948 | 2.033 | −0.0009 |
| `eata_ln_srcfilter_memreg` | 0.7520 | 0.5036 | 0.0913 | 2.021 | +0.0006 |

## Resultados de E7b — insights

Detalle completo con citas verificadas leyendo los papers en
`notebooks/08_e7b_tta/insights.md`. Sintesis:

1. **Gate tecnico pasado**: ninguna variante LayerNorm colapsa; `bn_stats_no_update`
   reproduce el colapso (0.032). Confirma SAR: el problema de E7 v1 era la superficie
   BatchNorm, no TENT/EATA. Adaptar LN manteniendo BN en running stats preserva el
   modelo antes del primer gradiente.
2. **Adaptacion inerte**: todas las variantes LN caen en ±0.003 de `source`.
   `tent_ln` empeora la calibracion (ECE 0.0948 vs 0.0903); `eata_ln` evita el dano
   pero no aporta ganancia. El affine de LayerNorm tiene leverage casi nulo sobre el
   clasificador con features congeladas.
3. **Hallazgo central — el regularizador de memoria es un no-op estructural**:
   `tent_ln` y `tent_ln_memreg` son **bit-a-bit identicos** (mismas 16 cifras),
   igual `eata_ln_srcfilter` vs su `_memreg`. Razon arquitectonica: en
   `VQSAFusion.forward`, `z`/`zq`/`soft_assign` se calculan **aguas arriba** de los
   bloques de self-attention; los LayerNorm tocan tokens *despues* del pooling y la
   VQ. Por lo tanto el gradiente de `latent_memory_loss` respecto a LN es **cero**.
   La memoria latente (codebook) es inalcanzable desde la unica superficie
   batch-agnostica.
4. **Corolario**: `hard_usage`/`dead_code_fraction` son identicos a source en todas
   las variantes LN. La memoria se "preserva" trivialmente porque no se puede tocar,
   no porque el regularizador haya hecho algo. **La tesis Q5 no se pudo testear con
   esta superficie de adaptacion**.

## Donde nos quedamos

E7b cierra limpio como **resultado negativo util**:

- TTA conservador (LayerNorm) es seguro pero inerte sobre este checkpoint.
- El regularizador de memoria esta estructuralmente desactivado: la memoria
  (codebook) vive aguas arriba de la unica superficie batch-agnostica del modelo.
- Para testear la tesis Q5 (pattern completion como plasticidad) hay que mover la
  superficie de adaptacion a un punto que **influya sobre `z`/`zq`**, que es
  justamente lo congelado/dependiente de BN.

Tests actuales:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest
# 23 passed (17 originales + 6 nuevos de E7b)
```

## Roadmap E7c (siguiente experimento)

### E7c-A (principal): plasticidad del codebook

Adaptar `model.vq.embedding.weight` (vq/ema_vq) o `model.vq.codebook_base`
(simvq_linear) en test-time, con `latent_memory_loss` como ancla anti-deriva. Es la
primera superficie aguas arriba que mueve `z`/`zq` sin tocar BN. El codebook como
parametro adaptable es la operacionalizacion directa de Q5 (plasticidad sinaptica).
FSQ queda fuera (codebook lookup-free).

Cambios incrementales sobre E7b:

- `configure_tta_codebook(model)` + `collect_tta_codebook_params(model)` en
  `src/dememte/tta.py`.
- Reusar `MemoryTentAdapter` y `SourceFilterEATAAdapter` tal cual: ahora **si muerden**
  porque `latent_memory_loss` tiene gradiente respecto al codebook (via `zq` y
  `soft_assign`).
- Notebook hermano `notebooks/09_e7c_codebook/e7c_codebook.ipynb` reusando el suite
  y los baselines `source`/`bn_stats_no_update`.

Variantes: `tent_codebook`, `eata_codebook`, `eata_codebook_srcfilter`,
`tent_codebook_memreg`, `eata_codebook_srcfilter_memreg`.

**Hito minimo de exito (antes de medir accuracy)**:

- `tent_codebook` colapsa el codebook (cae `hard_usage`, sube `dead_code_fraction`)
  → demuestra que el codebook es mojable desde TTA.
- `tent_codebook_memreg` mantiene el codebook cerca de source → demuestra que el
  regularizador finalmente muerde. Primer dato real sobre Q5.

Foco en `gaussian_noise` y `pixel_mask` (las dos corrupciones donde source rinde
peor, ~0.35): es ahi donde `z` deriva mas y donde la plasticidad podria ayudar.

### E7c-D (baseline barato, en paralelo): calibracion de stats del projector BN

Hook sobre `projector.net.1` (BatchNorm del projector) que reemplaza el forward por
`μ = α·running + (1−α)·batch`, `σ² = α·running + (1−α)·batch`, con α pequeño
(0.05–0.2). Sin gradiente. Familia [TTN (Lim et al., ICLR
2023)](https://arxiv.org/abs/2302.05155) / α-BN. Sirve para distinguir "ganancia por
mezclar stats" de "ganancia por plasticidad de codebook".

### E7c-B (escalar si A y D dan senal): adaptar conv 1x1 del projector

`projector.net.0` es `Conv2d(512, latent_dim, kernel_size=1)` (~131k params).
Adaptar sus pesos manteniendo `projector.net.1` (BN) congelada en running stats.
Combinar con `latent_memory_loss` como ancla.

### E7c-Z (plan B, solo si A/B/D no rinden): FOA caja negra

[FOA (Niu et al., ICML 2024)](https://arxiv.org/abs/2404.01650): prompt aditivo en
la entrada via CMA-ES, red entera congelada, sin retropropagacion. Elimina por
completo el problema de BN. Coste de implementacion mayor.

### Diagnosticos a instrumentar (deuda tecnica de E7b)

E7b no distingue "memoria preservada porque funciona" de "memoria preservada porque
es inalcanzable". Añadir columnas a `evaluate_dememte_tta` (todas baratas, ya hay
debug del teacher disponible):

- `z_drift` = `‖z − z_src‖`, `zq_drift` = `‖zq − zq_src‖`.
- `assignment_churn` = `mean(encoding_indices ≠ encoding_indices_src)`.
- `kl_assign_src` = `KL(soft_assign_src ‖ soft_assign)` registrado (no solo en la
  loss).

### Ablacion metodologica nueva: consolidacion vs episodico

`evaluate_dememte_tta_suite` resetea el adapter por cada `(corrupcion, severidad)`
(`adapter_factory()` fresco). Para la analogia de consolidacion neocortical de Q5
tiene sentido una ablacion **continual**: un solo stream que mezcla corrupciones,
sin reset, donde el codebook acumula. Cambio de 2 lineas en el suite. Vale en E7c-A.

## Lo que NO vale la pena (cerrar puertas)

- Mas variantes LN-only o sweeps de `lr`/`d_margin`: el techo ya esta confirmado.
- Reemplazo *post-hoc* de BN por GN/LN en el checkpoint: RESPONSES Q3 lo descarta
  explicitamente; SAR usa modelos pre-entrenados nativamente con GN/LN.
- Sweeps sobre `bn_stats_no_update`: es baseline reportado, no se va a salvar.

## Preguntas abiertas (RESPONSES.md, vigentes)

1. ¿Adaptar codebook/projector mantiene `hard_usage` cerca de source si el ancla
   de memoria muerde, o se necesita un termino explicito sobre el conteo?
2. ¿Conviene la ablacion continual sin reset (consolidacion) o el reset por
   condicion es la comparacion justa para reporte?
3. ¿La perdida de entropia debe pesar menos que `latent_memory_loss` cuando la
   superficie es el codebook (codebook es muy expresivo, puede colapsar mas
   rapido)?
4. ¿Tiene sentido un teacher con EMA del student (estilo CoTTA/Hybrid-TTA) en
   lugar del source crudo, para consolidacion lenta?
5. ¿FSQ (lookup-free) se comporta distinto bajo TTA porque no hay codebook
   aprendido? Vale como ablacion comparativa.
6. Para Flowers-102, el batch size 16 y el loader ordenado, ¿la comparacion
   online sigue siendo justa o conviene reportar tambien batch=64 / shuffled?

## Archivos clave

- `src/dememte/models/dememte.py`: arquitectura DeMemte VQSA.
- `src/dememte/models/vq.py`: cuantizadores VQ/EMA/SimVQ/FSQ.
- `src/dememte/evaluation.py`: evaluacion clean/corrupt y TTA.
- `src/dememte/tta.py`: adaptadores TENT/EATA (E7 v1) + E7b
  (`configure_tta_layernorm`, `collect_tta_ln_params`, `NoUpdateAdapter`,
  `latent_memory_loss`, `MemoryTentAdapter`, `SourceFilterEATAAdapter`).
- `src/dememte/__init__.py`: API publica con los exports de E7b.
- `tests/test_vqsa.py`: 23 tests (17 originales + 6 de E7b).
- `notebooks/06_e6_zq_alignment/out/e6_results.csv`: resultados E6.
- `notebooks/07_e7_tta/out/e7_results.csv`: resultados E7 v1 (colapso).
- `notebooks/08_e7b_tta/out/e7b_results.csv`: resultados E7b.
- `notebooks/08_e7b_tta/insights.md`: analisis E7b con bibliografia verificada
  (TENT, EATA, SAR, CoTTA, EcoTTA, SoTTA, FOA, Schneider, TTN, Hybrid-TTA,
  VQ-VAE, VQ-VAE-2, Jukebox, SimVQ, FSQ, Hendrycks, Flowers-102, etc.).
- `RESPONSES.md`: respuestas de los autores de los papers a las preguntas Q1–Q5
  que fundamentan E7b y E7c.

## Siguiente paso sugerido

Implementar **E7c-A (plasticidad del codebook)**:

1. Anadir `configure_tta_codebook` y `collect_tta_codebook_params` en `tta.py`.
2. Instrumentar los diagnosticos de deriva (`z_drift`, `assignment_churn`, etc.)
   en `evaluate_dememte_tta`.
3. Hand-write `notebooks/09_e7c_codebook/e7c_codebook.ipynb` espejando el de E7b.
4. Tests CPU mirroring los de E7b.

Criterio de exito del primer cut **no es accuracy** sino:

- `tent_codebook` colapsa el codebook (hard_usage baja) → la superficie es mojable.
- `tent_codebook_memreg` lo preserva cerca de source → el regularizador muerde.

Es el primer experimento donde la tesis de DeMemte (memoria como plasticidad
con preservacion) es realmente testeable.
