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
- `notebooks/09_e7c_codebook/`: plasticidad del codebook (E7c-A — ejecutado; primer
  dato real de Q5, ver abajo).
- `notebooks/10_memory_hippocampal/`: memoria asociativa biologica TTA-only
  (E10 — ejecutado; Hopfield + pattern completion + buffer episodico).

El checkpoint base usado para E7/E7b/E7c es
`notebooks/06_e6_zq_alignment/out/e6_ema_kmeans_restart/best.pt`. E7c-A usa ademas
la base SimVQ `e6_simvq_linear/best.pt` (codebook adaptable). E10 corre sobre ambas.

**Patron consolidado tras E7→E7c→E10**: cada mecanismo de "memoria" se activa
mecanicamente (drift > 0, completion > 0, buffer escribe) pero **es numericamente
inerte en accuracy** bajo hiperparametros conservadores. La tesis Q5 esta
mecanisticamente viva y nunca numericamente demostrada. El siguiente paso ya **no**
es un cuarto mecanismo, sino atacar la causa de la inercia (ver "Donde nos quedamos"
y el roadmap guiado por literatura abajo).

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

## E7c-A: plasticidad del codebook (ejecutado)

Detalle con citas en `notebooks/09_e7c_codebook/insights.md`. Base SimVQ
(`e6_simvq_linear`, codebook adaptable via `codebook_transform.weight`). Suite test
6149 imagenes, `lr=2.5e-4`, 1 step/batch, SGD momentum=0.9, `MEM_WEIGHTS=(1,1,1)`.

| variante | clean | corrupt | ECE | zq_drift | churn | nota |
|---|---:|---:|---:|---:|---:|---|
| `source` | 0.7317 | 0.4807 | 0.062 | — | — | — |
| `tent_codebook_softassign` | 0.7317 | 0.4808 | 0.062 | **0.0036** | 0.0021 | drift>0: codebook mojable |
| `tent_codebook_memreg` | 0.7317 | 0.4807 | 0.062 | **0.0000** | 0.0000 | memoria domina (drift exacto 0) |
| `eata_codebook_srcfilter_memreg` | 0.7317 | 0.4807 | 0.062 | 0.0000 | 0.0000 | idem |
| `codebook_loss_adapt` | 0.7318 | 0.4807 | 0.062 | 0.0005 | 0.0002 | ruta `codebook_loss` tambien muerde |
| `ttn_alpha_bn_095` | 0.7087 | 0.4622 | 0.073 | — | — | α-BN al 5% degrada (−2.3pp clean) |
| `bn_stats_no_update` | 0.0299 | 0.0257 | 0.407 | — | — | baseline colapso (reportado) |

Hitos: (1) **muro estructural de E7b superado** — el codebook SI es alcanzable cuando
la perdida toca rutas vivas (`soft_assign`/`codebook_loss`), no el straight-through
`q_st = z + (q-z).detach()`. `tent_codebook` puro queda excluido por inerte estructural
(test de regresion `test_tent_codebook_pure_is_structurally_inert`). (2) **Primer dato
cuantitativo de Q5**: el regularizador de memoria pina el drift a 0.0000 exacto — la
memoria es preservable bajo TTA. (3) **Pero accuracy inerte**: ±0.001 de source. Q5
operacionalmente viva, no numericamente demostrada.

## E10: memoria asociativa biologica TTA-only (ejecutado)

Detalle formal en `notebooks/10_memory_hippocampal/insights.md`. Tres mecanismos
sobre los checkpoints E6 sin reentrenar: recuperacion asociativa del codebook
(Hopfield moderno), pattern completion iterativa con gate de familiaridad, y doble via
codebook semantico + buffer episodico EMA. Integracion = **mezcla suave en `zq_pool`
con `λ_max ≤ 0.1`** (filosofia α-mix de TTN trasladada al espacio de tokens).

Phase 0 (gates duros, `out/e10_phase0.json`): familiaridad viable en ambas bases;
SimVQ restringido a `episodic_only` por `hard_usage=0.011 < 0.1` (codebook colapsado →
recuperacion semantica sin sentido); ambas pasan floor de clean acc.

Resultados base `e6_ema_kmeans_restart` (source clean 0.7523 / corrupt 0.5030):

| variante | clean | corrupt | Δcorrupt | nota |
|---|---:|---:|---:|---|
| `episodic_only` ★ | 0.7557 | **0.5051** | +0.0021 | unica net-positiva (dentro de ruido) |
| `consolidation_slow` | 0.7512 | 0.5035 | +0.0005 | |
| `hippocampal_full` | 0.7518 | 0.5031 | +0.0001 | |
| `assoc_recall_*` (const/fam/unfam) | ~0.750 | ~0.502 | −0.0004…−0.0009 | recall semantico regresiona |
| `completion_T3_best_gate` | 0.7504 | 0.5023 | −0.0006 | |

Insights clave: (1) **hitos mecanicos OK** (`completion_amount_corrupt`>0,
`episodic_buffer_churn`>0, trayectoria convergente) pero deltas en ±0.003 clean /
±0.002 corrupt → ruido. (2) **Aislamiento**: `assoc_recall_const` (solo softear el
argmin VQ, sin gate biologico) es la PEOR; añadir familiaridad/unfamiliaridad/T=3 no
mejora → el gate biologico no aporta sobre el soft-recall, y `λ_max=0.1` lo recorta.
(3) **Solo la via episodica ayuda** (marginalmente); el recall semantico perjudica.
(4) SimVQ episodico plano: la recuperacion asociativa exige codebook sano (refuerza el
headline E6).

## Donde nos quedamos

El patron es consistente a traves de E7b, E7c-A y E10: **el mecanismo se activa pero
no mueve accuracy**. Diagnostico de la causa (no del sintoma):

- El clasificador esta **congelado y downstream**; las perturbaciones pequeñas en
  `zq`/codebook se lavan antes del logit.
- Cada eleccion conservadora (LayerNorm inerte, `λ_max≤0.1`, memreg que pina drift a 0)
  **garantiza** la inercia para proteger calibracion. El proyecto elige conservador y
  luego se sorprende de no ver señal.
- La unica via con signo positivo (episodico en E10) sugiere **memoria no-parametrica /
  retrieval** antes que adaptacion parametrica diminuta.

Por eso el siguiente paso es una **busqueda de literatura dirigida** (usuario la hace
en NotebookLM, base de 48 papers) antes de implementar. Direcciones priorizadas en el
roadmap abajo.

Tests actuales:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest
# E7b: 23 passed; E7c añade tests de plasticidad/inercia estructural del codebook
```

## Roadmap (guiado por literatura — busqueda en curso por el usuario)

El usuario tiene una base de 48 papers en NotebookLM y va a extraer conocimiento con
preguntas dirigidas. Las direcciones, ordenadas por palanca real sobre la inercia:

### Prioridad 1 — Memoria no-parametrica / retrieval (el pivote)
La via episodica de E10 fue la unica net-positiva → convertir el "claim de memoria" en
un mecanismo de **retrieval que cambie la prediccion** (re-ranking del logit / cache
model tipo Tip-Adapter / kNN en test-time), no un soft-mix de `λ=0.1`. Ataca la causa:
hoy el efecto se lava antes del clasificador congelado. Riesgo de ingenieria medio,
maxima probabilidad de romper el patron de inercia.

### Prioridad 2 — Superficie de adaptacion que propague al logit
E7b mostro LN inerte (downstream de z/zq); E7c-A movio el codebook pero el efecto no
llega al logit congelado. Buscar superficies *upstream* con gradiente demostrable a la
salida (test-time prompt tuning, modulacion de features, test-time training con
auto-supervision). Formaliza la leccion "gradiente estructuralmente nulo".

### Prioridad 3 — λ derivado por muestra (experimento barato e inmediato)
Phase 0 de E10 muestra que la familiaridad SI discrimina (`g`: 0.50 clean → 0.39 bajo
corrupcion). Reemplazar el `λ_max=0.1` fijo por un λ **derivado por muestra** de esa
señal, midiendo explicitamente el trade-off ECE/NLL (calibracion bajo shift). Cambio
pequeño sobre el `HippocampalConfig` de E10.

### Prioridad 4 — Purificacion / proyeccion a manifold limpio
Replantear el codebook como *prior de manifold limpio* y **proyectar/denoise** la
feature corrupta (mas agresivo que mix 0.1), en vez de mezclar. Conecta directo con el
claim de "pattern completion". Mayor riesgo conceptual.

### Transversal — blindar el resultado negativo y verificar headroom
Si el patron sigue inerte, convertirlo en negative result bien diagnosticado
(failure modes de TTA, evaluacion rigurosa). Y verificar que Flowers-102 + la suite de
corrupcion no este **saturada** para el regimen frozen-backbone (regimen de shift mas
severo / gradual / online correlacionado donde exista margen medible).

### Deuda concreta pendiente
- Las preguntas para NotebookLM (10 principales) estan diseñadas; si se quiere, anclar
  respuesta↔direccion en un `literature_questions.md`.

## Lo que NO vale la pena (cerrar puertas)

- Mas variantes LN-only o sweeps de `lr`/`d_margin`: el techo ya esta confirmado (E7b).
- Mas mecanismos de memoria "biologica" en soft-mix con `λ` pequeño: E10 cierra que el
  gate biologico no aporta sobre softear el argmin, y `λ≤0.1` garantiza inercia.
- Reemplazo *post-hoc* de BN por GN/LN en el checkpoint: RESPONSES Q3 lo descarta;
  SAR usa modelos pre-entrenados nativamente con GN/LN.
- α-BN sobre el projector (E7c-D `ttn_alpha_bn_*`): degrada incluso al 5%; el projector
  BN esta perfectamente calibrado al source y no tolera mezcla. Cerrado.
- Sweeps sobre `bn_stats_no_update`: baseline de colapso reportado, no se salva.

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
- `notebooks/09_e7c_codebook/out/e7c_results.csv` + `insights.md`: resultados E7c-A
  (plasticidad del codebook, primer dato Q5).
- `notebooks/10_memory_hippocampal/out/{e10_results.csv,e10_phase0.json,e10_summary.md}`
  + `insights.md`: resultados E10 (memoria asociativa biologica TTA-only).
- `notebooks/08_e7b_tta/insights.md`: analisis E7b con bibliografia verificada
  (TENT, EATA, SAR, CoTTA, EcoTTA, SoTTA, FOA, Schneider, TTN, Hybrid-TTA,
  VQ-VAE, VQ-VAE-2, Jukebox, SimVQ, FSQ, Hendrycks, Flowers-102, etc.).
- `RESPONSES.md`: respuestas de los autores de los papers a las preguntas Q1–Q5
  que fundamentan E7b y E7c.

## Siguiente paso sugerido

1. **Esperar la busqueda de literatura** (NotebookLM, 48 papers) sobre las 4
   direcciones priorizadas; la apuesta principal es **retrieval no-parametrico**
   (Prioridad 1) porque ataca la causa de la inercia, no el sintoma.
2. **Experimento barato mientras tanto**: λ derivado por muestra (Prioridad 3) sobre
   el `HippocampalConfig` de E10 — usa la señal de familiaridad que Phase 0 ya valido,
   reportando el trade-off ECE/NLL.

El criterio de exito sigue **sin ser accuracy a secas**: romper el patron significa
mostrar un mecanismo cuyo efecto **llega al logit** con coste de calibracion acotado.
Hasta ahora E7c y E10 prueban que la memoria es *preservable/activable* pero no que
*mueva la prediccion*.
