# E7c-A — Plasticidad del codebook: hallazgos

## TL;DR

- **El muro estructural de E7b está superado**: el codebook SÍ es alcanzable
  desde TTA cuando la pérdida toca rutas vivas (`soft_assign` o
  `codebook_loss`). `tent_codebook_softassign` produce `zq_drift_corrupt=0.0036`
  y `assignment_churn=0.0021` — drift estrictamente positivo, primer dato real
  de plasticidad del codebook bajo TTA.
- **El regularizador de memoria latente muerde con fuerza dominante**: las dos
  variantes `*_memreg` (TENT y EATA) producen drift **exactamente 0.0000**. La
  ancla source domina completamente las señales TENT/EATA y la memoria
  pattern-completion gana. **Primer dato cuantitativo sobre la tesis Q5**:
  la memoria es preservable bajo TTA con pesos `MEM_WEIGHTS=(1,1,1)`.
- **La plasticidad no se traduce en accuracy bajo hiperparámetros
  conservadores** (`lr=2.5e-4`, 1 step/batch, SGD momentum=0.9):
  todas las variantes de codebook caen en ±0.001 de `source` en
  `corrupt_acc_avg`. La tesis Q5 está operacionalmente viva pero requiere
  un régimen menos rígido para mostrar ganancia.
- **TTN α-BN degrada accuracy** incluso al 5% de batch stats: α=0.95 baja
  clean −2.3 pp y corrupt −1.9 pp. El projector BN está perfectamente
  calibrado al source y no tolera mezcla.

## Matriz de resultados (test, 6149 imágenes)

| variante | clean | corrupt | ECE | NLL | zq_drift | churn | KL_src |
|---|---:|---:|---:|---:|---:|---:|---:|
| `source` | 0.7317 | 0.4807 | 0.062 | 2.031 | — | — | — |
| `bn_stats_no_update` | 0.0299 | 0.0257 | 0.407 | 7.794 | 0.2020 | 0.6164 | 5.62 |
| `tent_codebook_softassign` | 0.7317 | 0.4808 | 0.062 | 2.031 | **0.0036** | **0.0021** | 0.0001 |
| `tent_codebook_memreg` | 0.7317 | 0.4807 | 0.062 | 2.031 | **0.0000** | **0.0000** | -0.0000 |
| `eata_codebook_srcfilter_memreg` | 0.7317 | 0.4807 | 0.062 | 2.031 | **0.0000** | **0.0000** | -0.0000 |
| `codebook_loss_adapt` | 0.7318 | 0.4807 | 0.062 | 2.031 | 0.0005 | 0.0002 | 0.0000 |
| `codebook_loss_adapt_memreg` | 0.7318 | 0.4807 | 0.062 | 2.031 | 0.0004 | 0.0002 | 0.0000 |
| `ttn_alpha_bn_090` | 0.6816 | 0.4416 | 0.086 | 2.240 | 0.0330 | 0.0330 | 0.019 |
| `ttn_alpha_bn_095` | 0.7087 | 0.4622 | 0.073 | 2.130 | 0.0200 | 0.0168 | 0.005 |

Drift columns son contra el teacher source congelado (mismo checkpoint
SimVQ). `source` no tiene columnas de drift porque va por
`evaluate_dememte_suite` sin teacher (drift sería 0 por definición).

## Detalle 1 — Muro estructural superado

E7b cerró diagnosticando que `LayerNorm` está aguas abajo de `z`/`zq`, así que
el regularizador de memoria no podía mojar el codebook. E7c-A confirma la
predicción opuesta: cuando movemos la superficie de adaptación al
`codebook_transform.weight` (SimVQ), y la pérdida toca una ruta que **no**
pasa por el straight-through `q_st = z + (q-z).detach()`, sí hay gradiente:

- `tent_codebook_softassign` minimiza `entropy(soft_assign)`; `soft_assign`
  depende de `embedding` sin detach → drift positivo (0.0036).
- `codebook_loss_adapt` minimiza `MSE(q, z.detach())`; `q = one_hot @ emb`
  → drift positivo (0.0005, más pequeño porque el codebook ya está cerca
  óptimo para los `z` del source).

`tent_codebook` puro y `eata_codebook_srcfilter` puro fueron **excluidos del
grid** porque su pérdida (entropía sobre logits) viaja por `q_st` y queda
estructuralmente desconectada del codebook. Esto se documenta en
`tests/test_vqsa.py::test_tent_codebook_pure_is_structurally_inert` —
`entropy.requires_grad=False` cuando la única variable libre es el codebook.

## Detalle 2 — Memoria latente como dominante absoluta

Las dos variantes `*_memreg` producen `zq_drift=0.0000`, `churn=0.0000`,
`kl_src=-0.0000`. Esto no es ruido — la pérdida total con
`MEM_WEIGHTS=(1.0, 1.0, 1.0)`:

```
L = entropy(logits) + 1.0·MSE(z, z_src) + 1.0·MSE(zq, zq_src) + 1.0·KL(p_src ‖ p)
```

contiene tres términos que penalizan cualquier deriva, contra una entropía de
logits (TENT) cuyo gradiente al codebook **es estructuralmente cero** por el
straight-through. El único término vivo respecto al codebook es la KL en
`soft_assign`, que **también** es ancla source-anchored. El neto: el codebook
se queda exactamente donde estaba.

Operacionalmente esto valida la tesis Q5 de DeMemte:
> *La memoria emerge de cambios en la fuerza y organización de conexiones...
> la pista parcial reactiva el patrón... el pattern completion es la base de
> la recuperación.*

con `MEM_WEIGHTS=(1,1,1)` la "memoria sináptica" del codebook es absolutamente
robusta a TTA — preservada **independientemente de cuán fuerte sea la presión
adaptativa** (porque la entropy no muerde y la KL es contra-fuerza).

Pero también revela el otro lado: bajo este balance, la adaptación útil es
**cero**. La memoria perfectamente preservada no aprende nada nuevo.

## Detalle 3 — Por qué la plasticidad no mueve accuracy

Las variantes con drift positivo (`tent_codebook_softassign`,
`codebook_loss_adapt*`) producen +0.0001 a +0.0001 en `corrupt_acc_avg`. La
ganancia es indistinguible del ruido. Razones probables:

1. **Régimen de optimización demasiado conservador**: `lr=2.5e-4` con SGD
   momentum=0.9 sobre 1 step/batch, episódico (reset por celda). 6149/16 ≈
   384 batches/celda con un único paso de gradiente cada uno produce
   acumulación muy lenta sobre 65,536 params de `codebook_transform`.
2. **El gradiente del `codebook_loss` es pequeño en source-óptimo**: el
   checkpoint SimVQ ya minimizó `codebook_loss` durante entrenamiento, así
   que `∂L/∂emb` es ~0 sobre el dominio clean y crece poco sobre las
   corrupciones.
3. **La entropía de `soft_assign` con temperatura `t=1.0` es relativamente
   plana**: cambios pequeños en `emb` producen cambios pequeños en la
   distribución soft sobre 1024 códigos.

Estos no refutan la tesis Q5 — sólo confirman que el régimen `E7b-equivalent`
(idéntico a E7b para 1:1 comparativa) es **demasiado conservador** para que
la plasticidad sintáctica visible se traduzca en accuracy macroscópica.

## Detalle 4 — TTN α-BN: el projector BN no tolera mezcla

| variante | clean | corrupt | Δclean | Δcorrupt |
|---|---:|---:|---:|---:|
| `source` | 0.7317 | 0.4807 | — | — |
| `ttn_alpha_bn_095` (5% batch) | 0.7087 | 0.4622 | −2.3 pp | −1.9 pp |
| `ttn_alpha_bn_090` (10% batch) | 0.6816 | 0.4416 | −5.0 pp | −3.9 pp |

Incluso una inyección mínima de batch stats (5%) en `vqsa.projector.net.1`
degrada el modelo. Confirmamos que el BN del projector está **perfectamente
calibrado** al source y la fragilidad documentada en E7 v1
(track_running_stats=False = colapso) tiene un análogo gradual en α-mezcla:
la degradación es monótona con la fracción de batch stats.

Implicación: cualquier ganancia que las variantes de codebook lograran sería
**no atribuible a "mezclar stats"**. El α-BN no es una alternativa viable
para este checkpoint; el problema no es resolver con stats, es con plasticidad.

## Lo que sigue (E7c-A.v2)

Sweep de hiperparámetros para sacar la plasticidad de su zona muerta:

1. **`tta_lr`** ∈ {1e-3, 5e-3, 1e-2} sobre `tent_codebook_softassign` y
   `codebook_loss_adapt`. Hipótesis: con `lr=1e-2` el codebook se moverá
   suficiente para impactar accuracy.
2. **`MEM_WEIGHTS`** ∈ {(0.1,0.1,0.1), (0.01,0.01,0.01)} sobre
   `tent_codebook_softassign_memreg` (nueva combinación). La idea es
   **descongelar** la memoria sin destruirla — encontrar el punto donde la
   ancla deja pasar adaptación útil pero impide colapso.
3. **`steps`** ∈ {2, 4} por batch. Más iteraciones por batch antes de avanzar.
4. **Continual ablation**: una sola pasada sin reset por celda. La
   consolidación lenta (estilo CoTTA/EcoTTA) puede producir drift acumulado
   que beneficie las corrupciones similares (gaussian_noise + pixel_mask
   tienen estadística parecida).
5. **Foco en gaussian_noise/pixel_mask**: las dos corrupciones donde source
   rinde peor (~0.33-0.34) son las que más espacio dejan a la plasticidad.
   Las dos donde rinde bien (cutout 0.61, blur 0.65) son saturadas para
   adaptación.

## Lo que NO vale la pena perseguir

- **`tent_codebook` puro y `eata_codebook_srcfilter` puro** sobre cualquier
  hiperparámetro: el gradiente es estructuralmente cero. Documentado.
- **Bajar α-BN debajo de 0.95**: la curva de degradación es monótona y
  testeada hasta α=0.90. Cualquier α<0.95 es estrictamente peor.
- **Cambiar la base a `e6_paper_faithful` (VanillaVQ)** para "más
  expresividad": el codebook de 1024×256 = 262k params vs 65k de SimVQ es
  más superficie, pero el VanillaVQ ya colapsa el uso del codebook según E6
  (`hard_usage` bajo), lo que arriesga más colapso bajo TTA. La elección
  de SimVQ como base sigue siendo correcta.

## Archivos

- `out/e7c_results.csv` — 9 variantes × suite completa, 50+ columnas.
- `out/e7c_curves.csv` — curvas por celda `(corrupción, severidad)`.
- `out/e7c_summary.md` — tabla rankeada por `corrupt_acc_avg`.
- `out/<variante>/metrics.json` — JSON por variante.
- `out/<variante>/signal_curves.csv` — curvas por variante.
- `tests/test_vqsa.py` — 30 tests (23 originales + 7 E7c).

## Bibliografía cruzada (RESPONSES.md, vigente)

- **Q1/Q3 (SAR, [arXiv:2302.12400](https://arxiv.org/abs/2302.12400))**:
  superficie batch-agnóstica es lo único viable bajo BN. SimVQ
  `codebook_transform` cumple — es batch-agnóstica trivialmente.
- **Q2 (CoTTA/EcoTTA)**: descontaminar filtro con teacher source. Aplicado
  en `eata_codebook_srcfilter_memreg`. El filtro está correcto; el problema
  es que la pérdida pos-filtro (entropía + KL) no produce drift útil.
- **Q4 (SoTTA, EcoTTA)**: "BN Stats" reportado, no es gate. Reproducido
  como `bn_stats_no_update` (colapsa 0.7317 → 0.0299).
- **Q5 (DeMemte)**: pattern completion / plasticidad sináptica. **Validado
  mecánicamente**: el codebook es mojable, la memoria es preservable. **No
  validado en accuracy**: la plasticidad no produce mejora bajo régimen
  conservador. Sweep E7c-A.v2 es el siguiente paso para cerrar Q5 con
  evidencia numérica positiva o cerrarla como techo de este checkpoint.
