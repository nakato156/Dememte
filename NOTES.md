# DeMemteAttractor Notes

Fecha de registro: 2026-05-07

## Contexto inicial

El experimento partio del issue de memoria atractora residual con gate de ambiguedad. La motivacion era corregir una limitacion de DeMemte actual: el gate basado solo en error de cuantizacion VQ (`dq_map`) actuaba como denoising general, pero no capturaba el objetivo neuro-inspirado de activar memoria solo cuando la entrada es ambigua y parcialmente familiar.

Se creo el notebook:

- `experiments/VQ/attractor_memory.ipynb`

El notebook implementa:

- `DeMemteAttractor`
- `LatentProjector` / `LatentUnprojector`
- `AttractorMemory`
- `VectorQuantizer2D` con `soft_assign`
- `AmbiguityGate`
- entrenamiento en tres fases
- variantes y ablaciones:
  - `DeMemte gate simple entropy`
  - `DeMemteAttractor`
  - `Ablation no OOD`
  - `Ablation no conflict`
  - `Ablation no anti-pareidolia`

## Primera ejecucion

Artifact:

- `experiments/VQ/out/artifacts/dememte_attractor_memory_20260506_111651/attractor_memory_results.csv`

Resultado principal: el gate colapso casi completamente a `1.0` en todas las variantes. Esto invalido el objetivo conceptual del gate, aunque algunas metricas de robustez mejoraron frente al baseline.

Metricas principales de la primera corrida:

| Modelo | clean_acc | corrupt_acc_avg | gate_clean | gate_blur | gate_cutout | gate_gauss_heavy |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 baseline | 0.664824 | 0.231975 | N/A | N/A | N/A | N/A |
| DeMemte Transformer current | 0.314197 | 0.171966 | N/A | N/A | N/A | N/A |
| Gate simple entropy | 0.757034 | 0.411300 | 0.999957 | 0.999956 | 0.999956 | 0.999955 |
| DeMemteAttractor | 0.749065 | 0.401488 | 0.999938 | 0.999914 | 0.999929 | 0.999987 |
| No OOD | 0.773784 | 0.418916 | 0.999978 | 0.999972 | 0.999976 | 0.999976 |
| No conflict | 0.768092 | 0.412736 | 0.999953 | 0.999929 | 0.999942 | 0.999987 |
| No anti-pareidolia | 0.725972 | 0.382176 | 0.999983 | 0.999972 | 0.999980 | 0.999993 |

Observaciones:

- El gate estaba saturado en limpio, blur, cutout y Gaussian heavy.
- `DeMemteAttractor` tenia `gate_mean_gauss_heavy = 0.999987`, mayor que blur/cutout.
- Esto significa que el modelo usaba memoria incluso en OOD extremo, que es exactamente lo que se queria evitar.
- El problema no parecia venir de una sola senal, porque todas las ablaciones colapsaron de forma similar.

Diagnostico inicial:

- El gate estaba demasiado libre para ignorar las senales.
- La CE empujaba a usar memoria siempre.
- La regularizacion previa era muy debil para competir contra el objetivo de clasificacion.
- El espacio latente podia contribuir, pero el fallo dominante parecia estar en la parametrizacion/objetivo del gate.

## Decision tomada

Se decidio no anadir todavia una perdida explicita sobre el valor del gate. En su lugar, se aplicaron tres cambios:

1. Usar weak-SIGReg como regularizacion del espacio latente.
2. Inicializar el ultimo bias del gate a un valor bajo.
3. Hacer que el gate dependa estructuralmente de las senales mediante un prior multiplicativo.

La idea era atacar el colapso sin introducir una supervision manual directa del gate.

## Cambios implementados

Se actualizo `experiments/VQ/attractor_memory.ipynb`.

Nuevos hiperparametros:

```python
weak_sigreg_weight: float = 0.01
weak_sigreg_sketch_dim: int = 64
gate_init_prob: float = 0.1
gate_prior_floor: float = 0.02
```

Se implemento `sigreg_weak_loss(x, sketch_dim=64)` localmente, inspirado en `weak-SIGReg` de `kreasof-ai/sigreg`:

- recibe tokens `(N, C)`
- si `C > sketch_dim`, aplica un sketch aleatorio
- centra los tokens
- calcula covarianza
- penaliza la norma Frobenius contra la identidad

Aplicacion de weak-SIGReg:

- Fase 1: sobre `z`
- Fase 2: sobre `z_completed` clean/dirty
- Fase 3: sobre `z` y `z_completed` clean/dirty

Cambios en `AmbiguityGate`:

```python
gate_raw = sigmoid(mlp(signals))
prior = uncertainty * familiarity * (1 - ood_risk)
gate = gate_prior_floor + (1 - gate_prior_floor) * prior * gate_raw
```

Para ablaciones, una senal desactivada se reemplaza por factor `1.0` en el prior para no apagar el gate por completo.

Tambien se agrego al debug:

- `z`
- `z_completed`
- `gate_raw`
- `gate_prior`

## Segunda ejecucion

Artifact:

- `experiments/VQ/out/artifacts/dememte_attractor_memory_20260507_104816/attractor_memory_results.csv`

Metricas principales de la segunda corrida:

| Modelo | clean_acc | corrupt_acc_avg | gate_clean | gate_blur | gate_cutout | gate_gauss_heavy |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 baseline | 0.739145 | 0.287730 | N/A | N/A | N/A | N/A |
| DeMemte Transformer current | 0.175476 | 0.114856 | N/A | N/A | N/A | N/A |
| Gate simple entropy | 0.773947 | 0.417588 | 0.995924 | 0.995270 | 0.995247 | 0.993042 |
| DeMemteAttractor | 0.748252 | 0.399889 | 0.893022 | 0.876692 | 0.900244 | 0.953632 |
| No OOD | 0.776549 | 0.419526 | 0.952701 | 0.947533 | 0.955794 | 0.951543 |
| No conflict | 0.771020 | 0.412804 | 0.914584 | 0.890281 | 0.915227 | 0.954961 |
| No anti-pareidolia | 0.737681 | 0.384453 | 0.930622 | 0.915561 | 0.930306 | 0.940928 |

Mejores modelos por metrica en la segunda corrida:

- Mejor `clean_acc`: `Ablation no OOD`, 0.776549.
- Mejor `corrupt_acc_avg`: `Ablation no OOD`, 0.419526.
- Mejor Gaussian noise: `Ablation no conflict`, 0.211579.
- Mejor pixel mask: `Ablation no OOD`, 0.126145.
- Mejor cutout: `Gate simple entropy`, 0.670624.
- Mejor blur: `Ablation no OOD`, 0.687429.

## Comparacion: antes vs despues

Cambios para `DeMemteAttractor`:

| Metrica | Antes | Despues | Delta |
|---|---:|---:|---:|
| clean_acc | 0.749065 | 0.748252 | -0.000813 |
| corrupt_acc_avg | 0.401488 | 0.399889 | -0.001599 |
| gaussian_noise | 0.201442 | 0.203881 | +0.002439 |
| pixel_mask | 0.130157 | 0.121537 | -0.008619 |
| cutout | 0.640809 | 0.640971 | +0.000163 |
| blur | 0.633545 | 0.633165 | -0.000379 |
| gate_clean | 0.999938 | 0.893022 | -0.106916 |
| gate_blur | 0.999914 | 0.876692 | -0.123221 |
| gate_cutout | 0.999929 | 0.900244 | -0.099685 |
| gate_gauss_heavy | 0.999987 | 0.953632 | -0.046355 |
| beneficial_changes | 0.399157 | 0.396596 | -0.002561 |
| harmful_changes | 0.006275 | 0.006099 | -0.000176 |
| pareidolia_rate | 0.006275 | 0.005922 | -0.000352 |
| gate_entropy | 0.000524 | 0.157468 | +0.156944 |

Interpretacion:

- El colapso exacto del gate mejoro.
- El gate del modelo full bajo de casi `1.0` a valores entre `0.87` y `0.95`.
- La entropia del gate subio mucho, lo que confirma que ya no esta totalmente saturado.
- La accuracy casi no cambio.
- La robustez promedio del modelo full bajo muy levemente.
- `harmful_changes` y `pareidolia_rate` mejoraron apenas.

## Problema que sigue abierto

Aunque el gate ya no esta clavado en `1.0`, sigue siendo demasiado alto y todavia falla el criterio mas importante:

- `gate_mean_gauss_heavy` deberia ser menor que blur/cutout recuperable.
- En la segunda corrida, `DeMemteAttractor` obtuvo:
  - `gate_mean_blur = 0.876692`
  - `gate_mean_cutout = 0.900244`
  - `gate_mean_gauss_heavy = 0.953632`

Esto indica que Gaussian heavy sigue activando mas memoria que corrupciones recuperables. El prior multiplicativo esta ayudando, pero las senales que alimentan el prior no estan calibradas como se esperaba.

En los logs de entrenamiento tambien se observo que `gate_raw` vuelve a saturarse cerca de `1.0`. El valor final baja porque `gate_prior` lo limita, no porque el MLP haya aprendido a cerrar el gate.

Ejemplo de logs para `DeMemteAttractor`:

- Fase 2 final: `val_gate=0.9071`, `val_prior=0.9058`, `val_raw=0.9988`.
- Fase 3 final: `val_gate=0.9208`, `val_prior=0.9192`, `val_raw=1.0000`.

Conclusion tecnica:

- weak-SIGReg + bias bajo + prior multiplicativo fue una mejora parcial.
- El siguiente foco no deberia ser subir accuracy todavia.
- El siguiente foco deberia ser calibrar `familiarity` y `ood_risk`, porque el prior aun considera OOD pesado como suficientemente familiar/usable.

## Estado actual

Archivos relevantes:

- Notebook principal: `experiments/VQ/attractor_memory.ipynb`
- Primera corrida: `experiments/VQ/out/artifacts/dememte_attractor_memory_20260506_111651/attractor_memory_results.csv`
- Segunda corrida: `experiments/VQ/out/artifacts/dememte_attractor_memory_20260507_104816/attractor_memory_results.csv`

Estado de la conclusion:

- Arquitectura y entrenamiento funcionan.
- La regularizacion nueva redujo el colapso exacto.
- El comportamiento neuro-inspirado aun no esta logrado.
- La senal OOD/familiaridad necesita calibracion adicional antes de considerar el issue resuelto.

## Verificacion posterior del notebook

Fecha de verificacion: 2026-05-08

Se reviso si `experiments/VQ/attractor_memory.ipynb` habia sido ejecutado despues de implementar el plan de diagnostico y nuevos experimentos E0-E5.

Resultado:

- No hay evidencia de una tercera corrida completa.
- El notebook fue modificado el 2026-05-08, pero los artefactos disponibles siguen siendo solo:
  - `experiments/VQ/out/artifacts/dememte_attractor_memory_20260506_111651/`
  - `experiments/VQ/out/artifacts/dememte_attractor_memory_20260507_104816/`
- El ultimo `attractor_memory_results.csv` sigue fechado el 2026-05-07 11:54:28.
- No existen artefactos nuevos esperados por el plan implementado:
  - `attractor_signal_curves.csv`
  - `e0_attractor_metrics.json`
  - `e0_attractor_signal_curves.csv`
- La celda E0 nueva del notebook aparece sin ejecucion ni outputs.
- El output viejo de configuracion no contiene los campos nuevos agregados, como `gate_raw_entropy_reg`, `gate_dropout`, `phase3_memory_grad_mode`, `run_e0_latest_checkpoint_diagnostic` o `experiment_names`.

Validacion del codigo:

- El notebook sigue siendo JSON valido.
- Todas las celdas de codigo parsean correctamente con `ast`.
- `git diff --check` no reporta problemas de whitespace.

Conclusion:

- La implementacion del plan quedo en el notebook, pero todavia no fue ejecutada.
- Las metricas que siguen siendo validas son las de la segunda corrida del 2026-05-07.
- No se debe interpretar ningun output visible del notebook como resultado de E0-E5, porque son outputs heredados de la corrida anterior.

Proximo paso recomendado:

- Ejecutar el notebook desde cero en el entorno Jupyter/CUDA.
- Primero revisar E0:
  - confirmar si Gaussian heavy tiene `dq_norm` bajo o `familiarity` alta;
  - confirmar si `ood_risk` no sube lo suficiente;
  - confirmar si `gate_raw` sigue saturando cerca de `1.0`.
- Luego correr el screening por defecto de `experiment_names`:
  - `attractor_full`
  - `e1_freeze_vq_phase3`
  - `e2_vq_clean_only_phase3`
  - `e3_gate_dropout_lr`
  - `e4_ood_tau_075`
  - `e4_ood_tau_100`
  - `e4_ood_tau_150`
- Comparar con Pareto estricto usando:
  - `clean_acc >= 0.748252`
  - `corrupt_acc_avg >= 0.399889`
  - `gate_order_margin >= 0.03`

## Tercera ejecucion: screening E0-E4

Fecha de revision: 2026-05-11

Artifact principal:

- `experiments/VQ/out/artifacts/dememte_attractor_memory_20260511_100006/`

Fechas de outputs:

- `phase1_shared_attractor.pt`: 2026-05-11 10:01:25
- `e0_attractor_signal_curves.csv`: 2026-05-11 10:08:33
- `e0_attractor_metrics.json`: 2026-05-11 10:08:33
- `attractor_full_best.pt`: 2026-05-11 10:11:31
- `e1_freeze_vq_phase3_best.pt`: 2026-05-11 10:14:26
- `e2_vq_clean_only_phase3_best.pt`: 2026-05-11 10:17:23
- `e3_gate_dropout_lr_best.pt`: 2026-05-11 10:20:21
- `e4_ood_tau_075_best.pt`: 2026-05-11 10:23:20
- `e4_ood_tau_100_best.pt`: 2026-05-11 10:26:18
- `e4_ood_tau_150_best.pt`: 2026-05-11 10:29:17
- `attractor_memory_results.csv`: 2026-05-11 11:00:46
- `attractor_signal_curves.csv`: 2026-05-11 11:00:46

Validacion del codigo:

- El notebook es JSON valido.
- Todas las celdas de codigo parsean correctamente con `ast`.
- `git diff --check` no reporta problemas de whitespace.
- Los outputs del notebook apuntan a `./out/artifacts/dememte_attractor_memory_20260511_100006/`, por lo que no son outputs heredados.
- Se detecto una salvedad: el campo `pareto_strict_success` del CSV generado solo evalua `clean_acc`, `corrupt_acc_avg` y `gate_order_margin`. No incluye todavia los umbrales completos del plan (`harmful_changes`, `pareidolia_rate`, `gate_raw_mean <= 0.95`). Se corrigio el notebook despues de esta revision para agregar `strict_acceptance_success` en futuras corridas, pero los CSV de esta ejecucion no fueron regenerados con ese campo.

### E0: diagnostico del ultimo checkpoint previo

E0 cargo:

- `./out/artifacts/dememte_attractor_memory_20260508_105450/attractor_full_best.pt`

Metricas E0:

| Metrica | Valor |
|---|---:|
| clean_acc | 0.776549 |
| corrupt_acc_avg | 0.422372 |
| gate_clean | 0.870174 |
| gate_blur | 0.854612 |
| gate_cutout | 0.885645 |
| gate_gauss_heavy | 0.925407 |
| gate_order_margin | -0.070795 |
| gate_raw_clean | 0.999378 |
| gate_raw_blur | 0.999303 |
| gate_raw_cutout | 0.999567 |
| harmful_changes | 0.005990 |
| pareidolia_rate | 0.005882 |

Conclusion E0:

- El checkpoint previo ya tenia buena accuracy, pero seguia fallando el criterio conceptual: Gaussian heavy abria mas gate que blur/cutout recuperable.
- `gate_raw` seguia saturado cerca de `1.0`.
- Esto confirmo que el problema no era solo falta de metricas: el mecanismo seguia dependiendo del prior.

### Resultados principales de la tercera corrida

| Modelo | clean_acc | corrupt_acc_avg | gate_clean | gate_blur | gate_cutout | gate_gauss_heavy | gate_order_margin | gate_raw_clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeMemteAttractor | 0.771833 | 0.420665 | 0.896802 | 0.893503 | 0.906759 | 0.922545 | -0.029042 | 0.999687 |
| E1 freeze VQ Phase 3 | 0.493088 | 0.217827 | 0.897020 | 0.874780 | 0.875253 | 0.912050 | -0.037271 | 0.999785 |
| E2 VQ clean-only Phase 3 | 0.767117 | 0.384805 | 0.969678 | 0.958673 | 0.974043 | 0.984140 | -0.025467 | 0.999989 |
| E3 gate dropout + low LR | 0.779476 | 0.403562 | 0.375518 | 0.383221 | 0.379919 | 0.304687 | 0.075232 | 0.497934 |
| E4 OOD tau 0.75 | 0.729550 | 0.408318 | 0.366969 | 0.389319 | 0.350380 | 0.236679 | 0.113701 | 0.998511 |
| E4 OOD tau 1.00 | 0.775899 | 0.400973 | 0.372658 | 0.397348 | 0.373922 | 0.337637 | 0.036285 | 0.997651 |
| E4 OOD tau 1.50 | 0.769068 | 0.422752 | 0.370953 | 0.397076 | 0.363899 | 0.122002 | 0.241896 | 0.998899 |

Comparacion contra la segunda corrida (`DeMemteAttractor`, 2026-05-07):

| Metrica | 2026-05-07 DeMemteAttractor | 2026-05-11 E4 tau 1.50 | Delta |
|---|---:|---:|---:|
| clean_acc | 0.748252 | 0.769068 | +0.020816 |
| corrupt_acc_avg | 0.399889 | 0.422752 | +0.022863 |
| gaussian_noise | 0.203881 | 0.226595 | +0.022714 |
| pixel_mask | 0.121537 | 0.151298 | +0.029761 |
| cutout | 0.640971 | 0.648181 | +0.007210 |
| blur | 0.633165 | 0.664932 | +0.031767 |
| gate_clean | 0.893022 | 0.370953 | -0.522069 |
| gate_blur | 0.876692 | 0.397076 | -0.479616 |
| gate_cutout | 0.900244 | 0.363899 | -0.536345 |
| gate_gauss_heavy | 0.953632 | 0.122002 | -0.831629 |
| harmful_changes | 0.006099 | 0.005868 | -0.000230 |
| pareidolia_rate | 0.005922 | 0.001138 | -0.004784 |

### Interpretacion por experimento

`DeMemteAttractor` nuevo:

- Mejoro mucho frente a la segunda corrida en accuracy: `clean_acc +0.0236` y `corrupt_acc_avg +0.0208`.
- Bajo `gate_gauss_heavy` de `0.953632` a `0.922545`, pero todavia falla el orden: `gate_order_margin = -0.029042`.
- `gate_raw` sigue saturado (`0.999687` clean), por lo que el MLP no aprendio a cerrar por si mismo.

E1 `freeze VQ Phase 3`:

- Falla fuerte en accuracy (`clean_acc = 0.493088`, `corrupt_acc_avg = 0.217827`).
- Congelar VQ/proyector/unproyector en Phase 3 no es viable tal como esta.
- La hipotesis de contaminacion del VQ no explica por si sola el problema; el modelo necesita adaptar esa ruta para clasificar.

E2 `VQ clean-only Phase 3`:

- Mantiene clean razonable (`0.767117`) pero baja robustez promedio (`0.384805`).
- Empeora el gate: `gate_clean = 0.969678`, `gate_gauss_heavy = 0.984140`.
- La rama dirty sin gradiente hacia VQ no corrige familiaridad/OOD; mas bien hace que el prior vuelva a abrir casi siempre.

E3 `dropout + low LR`:

- Es el unico que desatura realmente `gate_raw`: clean `0.497934`, blur `0.499129`, cutout `0.501084`.
- Cumple el orden conceptual: Gaussian heavy `0.304687` queda por debajo de blur/cutout, con `gate_order_margin = 0.075232`.
- Mantiene buen clean (`0.779476`) y supera el piso de corrupt promedio (`0.403562`).
- Pero `harmful_changes = 0.006126`, apenas por encima del umbral previo `0.006099`. Por criterio completo estricto, queda practicamente en frontera pero no aceptado.

E4 calibracion OOD/familiarity:

- Los tres E4 corrigen el orden del gate sin desaturar `gate_raw`; el cierre viene casi completamente del prior.
- `tau=0.75` cierra bastante el gate OOD, pero sacrifica clean (`0.729550`), por debajo del piso.
- `tau=1.00` cumple clean/corrupt/order, pero tiene `harmful_changes = 0.007210`, peor que el umbral.
- `tau=1.50` es el mejor balance global:
  - mejor `corrupt_acc_avg` de la corrida: `0.422752`;
  - `gate_gauss_heavy = 0.122002`;
  - `gate_order_margin = 0.241896`;
  - `harmful_changes = 0.005868`;
  - `pareidolia_rate = 0.001138`.
- Aun asi `gate_raw` sigue saturado (`0.998899` clean), asi que E4 arregla el comportamiento observable por el prior, no por decision aprendida del MLP.

### Insights tecnicos

1. La causa dominante confirmada es doble:
   - `gate_raw` tiende a saturar si no se le frena explicitamente.
   - La calibracion de `familiarity` es mas importante que `ood_risk` para Gaussian heavy.

2. `ood_risk` no esta detectando Gaussian heavy como OOD por el lado esperado. En varios modelos, Gaussian heavy tiene `dq_norm` negativo, por ejemplo:
   - `DeMemteAttractor`: Gaussian heavy `dq_norm_mean = -1.242656`, `ood_risk_mean = 0.000262`.
   - `E4 tau 1.50`: Gaussian heavy `dq_norm_mean = -1.419780`, `ood_risk_mean = 0.000366`.
   Esto significa que, con la formula actual `ood_risk = sigmoid(beta * (dq_norm - tau))`, Gaussian heavy no parece "alto dq"; parece mas cercano o incluso mas bajo que clean.

3. La senal que realmente cierra Gaussian heavy en E4 es `familiarity`, no `ood_risk`:
   - `E4 tau 1.50` clean: `familiarity_mean = 0.362352`.
   - `E4 tau 1.50` Gaussian heavy: `familiarity_mean = 0.105834`.
   - Como `gate_raw` sigue en `0.9995`, el gate final baja porque el prior baja.

4. Dropout + menor LR ataca el fallo que E4 no toca:
   - En E3, `gate_raw` queda alrededor de `0.5` en clean, blur, cutout y Gaussian heavy.
   - Esto demuestra que si faltaba una cosa simple, era exactamente anti-saturacion del MLP: dropout + `lr_gate` bajo + regularizacion de `gate_raw`.

5. El mejor candidato practico de esta corrida es `E4 OOD tau 1.50`, porque mejora accuracy y reduce pareidolia/harmful. El mejor candidato conceptual para el MLP es `E3`, porque es el unico que desatura `gate_raw`.

6. Ningun modelo cumple el criterio completo del plan simultaneamente:
   - E3 falla por muy poco `harmful_changes`.
   - E4 tau 1.50 falla `gate_raw_mean <= 0.95`.
   - Por eso el issue aun no esta completamente cerrado.

### Proximo paso recomendado

Hacer una cuarta corrida combinando lo mejor de E3 y E4:

- usar calibracion E4 con `ood_tau = 1.5`, `ood_beta = 8.0`, `familiarity_width = 0.5`, `phase3_lock_familiarity = True`;
- agregar anti-saturacion de E3: `gate_dropout = 0.1`, `lr_gate = 1e-4`, `gate_raw_entropy_reg = 0.01`;
- mantener `gate_prior_floor = 0.02`;
- evaluar con el nuevo campo `strict_acceptance_success`.

Hipotesis: esta combinacion deberia mantener el buen orden y robustez de E4 tau 1.50, pero con `gate_raw` no saturado como E3.


## Cambio posterior al analisis del 2026-05-11

Despues de revisar la tercera ejecucion se actualizo el notebook para preparar la cuarta corrida:

- Se agrego `e5_combined_dropout_ood_tau_150` a `experiment_names`.
- Esta variante combina lo mejor de E3 y E4:
  - `ood_tau = 1.5`
  - `ood_beta = 8.0`
  - `familiarity_width = 0.5`
  - `phase3_lock_familiarity = True`
  - `gate_dropout = 0.1`
  - `lr_gate = 1e-4`
  - `gate_raw_entropy_reg = 0.01`
- Se parametrizaron los umbrales de aceptacion completa en `Config`:
  - `acceptance_harmful_max = 0.006099`
  - `acceptance_pareidolia_max = 0.005922`
  - `acceptance_gate_raw_max = 0.95`
- `strict_acceptance_success` ahora usa esos umbrales configurables en futuras corridas.
