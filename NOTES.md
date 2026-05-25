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


## Actualizacion historica: refactor reproducible y evaluacion critica

Fecha de registro: 2026-05-25

Commits cubiertos:

- `687cc10` - `Enhance training configuration for DeMemte model phases and add legacy sanity checks`
- `7195625` - `Refactor notebook 01 to define two baselines for ResNet18 and streamline training/evaluation process`

Objetivo de esta actualizacion:

- Registrar los cambios hechos despues del analisis de E5 y de la tercera corrida.
- Dejar documentado como se reorganizo el flujo reproducible de notebooks.
- Registrar las metricas actuales que cambian la lectura historica del proyecto: E5 es fuerte contra un backbone congelado, pero no domina a ResNet18 con fine-tuning completo.

### Commit `687cc10`: entrenamiento critico y checks legacy

Archivo modificado:

- `src/dememte/training.py`

Cambios principales:

- `configure_phase1` y `configure_phase2` ahora fuerzan `model.backbone.eval()` ademas de congelar parametros. Esto evita que el backbone congelado quede en modo entrenamiento por efecto de `model.train(train)`.
- `configure_phase3` ahora recibe `train_backbone: bool = False`.
- En Phase 3, si `phase3_backbone_train_mode == "partial_unfreeze"`, solo `layer4` del backbone puede ponerse en modo train cuando corresponde a entrenamiento real.
- `run_epoch_phase3` llama `configure_phase3(model, config, train_backbone=train)`.
- `train_phase3` configura Phase 3 con `train_backbone=True` antes de construir el optimizador.
- Se agrego `_legacy_sanity_forward`:
  - toma un batch del train loader;
  - ejecuta forward sin gradiente;
  - valida shapes de logits, gate, prior y gate raw;
  - verifica finitud de SigReg;
  - imprime diagnostico de `denoise_loss`, `vq_loss` y `sigreg`.
- Se agrego `_legacy_phase2_gate_sanity`:
  - toma un batch corrupto;
  - mide `gate_mean`, `gate_prior` y `gate_raw`;
  - advierte si el gate cae fuera de `[0.1, 0.9]`.
- Se agrego `train_dememte_critical`, que reproduce el orden historico usado por el checkpoint E5 critico:
  1. Phase 1.
  2. Reset de calibracion del gate desde config.
  3. Forward sanity check legacy antes de Phase 2.
  4. Phase 2.
  5. Gate sanity check legacy despues de Phase 2.
  6. Phase 3.

Motivacion tecnica:

- Los checks legacy no son solo cosmeticos: consumen batches del DataLoader y avanzan la secuencia RNG.
- Para reproducir el camino historico de entrenamiento del checkpoint E5, esos forward checks deben permanecer dentro del driver de entrenamiento y no como celdas opcionales de notebook.
- La configuracion explicita de `eval()` en fases congeladas reduce drift por capas con comportamiento dependiente del modo train/eval.

Impacto esperado:

- Mayor reproducibilidad del protocolo E5 critico.
- Mejor separacion entre backbone congelado, partial unfreeze y entrenamiento completo.
- Menos ambiguedad entre "parametros congelados" y "modulo en modo eval".

### Commit `7195625`: refactor de notebooks y baseline fuerte

Archivos modificados:

- `notebooks/01_baseline/baseline.ipynb`
- `notebooks/02_e5_winner/e5_winner.ipynb`
- `notebooks/03_ablations/ablations.ipynb`
- `notebooks/04_finetune_vs_frozen/finetune_vs_frozen.ipynb`
- `scripts/build_notebooks.py`

Cambio de escala:

- El proyecto paso de un baseline unico congelado a una bateria de notebooks mas explicita:
  - Notebook 01: dos baselines ResNet18.
  - Notebook 02: carga/evaluacion de E5 critico.
  - Notebook 03: ablaciones del set critico.
  - Notebook 04: comparacion entre fine-tuning completo y E5 con backbone congelado.
- `scripts/build_notebooks.py` quedo como generador central de esa estructura.

### Notebook 01: dos baselines oficiales

Antes:

- Notebook 01 describia principalmente un `ResNet18` con backbone congelado como baseline 1:1.

Despues:

- Se definieron dos baselines oficiales:
  - `resnet18_frozen_linear_probe`
  - `resnet18_finetuned_aug`
- Se agrego una lista `BASELINES` con:
  - `id`
  - `label`
  - `freeze_backbone`
  - `train_corrupt_prob`
  - checkpoint legacy opcional.
- El loop ahora entrena/carga/evalua multiples baselines.
- Se agrego manejo de checkpoints por subdirectorio:
  - `notebooks/01_baseline/out/resnet18_frozen_linear_probe/best.pt`
  - `notebooks/01_baseline/out/resnet18_finetuned_aug/best.pt`
- Si existe checkpoint legacy del frozen baseline, se puede sembrar el nuevo path.
- Se consolidan outputs agregados:
  - `baseline_summary.csv`
  - `baseline_curves.csv`
  - `baselines_metrics.json`
- Se mantienen outputs historicos top-level para el frozen linear probe:
  - `metrics.json`
  - `predictions.csv`
  - `corrupt_curves.csv`
- Las curvas de robustez ahora comparan modelos por corrupcion en una grilla 2x2.

Metricas actuales del notebook 01:

| Modelo | clean_acc | corrupt_acc_avg | gaussian_noise | pixel_mask | cutout | blur | ECE clean | ECE corrupt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 frozen linear-probe | 0.562043 | 0.291904 | 0.144034 | 0.112593 | 0.463978 | 0.447010 | 0.465952 | 0.225121 |
| ResNet18 fine-tuned + data augmentation | 0.879167 | 0.578089 | 0.396162 | 0.306012 | 0.795360 | 0.814821 | 0.172045 | 0.167071 |

Interpretacion:

- El baseline fine-tuned completo es mucho mas fuerte que el baseline congelado.
- Esto cambia el marco de comparacion de E5: ya no basta con superar al frozen linear probe para afirmar robustez competitiva.

### Notebook 02: E5 critico reproducible

Cambios principales:

- El notebook 02 quedo centrado en `e5_combined_dropout_ood_tau_150`.
- Usa `train_dememte_critical` cuando `RUN_TRAINING=True`, preservando los sanity checks legacy del commit `687cc10`.
- Con `RUN_TRAINING=False`, carga el checkpoint legacy ganador tau 1.50 y lo copia a:
  - `notebooks/02_e5_winner/out/e5_best.pt`
- La evaluacion escribe:
  - `metrics.json`
  - `predictions.csv`
  - `signal_curves.csv`
  - `gate_plots/gate_and_robustness.png`

Metricas actuales del notebook 02:

| Metrica | Valor |
|---|---:|
| clean_acc | 0.811839 |
| corrupt_acc_avg | 0.521264 |
| gaussian_noise | 0.339567 |
| pixel_mask | 0.315715 |
| cutout | 0.704396 |
| blur | 0.725375 |
| gate_mean_clean | 0.234758 |
| pred_change_rate | 0.993508 |
| beneficial_changes | 0.517211 |
| harmful_changes | 0.004608 |
| pareidolia_rate | 0.000312 |
| ECE clean | 0.359240 |
| ECE corrupt avg | 0.261126 |

Curvas relevantes de E5:

| Condicion | Severidad | acc | gate | familiarity | ood_risk |
|---|---:|---:|---:|---:|---:|
| clean | 0.00 | 0.8118 | 0.2348 | 0.4399 | 0.0721 |
| gaussian_noise | 0.50 | 0.5531 | 0.2270 | 0.4254 | 0.0482 |
| gaussian_noise | 1.00 | 0.3381 | 0.1674 | 0.3054 | 0.0122 |
| gaussian_noise | 1.50 | 0.1275 | 0.1002 | 0.1686 | 0.0016 |
| blur | 0.35 | 0.7944 | 0.2297 | 0.4303 | 0.0907 |
| blur | 0.60 | 0.7440 | 0.2257 | 0.4219 | 0.0854 |
| blur | 0.85 | 0.6377 | 0.2193 | 0.4080 | 0.0548 |

Interpretacion:

- E5 mejora mucho sobre `ResNet18 frozen linear-probe`.
- E5 queda por debajo de `ResNet18 fine-tuned + data augmentation` en clean y corrupt promedio.
- El gate baja con ruido gaussiano severo y se mantiene mas abierto en blur, lo cual es compatible con la idea de intervencion selectiva.
- La senal `ood_risk` no sube con severidad gaussiana; en estos artefactos baja. Por tanto, el cierre observable del gate no debe atribuirse solo a OOD, sino tambien a familiaridad/prior.
- La tasa de pareidolia queda muy baja bajo la definicion operacional actual.

### Notebook 03: ablaciones del set critico

Cambios principales:

- Se organiza la evaluacion de 8 variantes:
  - `e5_combined_dropout_ood_tau_150`
  - `no_ood`
  - `no_familiarity`
  - `no_antipareidolia`
  - `freeze_vq_phase3`
  - `partial_unfreeze_backbone`
  - `attractor_disabled`
  - `resnet18_transfer_baseline`
- La variante E5 carga siempre:
  - `notebooks/02_e5_winner/out/e5_best.pt`
- Las demas variantes buscan primero checkpoint local y luego checkpoint legacy en:
  - `experiments/atracctor/out/artifacts/dememte_e5_critical/seed_42/`
- Se escriben:
  - `ablation_summary.csv`
  - `ablation_summary.md`
  - `ablation_curves.csv`
  - `plots/ablations_clean_vs_corrupt.png`

Resultados actuales de ablacion:

| Variante | clean_acc | corrupt_acc_avg | gaussian_noise | pixel_mask | cutout | blur |
|---|---:|---:|---:|---:|---:|---:|
| resnet18_transfer_baseline | 0.814441 | 0.548802 | 0.386730 | 0.313438 | 0.747059 | 0.747981 |
| no_antipareidolia | 0.821597 | 0.531943 | 0.353228 | 0.336369 | 0.711064 | 0.727110 |
| no_ood | 0.815742 | 0.530981 | 0.362173 | 0.330081 | 0.701957 | 0.729712 |
| e5_combined_dropout_ood_tau_150 | 0.811839 | 0.521264 | 0.339567 | 0.315715 | 0.704396 | 0.725375 |
| partial_unfreeze_backbone | 0.860790 | 0.518011 | 0.276956 | 0.220361 | 0.764298 | 0.810430 |
| no_familiarity | 0.797691 | 0.514094 | 0.335447 | 0.303247 | 0.704722 | 0.712961 |
| attractor_disabled | 0.802570 | 0.513403 | 0.330894 | 0.306391 | 0.698433 | 0.717895 |
| freeze_vq_phase3 | 0.488209 | 0.240961 | 0.126579 | 0.116821 | 0.328834 | 0.391608 |

Insights:

- `freeze_vq_phase3` confirma que la ruta VQ/proyector/desproyector necesita ajuste en Phase 3; congelarla destruye performance.
- `no_antipareidolia` y `no_ood` superan a E5 en corrupt promedio. Esto indica que OOD y anti-pareidolia se comportan como restricciones de seguridad/conservadurismo, no como maximizadores directos de accuracy.
- `attractor_disabled` queda cerca de E5 en corrupt promedio (`0.5134` vs `0.5213`), por lo que el atractor aporta pero no explica por si solo la robustez.
- `resnet18_transfer_baseline` queda como variante mas fuerte en corrupt promedio dentro del set de ablaciones.
- `partial_unfreeze_backbone` mejora clean y corrupciones recuperables como cutout/blur, pero empeora Gaussian y pixel mask.

Conclusion de ablaciones:

- El framework Dememte aporta mecanismos auditables, pero la evidencia no sostiene que todos sus componentes mejoren accuracy.
- La lectura correcta es trade-off: seguridad/interpretabilidad de intervencion versus rendimiento bruto.

### Notebook 04: fine-tuning completo vs E5 congelado

Cambios principales:

- Se define la condicion A:
  - `A_resnet_ft_corrupt`
  - ResNet18 con backbone completo entrenable.
  - `train_corrupt_prob = 0.85`.
  - Agenda mas agresiva: warmup 2, corrupt 8, joint 12.
- Se define la condicion B:
  - `B_dememte_e5_frozen`
  - carga E5 desde `notebooks/02_e5_winner/out/e5_best.pt` o checkpoint legacy.
  - no reentrena E5.
- Se escriben:
  - `comparison_table.csv`
  - `comparison_curves.csv`
  - `plots/robustness_curves.png`
  - `plots/scatter_tradeoff.png`

Resultados actuales del notebook 04:

| Condicion | clean_acc | corrupt_acc_avg | gaussian_noise | pixel_mask | cutout | blur | ECE clean | gap clean-corrupt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_resnet_ft_corrupt | 0.895593 | 0.683539 | 0.566542 | 0.484577 | 0.837155 | 0.845883 | 0.070239 | 0.212053 |
| B_dememte_e5_frozen | 0.765653 | 0.426018 | 0.217163 | 0.143817 | 0.667588 | 0.675503 | 0.392510 | 0.339635 |

Interpretacion:

- En esta comparacion, el fine-tuning completo con corrupciones agresivas domina claramente a E5.
- La tesis fuerte de que E5 rompe el trade-off clean/corrupt no queda respaldada por estos artefactos.
- E5 tiene un gap clean-corrupt mayor y peor calibracion que la ResNet18 fine-tuned corruptiva.
- El resultado obliga a presentar Dememte como framework auditable de memoria selectiva, no como baseline dominante de accuracy.

### Inconsistencia detectada entre notebooks 02 y 04

Se detecto una diferencia importante:

- Notebook 02 reporta E5:
  - `clean_acc = 0.811839`
  - `corrupt_acc_avg = 0.521264`
- Notebook 04 reporta `B_dememte_e5_frozen`:
  - `clean_acc = 0.765653`
  - `corrupt_acc_avg = 0.426018`

Posibles causas:

- Checkpoints no sincronizados entre notebook 02 y notebook 04.
- Ejecuciones en momentos distintos con artefactos heredados.
- Diferencias de carga entre `E5_CKPT` y `LEGACY_E5`.
- Estado de notebooks con outputs no regenerados en orden lineal.

Recomendacion:

- Reejecutar en orden:
  1. `notebooks/01_baseline/baseline.ipynb`
  2. `notebooks/02_e5_winner/e5_winner.ipynb`
  3. `notebooks/03_ablations/ablations.ipynb`
  4. `notebooks/04_finetune_vs_frozen/finetune_vs_frozen.ipynb`
- Registrar hashes de checkpoints usados por cada notebook.
- Confirmar si `notebooks/04_finetune_vs_frozen/out/comparison_table.csv` se genero despues de copiar/cargar exactamente el mismo `e5_best.pt` del notebook 02.

### Cambio de conclusion historica

Antes de estos commits, la narrativa principal era:

- E5 como variante ganadora del set critico.
- Exito definido por balance entre clean, corrupt, gate order, harmful changes, pareidolia y gate raw.

Despues de estos commits, la narrativa queda mas matizada:

- E5 sigue siendo fuerte frente a una ResNet18 congelada.
- E5 no supera a ResNet18 con fine-tuning completo y augmentacion.
- En ablaciones, quitar OOD o anti-pareidolia mejora accuracy bruta.
- El valor diferencial de Dememte queda en:
  - memoria latente explicita;
  - gate auditable;
  - medicion de intervencion;
  - metricas de cambios beneficiosos/daninos;
  - baja pareidolia bajo la definicion operacional actual.
- No queda demostrado que Dememte sea superior como metodo puro de accuracy robusta.

### Estado actual despues de los dos commits

Archivos relevantes:

- Entrenamiento critico:
  - `src/dememte/training.py`
- Generador de notebooks:
  - `scripts/build_notebooks.py`
- Notebooks activos:
  - `notebooks/01_baseline/baseline.ipynb`
  - `notebooks/02_e5_winner/e5_winner.ipynb`
  - `notebooks/03_ablations/ablations.ipynb`
  - `notebooks/04_finetune_vs_frozen/finetune_vs_frozen.ipynb`

Outputs relevantes:

- `notebooks/01_baseline/out/baseline_summary.csv`
- `notebooks/01_baseline/out/baseline_curves.csv`
- `notebooks/02_e5_winner/out/metrics.json`
- `notebooks/02_e5_winner/out/signal_curves.csv`
- `notebooks/03_ablations/out/ablation_summary.csv`
- `notebooks/03_ablations/out/ablation_curves.csv`
- `notebooks/04_finetune_vs_frozen/out/comparison_table.csv`
- `notebooks/04_finetune_vs_frozen/out/comparison_curves.csv`

Proximos pasos recomendados:

- Resolver la inconsistencia E5 notebook 02 vs notebook 04.
- Ejecutar multiples semillas para estimar varianza.
- Agregar hashes de checkpoints a las tablas de resultados.
- Separar explicitamente dos objetivos en futuros papers/reportes:
  - robustez predictiva pura;
  - seguridad/auditabilidad de intervencion memoristica.
- Si se quiere competir en accuracy, comparar contra fine-tuning completo como baseline principal, no solo contra frozen linear probe.
