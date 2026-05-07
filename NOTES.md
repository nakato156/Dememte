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
