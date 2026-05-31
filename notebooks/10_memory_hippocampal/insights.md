# E10 - Memoria asociativa biologica TTA-only: hallazgos ImageNet-C

## TL;DR

- **E10 funciona mecanicamente en ImageNet-C**: el checkpoint DeMemte ImageNet
  carga correctamente, el codebook esta vivo (`hard_usage=0.722`) y las variantes
  de memoria producen `completion_amount_corrupt > 0` sin reentrenamiento.
- **Pero el efecto en accuracy es practicamente neutro**. La mejor variante,
  `assoc_recall_const`, solo mejora `corrupt_acc_avg` de `0.217402` a `0.217463`
  (`+0.000060`). Eso es demasiado pequeno para leerlo como ganancia robusta.
- **Clean accuracy no se rompe**. Varias variantes suben clean de `0.614` a
  `0.616-0.617`, y todas pasan el clean floor. La interfaz es segura, pero no
  suficientemente efectiva.
- **La memoria se mueve, pero no llega al logit con fuerza util**:
  `completion_T3_best_gate` alcanza `completion_amount_corrupt_avg=0.0700`,
  `hippocampal_full` activa buffer episodico (`churn=0.1934`), y aun asi no mejora
  corrupciones.
- **La familiaridad es viable pero poco discriminativa**: `g_clean=0.500`,
  `g_gaussian_noise_s3=0.463`, `g_pixelate_s3=0.465`. El gate esta vivo, pero no
  separa clean/corrupt con suficiente margen.

## Setup

E10 prueba tres mecanismos biologicamente inspirados sobre un checkpoint DeMemte
ImageNet ya entrenado, sin actualizar parametros:

1. **Recall asociativo** del codebook como Modern Hopfield:
   `softmax(-||z - E||^2 / tau) E`.
2. **Pattern completion** iterativo sobre `z_pool`, con gate de familiaridad o
   unfamiliaridad.
3. **Doble via CLS**: codebook semantico mas buffer episodico EMA.

La integracion sigue siendo deliberadamente conservadora: mezcla suave en
`zq_pool` con `lambda_max = 0.1`, para no sacar de distribucion al bloque de
self-attention y al clasificador congelado.

Resultados actuales en:

- `notebooks/10_memory_hippocampal/out/e10_results.csv`
- `notebooks/10_memory_hippocampal/out/e10_curves.csv`
- `notebooks/10_memory_hippocampal/out/e10_phase0.json`
- `notebooks/10_memory_hippocampal/out/e10_summary.md`

Datos y modelo:

- Clean: `experiments/data/imagenet-clean-5k`, `val_size=1000`.
- Corrupt: `experiments/data/imagenet-c-subset`.
- Condiciones: `gaussian_noise`, `motion_blur`, `pixelate`,
  `jpeg_compression`, severidades `3` y `5`.
- Cada condicion ImageNet-C evaluada con `50000` imagenes.
- Checkpoint:
  `experiments/imagenet_dememte/out/dememte_imagenet_resnet50_vqsa_best.pt`.

## Phase 0

| base | check | resultado | decision |
|---|---:|---:|---|
| `dememte_imagenet_resnet50_vqsa` | median `g_clean` | 0.500 | calibrado |
| `dememte_imagenet_resnet50_vqsa` | median `g_gaussian_noise_s3` | 0.463 | familiarity viable |
| `dememte_imagenet_resnet50_vqsa` | median `g_pixelate_s3` | 0.465 | familiarity viable |
| `dememte_imagenet_resnet50_vqsa` | `hard_usage` | 0.722 | all variants |
| `dememte_imagenet_resnet50_vqsa` | clean floor con `assoc_recall_const` | +0.0020 | pasa |

Phase 0 deja dos lecturas. Primero, el codebook ImageNet esta sano: usa alrededor
del 72% de las entradas en clean, asi que no hay indicio de colapso. Segundo, el
gate de familiaridad detecta algo de shift, pero la caida clean -> corrupt es
pequena (`0.500` a `0.463-0.465`), no una separacion fuerte.

## Matriz principal

Base `dememte_imagenet_resnet50_vqsa`, dataset `imagenet_c`.
Source: clean `0.6140`, corrupt `0.2174`.

| variante | clean | corrupt avg | Delta clean | Delta corrupt | ECE corrupt | NLL corrupt | completion | epi churn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `assoc_recall_const` | 0.6160 | **0.2175** | +0.0020 | **+0.0001** | 0.1478 | 5.1223 | 0.0523 | 0.0000 |
| `assoc_recall_unfamiliarity` | 0.6160 | 0.2174 | +0.0020 | +0.0000 | 0.1473 | 5.1166 | 0.0294 | 0.0000 |
| `completion_T3_best_gate` | **0.6170** | 0.2174 | **+0.0030** | +0.0000 | 0.1473 | 5.1161 | **0.0700** | 0.0000 |
| `assoc_recall_familiarity` | 0.6160 | 0.2174 | +0.0020 | +0.0000 | 0.1473 | 5.1153 | 0.0230 | 0.0000 |
| `source` | 0.6140 | 0.2174 | 0.0000 | 0.0000 | 0.1468 | 5.1102 | - | - |
| `hippocampal_full` | **0.6170** | 0.2173 | **+0.0030** | -0.0001 | **0.1466** | 5.1125 | 0.0615 | 0.1934 |
| `consolidation_slow` | **0.6170** | 0.2173 | **+0.0030** | -0.0001 | **0.1466** | 5.1124 | 0.0616 | 0.2037 |
| `episodic_only` | 0.6140 | 0.2171 | 0.0000 | -0.0003 | **0.1449** | **5.1083** | 0.0462 | 0.2009 |

Por corrupcion, la robustez base es:

| corrupcion | source acc | mejor variante | mejor acc | lectura |
|---|---:|---|---:|---|
| `gaussian_noise` | 0.1453 | `assoc_recall_const` | 0.1455 | mejora minima |
| `motion_blur` | **0.1327** | `source` | **0.1327** | E10 no ayuda |
| `pixelate` | **0.2408** | `source` / empates | **0.2408** | empate numerico |
| `jpeg_compression` | 0.3508 | `assoc_recall_const` | 0.3510 | mejora minima |

## Insight 1 - El mecanismo se activa, pero no llega con fuerza al logit

E10 no falla por estar desconectado. Las senales internas son positivas:

- `completion_amount_corrupt_avg`: de `0.0230` a `0.0700` segun variante.
- `recall_sharpness_corrupt_avg`: alrededor de `0.487-0.530`, no uniforme.
- `episodic_buffer_churn_corrupt_avg`: `0.1934` en `hippocampal_full`, `0.2037`
  en `consolidation_slow`, `0.2009` en `episodic_only`.
- `traj_max_step_corrupt_avg`: finito, de `0.1242` a `0.3251`, sin explosion del
  loop.

La lectura es directa: la memoria altera el espacio latente, pero esa alteracion
no cambia la decision de clase de forma consistente. El cuello de botella sigue
siendo la interfaz: una mezcla pequena en `zq_pool` antes de un clasificador
congelado.

## Insight 2 - El codebook semantico no corrige robustez, solo mueve poco

`assoc_recall_const` aisla el mecanismo minimo: recall semantico suave desde el
codebook, sin gate biologico ni buffer episodico. Es la mejor variante por
`corrupt_acc_avg`, pero la mejora es casi nula:

```text
source.corrupt_acc_avg             = 0.217402
assoc_recall_const.corrupt_acc_avg = 0.217463  (+0.000060)
```

La buena noticia es que el recall semantico no rompe el modelo. La mala noticia
es que tampoco corrige ImageNet-C. En la practica, softear el codebook produce
movimiento latente medible (`completion=0.0523`) pero no una recuperacion
semantica fuerte.

## Insight 3 - El gate biologico esta vivo, pero discrimina poco el shift

Phase 0 calibra `sigma` para que `median(g_clean) ~= 0.5`. Bajo corrupcion real,
la mediana baja solo a:

```text
g_clean             = 0.500
g_gaussian_noise_s3 = 0.463
g_pixelate_s3       = 0.465
```

Eso explica por que familiarity/unfamiliarity no dominan. La senal existe, pero
su margen es pequeno. En el ranking:

```text
assoc_recall_familiarity.delta_corrupt   = +0.000015
assoc_recall_unfamiliarity.delta_corrupt = +0.000038
completion_T3_best_gate.delta_corrupt    = +0.000025
```

El gate es util como diagnostico, pero no como multiplicador suave de
`lambda_max=0.1` en este regimen.

## Insight 4 - La via episodica escribe, pero no aporta accuracy

`episodic_only` activa claramente el buffer:

```text
episodic_only.epi_churn_corrupt = 0.2009
completion_amount_corrupt       = 0.0462
recall_sharpness_corrupt        = 0.5299
```

Pero queda por debajo de source en corrupcion:

```text
source.corrupt_acc_avg        = 0.217402
episodic_only.corrupt_acc_avg = 0.217107  (-0.000295)
```

La via episodica mejora ligeramente ECE/NLL corrupto (`ECE=0.1449`,
`NLL=5.1083`, ambos mejores que source), pero no acierta mas clases. Esto sugiere
un efecto de calibracion/confianza mas que de correccion de prediccion.

## Insight 5 - Mas biologico no significa mas efectivo bajo esta interfaz

`hippocampal_full` y `consolidation_slow` combinan recall semantico, buffer
episodico, completion T=3 y gate biologico. Son las variantes mas completas de la
tesis biologica, pero no las mas efectivas:

```text
hippocampal_full.corrupt_acc_avg   = 0.217310  (-0.000092)
consolidation_slow.corrupt_acc_avg = 0.217307  (-0.000095)
```

Ambas mejoran clean a `0.617`, y mejoran ligeramente ECE corrupto frente a
source, pero pierden accuracy corrupta. La combinacion mueve mas piezas internas,
pero no direcciona mejor el logit.

## Cierre

E10 ImageNet-C cierra una lectura mas precisa de la tesis de memoria:

- El checkpoint ImageNet esta bien acoplado al pipeline E10.
- El codebook esta sano y disponible para recall asociativo.
- El buffer episodico escribe y las trayectorias de completion son estables.
- La accuracy corrupta permanece esencialmente igual a source.

La conclusion no es que la memoria este muerta. Es mas estrecha: **la memoria
esta viva como mecanismo interno, pero no cambia la prediccion cuando se inyecta
solo como soft-mix pequeno antes de un clasificador congelado**.

## Proximo paso

No conviene agregar otro mecanismo biologico con la misma interfaz. El siguiente
experimento debe atacar el cuello de botella:

1. **Barrido de `lambda_max` y `tau`**: comprobar si el regimen actual es
   demasiado conservador para ImageNet-C.
2. **Lambda por muestra**: usar familiaridad como control de riesgo, permitiendo
   mezclas mas fuertes solo cuando la muestra lo justifique.
3. **Retrieval que afecte el logit**: cache/kNN o re-ranking tipo Tip-Adapter
   sobre embeddings limpios/corruptos, no solo mezcla en `zq_pool`.
4. **Denoising a manifold limpio**: usar el codebook como prior de correccion mas
   agresivo que un blend `0.1`.

El criterio de exito de la siguiente etapa no debe ser solo `completion > 0` o
`episodic_churn > 0`; debe demostrar que la memoria **llega al logit** con coste
de calibracion acotado.
