# E11 - Retrieval memory que vota en el logit: hallazgos ImageNet-C

## TL;DR

- **E11 confirma el cuello de botella de E10**: cuando la memoria deja de entrar
  como soft-mix latente y vota directamente en el logit,
  `logits_final = logits_base + alpha * logits_cache`, si puede mover accuracy.
- **La clave buena es `z_pool`, no `zq_pool`**. La mejor variante,
  `source_cache_z_pool_fixed_alpha`, sube `corrupt_acc_avg` de `0.217402` a
  `0.242362` (`+0.024960`, +2.50 pp) y clean de `0.614` a `0.636`.
- **`zq_pool` como retrieval key esta aliasado**: con `alpha=1`, corrige
  `0.0237` de las muestras corruptas pero rompe `0.0246`; neto negativo leve.
  El gate de unfamiliarity reduce dano, pero solo deja una mejora pequena
  (`+0.002592`).
- **La memoria episodica con pseudo-labels falla**. `episodic_cache_zq_pool` y
  `dual_cache_zq_pool` escriben ejemplos, pero rompen mas de lo que reparan y
  caen por debajo del source (`-1.36 pp` y `-1.01 pp`).
- **La ganancia de accuracy tiene coste de calibracion**. El mejor metodo por
  accuracy (`z_pool`) empeora ECE corrupto de `0.146770` a `0.188611` y NLL de
  `5.110213` a `5.124382`. E11 es positivo, pero no esta cerrado.

## Setup

E11 prueba la hipotesis post-E10: la memoria asociativa esta mecanicamente viva,
pero el blend suave en `zq_pool` es demasiado debil para cambiar predicciones.
Aqui la memoria se convierte en una cabeza kNN/cache que suma logits al
clasificador congelado:

```text
logits_final = logits_base + alpha_eff(x) * logits_cache
```

La cache usa afinidades tipo Tip-Adapter sobre vecinos normalizados. Las caches
source se construyen con etiquetas limpias de ImageNet train; las caches
episodicas se escriben online con pseudo-labels del teacher cuando la confianza
supera `write_confidence=0.8`.

Variantes evaluadas:

- `source_cache_zq_pool_fixed_alpha`
- `source_cache_z_pool_fixed_alpha`
- `source_cache_fused_fixed_alpha`
- `source_cache_zq_pool_unfamiliarity_alpha`
- `episodic_cache_zq_pool`
- `dual_cache_zq_pool`

Resultados actuales en:

- `notebooks/11_retrieval_memory/out/e11_results.csv`
- `notebooks/11_retrieval_memory/out/e11_curves.csv`
- `notebooks/11_retrieval_memory/out/e11_summary.md`
- `notebooks/11_retrieval_memory/out/<variante>/metrics.json`
- `notebooks/11_retrieval_memory/out/<variante>/signal_curves.csv`

Datos y modelo:

- Base: `dememte_imagenet_resnet50_vqsa`.
- Checkpoint:
  `experiments/imagenet_dememte/out/dememte_imagenet_resnet50_vqsa_best.pt`.
- Clean: `experiments/data/imagenet-clean-5k`, `val_size=1000`.
- Cache source: `5000` ejemplos limpios de train.
- Corrupt: `experiments/data/imagenet-c-subset`.
- Condiciones: `gaussian_noise`, `motion_blur`, `pixelate`,
  `jpeg_compression`, severidades `3` y `5`.
- Cada condicion ImageNet-C evaluada con `50000` imagenes.

`RUN_ORACLE_DIAGNOSTIC=False` en esta corrida. Cualquier cache oracle con labels
de evaluacion es solo headroom diagnostico y no debe reportarse como metodo
test-time valido.

## Matriz principal

Base `dememte_imagenet_resnet50_vqsa`, dataset `imagenet_c`.
Source: clean `0.6140`, corrupt `0.2174`.

| variante | clean | corrupt avg | Delta clean | Delta corrupt | ECE corrupt | NLL corrupt | flip rate | corrige | rompe | alpha |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `source_cache_z_pool_fixed_alpha` | **0.6360** | **0.2424** | **+0.0220** | **+0.0250** | 0.1886 | 5.1244 | 0.1642 | **0.0317** | 0.0067 | 1.0000 |
| `source_cache_fused_fixed_alpha` | 0.6300 | 0.2254 | +0.0160 | +0.0080 | 0.1771 | 5.1655 | **0.0779** | 0.0129 | **0.0049** | 1.0000 |
| `source_cache_zq_pool_unfamiliarity_alpha` | 0.6180 | 0.2200 | +0.0040 | +0.0026 | 0.1861 | 5.1684 | 0.2195 | 0.0170 | 0.0144 | 0.6358 |
| `source` | 0.6140 | 0.2174 | 0.0000 | 0.0000 | **0.1468** | **5.1102** | - | - | - | - |
| `source_cache_zq_pool_fixed_alpha` | 0.6120 | 0.2165 | -0.0020 | -0.0009 | 0.2080 | 5.2746 | 0.2750 | 0.0237 | 0.0246 | 1.0000 |
| `dual_cache_zq_pool` | 0.6170 | 0.2073 | +0.0030 | -0.0101 | 0.2985 | 5.7714 | 0.3852 | 0.0193 | 0.0294 | 0.6358 |
| `episodic_cache_zq_pool` | 0.6110 | 0.2039 | -0.0030 | -0.0136 | 0.2739 | 5.7071 | 0.3113 | 0.0119 | 0.0255 | 0.6358 |

Por corrupcion, la mejor variante valida es siempre `source_cache_z_pool_fixed_alpha`:

| corrupcion | source acc | mejor variante | mejor acc | delta |
|---|---:|---|---:|---:|
| `gaussian_noise` | 0.1453 | `source_cache_z_pool_fixed_alpha` | **0.1647** | +0.0194 |
| `motion_blur` | 0.1327 | `source_cache_z_pool_fixed_alpha` | **0.1526** | +0.0199 |
| `pixelate` | 0.2408 | `source_cache_z_pool_fixed_alpha` | **0.2689** | +0.0280 |
| `jpeg_compression` | 0.3508 | `source_cache_z_pool_fixed_alpha` | **0.3833** | +0.0326 |

## Insight 1 - E11 si llega al logit

E10 dejo una lectura clara: la memoria se activaba, pero no cambiaba la
decision. E11 cambia la interfaz y la memoria pasa a votar al lado del
clasificador. La diferencia es visible:

```text
source.corrupt_acc_avg                         = 0.217402
source_cache_z_pool_fixed_alpha.corrupt_acc_avg = 0.242362  (+0.024960)
```

La mejora no viene de una metrica interna cosmetica. Se ve como reparacion neta
de predicciones:

```text
corrected_by_retrieval_corrupt_avg = 0.031695
broken_by_retrieval_corrupt_avg    = 0.006735
net repair                         = +0.024960
```

Eso cierra el diagnostico de E10: el problema no era que la memoria estuviera
muerta, sino que su via de entrada al modelo era demasiado indirecta.

## Insight 2 - `z_pool` es la representacion correcta para recuperar vecinos

`z_pool` domina a `zq_pool` y a `fused` tanto en clean como en corrupcion:

```text
z_pool_fixed.corrupt_acc_avg = 0.242362
fused_fixed.corrupt_acc_avg  = 0.225400
zq_fixed.corrupt_acc_avg     = 0.216537
```

La razon probable esta en la geometria. `z_pool` conserva vecindad continua del
backbone/proyector antes de cuantizar; `zq_pool` fuerza muchos ejemplos a
prototipos discretos. Bajo corrupcion, esa discretizacion parece mezclar clases
o perder detalle fino.

El diagnostico de acuerdo tambien apunta ahi:

```text
retrieval_agreement_corrupt_avg(z_pool) = 0.3210
retrieval_agreement_corrupt_avg(zq)     = 0.1084
retrieval_agreement_corrupt_avg(fused)  = 0.3876
```

`fused` acuerda mas y rompe poco, pero tambien corrige poco. `z_pool` tiene el
mejor balance: suficiente desacuerdo para reparar, pero no tanto como para
destruir.

## Insight 3 - `zq_pool` activa memoria, pero esta demasiado aliasado

`source_cache_zq_pool_fixed_alpha` no es inerte. Cambia muchas predicciones:

```text
flip_rate_corrupt_avg = 0.274972
```

El problema es la direccion del cambio:

```text
corrected_by_retrieval_corrupt_avg = 0.023687
broken_by_retrieval_corrupt_avg    = 0.024552
delta_corrupt_vs_source            = -0.000865
```

Es decir, la cache `zq_pool` tiene fuerza suficiente para intervenir, pero no
precision suficiente para elegir vecinos utiles. La cuantizacion hace que la
memoria vote, pero vota con ruido semantico.

El gate por unfamiliarity es un buen control de riesgo:

```text
zq_fixed.delta_corrupt         = -0.000865
zq_unfamiliarity.delta_corrupt = +0.002592
```

Pero incluso con gate queda muy lejos de `z_pool`. La lectura practica es que
`zq_pool` puede servir como diagnostico de memoria cuantizada, no como clave
principal de retrieval.

## Insight 4 - La memoria episodica online se contamina con pseudo-labels

Las variantes episodicas escriben ejemplos durante evaluacion. No estan
apagadas:

```text
tta_selection_rate_corrupt_avg(episodic) = 0.1417
tta_selection_rate_corrupt_avg(dual)     = 0.1417
```

Pero la escritura no ayuda. En corrupt average:

```text
episodic_cache_zq_pool.corrected = 0.011922
episodic_cache_zq_pool.broken    = 0.025472
dual_cache_zq_pool.corrected     = 0.019322
dual_cache_zq_pool.broken        = 0.029388
```

El patron es el esperado para una cache online con pseudo-labels bajo shift: si
el teacher esta confiado pero equivocado, la memoria acumula errores y luego los
reinyecta con margen alto. El caso dual no lo arregla; al sumar source + episodic
en `zq_pool`, aumenta el flip rate (`0.3852`) y empeora ECE/NLL.

Esta es una senal importante para la tesis biologica: **plasticidad episodica sin
filtro de verdad o sin control de contaminacion no es memoria util, es
auto-refuerzo**.

## Insight 5 - Accuracy mejora, calibracion no

E11 tiene un resultado positivo en accuracy, pero la calibracion queda como
deuda:

```text
source.ece_corrupt_avg = 0.146770
z_pool.ece_corrupt_avg = 0.188611

source.nll_corrupt_avg = 5.110213
z_pool.nll_corrupt_avg = 5.124382
```

La cache ayuda a elegir mas clases correctas, pero modifica la confianza de una
forma peor calibrada. `fused` tiene menor flip rate (`0.0779`) y ECE menor que
`z_pool` (`0.1771`), pero sacrifica casi dos tercios de la ganancia de accuracy.

La siguiente etapa no debe preguntar solo "puede retrieval mejorar accuracy?".
E11 ya responde que si. La pregunta ahora es:

```text
como conservar el +2.5 pp de z_pool con ECE/NLL acotados?
```

## Cierre

E11 cierra la lectura abierta por E10:

- La memoria no estaba muerta; la interfaz era demasiado suave.
- Un cache/kNN que vota en logits si produce reparacion neta.
- La representacion util para recuperar vecinos es `z_pool`.
- La representacion cuantizada `zq_pool` es demasiado ruidosa como clave.
- La memoria episodica online con pseudo-labels empeora bajo ImageNet-C.
- La calibracion queda peor y debe formar parte del criterio de exito.

La conclusion practica es clara: **E11 es el primer resultado positivo de
memoria en accuracy ImageNet-C para este checkpoint**, pero el resultado ganador
no es "mas biologico"; es una cache source supervisada sobre embeddings
continuos.

## Proximo paso

No conviene seguir ampliando episodic/dual sobre `zq_pool` sin resolver
contaminacion. El siguiente experimento debe consolidar el hallazgo `z_pool`:

1. **Sweep de `alpha_max`** en `source_cache_z_pool`: probar `0.25`, `0.5`,
   `0.75`, `1.0`, y quizas `1.5`, midiendo accuracy y ECE/NLL.
2. **Sweep de `top_k` y `beta`**: separar si la ganancia viene de vecinos
   pocos y fuertes o de una distribucion mas suave.
3. **Gate por acuerdo/margen**: usar `retrieval_margin`, `retrieval_entropy` y
   acuerdo base-cache para evitar intervenciones de baja confianza.
4. **Calibracion post-cache**: temperatura o escala separada para
   `logits_cache`, con objetivo de mantener la ganancia sin inflar ECE.
5. **Cache source mas grande o balanceada**: comprobar si 5k ejemplos son el
   techo o si mas cobertura de clases mejora corrupciones severas.

El criterio de exito de E11.v2 debe ser doble: mantener una mejora robusta de
accuracy contra source y demostrar que el coste de calibracion queda controlado.
