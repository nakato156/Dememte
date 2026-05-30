# E6b - DeMemte-ImageNet sobre subset limpio 5k: hallazgos

## TL;DR

- **E6b ya produce un DeMemte-ImageNet utilizable**: el checkpoint queda en
  `experiments/imagenet_dememte/out/dememte_imagenet_resnet50_vqsa_best.pt` y
  E10b puede cargarlo para evaluar memoria sobre ImageNet-C.
- **El modelo aprende de forma clara aun con solo 5k imagenes limpias**:
  `val_acc` sube de `0.099` a `0.614` en 10 epocas.
- **El entrenamiento no usa ImageNet-C**: la base es
  `experiments/data/imagenet-clean-5k`, con `train_size=5000` y
  `val_size=1000`. Las corrupciones entran como augmentation on-the-fly dentro
  de `run_epoch_vqsa`, duplicando cada batch como limpio + corrupto.
- **El codebook deja de ser aleatorio y se abre**: en validacion,
  `hard_usage` sube de `0.336` a `0.722`, `dead_code_fraction` baja de `0.664`
  a `0.278`, y `hard_perplexity` sube de `106.6` a `365.8`.
- **Hay un gap train/val razonable, no colapso evidente**:
  epoch 10 termina con `train_acc=0.673` y `val_acc=0.614`.

## Setup

E6b entrena DeMemte para ImageNet-1K con:

- backbone `ResNet-50 IMAGENET1K_V2` congelado;
- `num_classes=1000`;
- `quantizer_type="ema_vq"`;
- `vq_kmeans_init=True`;
- `dead_code_restart=True`;
- `train_corrupt_prob=0.7`;
- `batch_size=32`;
- `epochs=10`.

El dataset limpio usado por esta corrida:

```text
experiments/data/imagenet-clean-5k
train_size = 5000
val_size   = 1000
```

Cada batch de entrenamiento se duplica internamente:

```text
x_in = [x_clean, corrupt(x_clean)]
y_in = [y, y]
```

Por eso 5k imagenes limpias producen 10k ejemplos efectivos por epoca, con
corrupciones sinteticas nuevas a traves de las epocas.

Resultados en:

- `experiments/imagenet_dememte/out/train_history.json`
- `experiments/imagenet_dememte/out/train_config.json`
- `experiments/imagenet_dememte/out/dememte_imagenet_resnet50_vqsa_best.pt`

## Curva principal

| epoch | train acc | val acc | train loss | val loss | val usage | val dead | val hard ppl |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.011 | 0.099 | 6.802 | 5.962 | 0.336 | 0.664 | 106.6 |
| 2 | 0.097 | 0.320 | 5.343 | 3.660 | 0.561 | 0.439 | 226.0 |
| 3 | 0.286 | 0.480 | 3.697 | 2.640 | 0.563 | 0.437 | 293.9 |
| 4 | 0.444 | 0.538 | 2.730 | 2.221 | 0.653 | 0.347 | 322.9 |
| 5 | 0.536 | 0.570 | 2.309 | 2.118 | 0.641 | 0.359 | 349.7 |
| 6 | 0.596 | 0.587 | 1.991 | 2.040 | 0.639 | 0.361 | 359.8 |
| 7 | 0.620 | 0.591 | 1.871 | 2.004 | 0.649 | 0.351 | 365.1 |
| 8 | 0.642 | 0.593 | 1.791 | 2.080 | 0.667 | 0.333 | 387.4 |
| 9 | 0.644 | 0.604 | 1.759 | 2.028 | 0.676 | 0.324 | 332.2 |
| 10 | **0.673** | **0.614** | **1.646** | 2.106 | **0.722** | **0.278** | 365.8 |

## Insight 1 - La migracion a ImageNet ya tiene una base entrenada

Antes de E6b, E10b corria con un DeMemte-ImageNet aleatorio: accuracy cercana a
`1/1000` y codebook sin semantica de clase. E6b cambia eso. Con solo 5 imagenes
limpias por clase, el modelo alcanza:

```text
best_val_acc = 0.614
```

Esto no debe interpretarse como accuracy ImageNet full, porque el split es un
subset pequeno. Pero si es suficiente para desbloquear la pregunta de E10b:
ahora la memoria se evalua sobre un clasificador y un codebook no aleatorios.

## Insight 2 - El codebook no colapsa; se organiza durante el entrenamiento

El cambio mas importante no es solo accuracy. El codebook gana cobertura:

```text
val_hard_usage:          0.336 -> 0.722
val_dead_code_fraction:  0.664 -> 0.278
val_hard_perplexity:     106.6 -> 365.8
```

En train, `hard_usage` llega casi a `1.0` desde epoch 2, lo que indica que el
modelo explora casi todos los codigos cuando ve batches limpios+corruptos. En
val, la cobertura es menor pero mejora sostenidamente. No estamos viendo el
colapso severo que preocupaba en E10/E10-A.

La lectura: para ImageNet, el codebook necesita entrenamiento supervisado
limpio con augmentation; no basta con instanciar la arquitectura sobre un
backbone pretrained.

## Insight 3 - La augmentation limpia+corrupta parece crucial

El entrenamiento no mezcla ImageNet-C. Las corrupciones son sinteticas y se
aplican on-the-fly sobre imagenes limpias. Eso conserva la separacion correcta:

```text
ImageNet limpio + augmentation sintetica -> train
ImageNet-C real                           -> eval
```

Este punto es importante metodologicamente. E6b crea una base razonable sin
contaminar ImageNet-C, y por eso E10b puede volver a tener valor como benchmark
de robustez y memoria bajo shift.

## Insight 4 - Hay senal de overfit leve, pero no invalida el checkpoint

La curva de `val_loss` mejora hasta epoch 7:

```text
val_loss epoch 7 = 2.0037
```

Luego oscila:

```text
epoch 8 = 2.0795
epoch 9 = 2.0280
epoch 10 = 2.1063
```

Sin embargo `val_acc` sigue subiendo hasta `0.614`. Con `val_size=1000` y solo
1 imagen por clase, la metrica es ruidosa. No hay una senal fuerte de colapso,
pero si conviene guardar tambien una variante seleccionada por `val_loss` o
calibracion en una siguiente corrida si E10b muestra problemas de NLL/ECE.

## Insight 5 - El checkpoint es suficiente para reabrir E10, E10-A y E11

E6b no es el experimento final de robustez. Es el checkpoint base que faltaba.
Ahora se puede medir:

1. si E10 memoria asociativa mejora o rompe bajo ImageNet-C;
2. si E10-A mejora `zq_pool` reparando el codebook localmente;
3. si E11 retrieval-logit funciona mejor usando `z_pool`, `zq_pool` o `fused`;
4. si el codebook entrenado mantiene `hard_usage`, `dead_code_fraction` y
   `hard_perplexity` bajo corrupciones reales.

La comparacion importante ya no es contra ResNet-50 solamente, sino contra:

```text
dememte_imagenet_source
```

usando el checkpoint E6b.

## Cierre

E6b corrige el bloqueo principal de la migracion a ImageNet: ya no estamos
evaluando memoria sobre una cabeza aleatoria. La base entrenada es pequena, pero
metodologicamente limpia y suficientemente fuerte para diagnosticar memoria.

La conclusion practica es clara: **re-ejecutar E10b ahora si tiene sentido**.
Si E10b sigue sin mejorar, el problema ya no sera que DeMemte no aprendio nada;
sera la interfaz de memoria, el routing hacia el logit o la calidad de la clave
`zq_pool` bajo corrupcion.

## Proximo paso

1. Re-ejecutar `notebooks/10b_imagenet_c/e10_imagenet_c.ipynb`.
2. Revisar:
   - delta de `e10_*` contra `dememte_imagenet_source`;
   - `recall_sharpness`;
   - `completion_amount`;
   - `hard_usage` y `dead_code_fraction` bajo corrupcion;
   - ECE/NLL para detectar mejoras que no sean solo accuracy.
3. Si E10 sigue plano, pasar a E11 sobre este checkpoint, porque E11 permite
   que la memoria vote en el logit en vez de entrar solo como mezcla en
   `zq_pool`.
