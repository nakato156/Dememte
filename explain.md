# DeMemte Transformer (`dememte_transformer`) - Explicacion de Arquitectura

Este documento describe en profundidad la variante `dememte_transformer` implementada en `experiments/VQ/dememte_variants.ipynb`.

## 1. Objetivo de la variante

La variante combina un backbone convolucional fuerte (ResNet18 truncada) con una memoria discreta (VQ) y un modulador adaptativo (gate) para:

- preservar rendimiento en entradas limpias,
- mejorar estabilidad bajo perturbaciones,
- reconstruir/regularizar representaciones en espacio de features (no en pixel space).

La idea central: **no reemplazar siempre la representacion del backbone**, sino mezclar dinamicamente `clean_feats` y `rec_feats` segun una senal aprendida de dificultad basada en error de cuantizacion.

## 2. Vista general del pipeline

Flujo de alto nivel por batch:

1. Imagen `x` -> `backbone` -> `input_feats` (mapa 2D de canales semanticos).
2. `input_feats` -> `AttentionSpatialVQVAE` -> `rec_feats`, `vq_loss`, `dq_map`.
3. `dq_map` se normaliza con estadisticas EMA y se convierte en `signal` (gate sigmoidal).
4. Mezcla adaptativa:

   `enhanced = (1 - signal) * input_feats + signal * rec_feats`

5. `enhanced` -> avg pool + linear -> `logits`.
6. Ademas se optimizan perdidas de reconstruccion y cuantizacion.

## 3. Componentes principales

### 3.1 Backbone (`make_backbone`)

- Base: `torchvision.models.resnet18` preentrenada en ImageNet.
- Se eliminan las dos ultimas capas (`avgpool`, `fc`), dejando salida espacial (`B, 512, H, W`).
- En este notebook se congela en entrenamiento de la variante (`set_backbone_trainable(False)`).

Rol:

- extraer representaciones visuales robustas de nivel medio/alto,
- servir como ancla semantica para reconstruccion en espacio feature.

### 3.2 `VectorQuantizer2D`

Entrada: tensor `z_e` de forma `(B, C, H, W)`.

Pasos:

1. Reordena a tokens espaciales (`B*H*W, C`).
2. Calcula distancia L2 a cada embedding del codebook (`num_embeddings x embedding_dim`).
3. Selecciona indice minimo por token (`argmin`).
4. Recupera vector cuantizado y recompone mapa 2D.
5. Usa straight-through estimator para mantener gradiente al encoder.

Salida:

- `q_st`: cuantizado con gradiente ST,
- `vq_loss`: perdida de codebook + commitment,
- `dq_map`: error de cuantizacion por posicion (`mean((z_e - q)^2, dim=canal)`).

Interpretacion:

- `dq_map` funciona como indicador local de mismatch entre feature continua y memoria discreta.

### 3.3 `AttentionSpatialVQVAE`

Es el modulo de memoria/denoising en espacio de features del backbone.

Estructura:

1. **Pre-proyeccion (`pre`)**
   - `1x1 conv -> BN -> ReLU -> 1x1 conv`
   - Reduce/ajusta dimensionalidad de 512 a `embedding_dim`.

2. **Positional embedding aprendible (`pos_embed`)**
   - Parametro `(1, embedding_dim, 7, 7)` con `trunc_normal_`.
   - Si `H,W` difieren, se interpola bilinealmente.

3. **Bloques Transformer espaciales**
   - Flatten espacial: `(B, C, H, W) -> (B, HW, C)`.
   - Para cada capa:
     - Multihead self-attention,
     - residual + LayerNorm,
     - FFN (`Linear -> GELU -> Linear`),
     - residual + LayerNorm.

4. **Cuantizacion discreta (`vq`)**
   - `VectorQuantizer2D` sobre el latente contextualizado por atencion.

5. **Post-proyeccion (`post`)**
   - `1x1 conv -> BN -> ReLU -> 1x1 conv`
   - Reconstruye al espacio de canales del backbone (`in_channels=512`).

Salida:

- `x_rec`: feature reconstruida,
- `vq_loss`,
- `dq_map`.

### 3.4 `DeMemteSpatial`

Encapsula backbone + VQ-VAE espacial + clasificador + gate adaptativo.

Elementos clave:

- `classifier`: linear sobre pooled features.
- Parametros de gate aprendibles:
  - `gate_tau` (umbral),
  - `gate_alpha` (pendiente, pasada por `softplus` para positividad).
- Estadisticas EMA registradas como buffers:
  - `dq_ema_mean`, `dq_ema_var`, `dq_ema_counted`.

#### Forward detallado

1. `input_feats = backbone(x)`.
2. En train puede aplicar `feature_mask_ratio` (masked feature modeling) antes del VQ.
3. `rec_feats, vq_loss, dq_map = vq_vae(vq_input)`.
4. `denoise_loss = MSE(rec_feats, target_feats)`.
   - Por defecto `target_feats = input_feats.detach()`.
5. Actualiza EMA de `dq_map` si corresponde.
6. Normaliza dificultad:

   `dq_norm = (dq_map - mean_ema) / (sqrt(var_ema) + 1e-5)`

7. Convierte a gate:

   `alpha = softplus(gate_alpha)`

   `signal = sigmoid(alpha * (dq_norm - gate_tau))`

8. Mezcla adaptativa:

   `enhanced = (1 - signal) * input_feats + signal * rec_feats`

9. Clasifica usando `enhanced`.

Intuicion del gate:

- Si `dq_norm` es bajo (region confiable), la mezcla favorece `input_feats`.
- Si `dq_norm` sube (posible ruido/out-of-distribution local), aumenta peso de `rec_feats`.

## 4. Funcion de perdida

En fase 2 (clasificacion + regularizacion), la loss total es:

`loss = ce_noisy + denoise_weight * denoise_loss + vq_weight * vq_loss`

Donde:

- `ce_noisy`: cross-entropy sobre la rama con entrada corrupta,
- `denoise_loss`: promedio entre clean y noisy en feature-space,
- `vq_loss`: promedio clean/noisy de la cuantizacion.

Efecto:

- obliga a clasificar correctamente bajo corrupcion,
- mantiene consistencia de representacion,
- evita que el codebook colapse o quede desconectado.

## 5. Regimen de entrenamiento por fases

### Fase 1 - Preentrenamiento de memoria

- Backbone congelado.
- Se optimiza solo reconstruccion + cuantizacion.
- Objetivo: estabilizar codebook y decoder antes de presionar por clasificacion.

### Fase 2 - Entrenamiento conjunto (parcial)

- Backbone sigue congelado.
- Se optimizan:
  - parametros de `vq_vae`,
  - `gate_tau`, `gate_alpha`,
  - clasificador lineal.
- Se usa par clean/noisy del mismo batch:
  - clean para referencia semantica,
  - noisy para objetivo de clasificacion robusta.

Razon del diseno:

- reduce interferencia catasrofica sobre extractor base,
- concentra capacidad en memoria discreta + mecanismo adaptativo.

## 6. Corrupciones en entrenamiento

`apply_train_corruption` usa una mezcla estocastica:

- gaussian noise,
- pixel masking,
- cutout,
- blur por convolucion depthwise con kernel uniforme.

No se aplica siempre; depende de `train_corrupt_prob`.

Beneficio:

- aumenta cobertura de modos de degradacion,
- entrena al gate y al VQ para actuar cuando hay incertidumbre local.

## 7. Por que esta variante puede funcionar mejor que un VQ puramente conv

1. **Contexto global/local por atencion**
   - El latente antes de cuantizar ya integra relaciones espaciales largas.

2. **Memoria discreta condicionada por contexto**
   - El codebook no solo cuantiza texturas locales aisladas.

3. **Gate continuo por dificultad**
   - Evita reemplazo rigido de features.
   - Permite transicion suave entre ruta directa y ruta reconstruida.

4. **Normalizacion temporal (EMA) del error de cuantizacion**
   - Hace el criterio de gating mas estable entre batches.

## 8. Limitaciones y trade-offs

- Costo computacional superior a rutas sin atencion.
- Dependencia de hiperparametros (`embedding_dim`, `num_embeddings`, pesos de loss).
- Con backbone congelado, la adaptacion recae casi toda en memoria/gate.
- Si `dq_ema_var` se estima mal al inicio, el gate puede saturarse temporalmente.

## 9. Hiperparametros mas influyentes

- `embedding_dim`, `num_embeddings`: capacidad/compresion de memoria.
- `attn_heads`, `attn_layers`: capacidad contextual del modulo transformer.
- `denoise_weight`, `vq_weight`: equilibrio clasificacion vs reconstruccion.
- `masked_feature_ratio`: regularizacion del encoder de memoria.
- `init_tau`, `init_alpha`: punto de partida del gate.

## 10. Resumen ejecutivo

La variante `dememte_transformer` es una arquitectura hibrida:

- **Backbone CNN congelada** para semantica base,
- **VQ-VAE espacial con self-attention** para memoria discreta contextual,
- **Gate adaptativo guiado por error de cuantizacion normalizado** para fusion dinamica,
- **Entrenamiento en dos fases** para estabilidad y robustez.

En terminos practicos, actua como un sistema de "correccion de representacion" en feature-space que se activa con mayor intensidad donde la memoria detecta incongruencia local.
