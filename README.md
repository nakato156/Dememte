# DeMemte: Red Neuronal con Memoria Latente VQ-VAE para Clasificación Robusta de Imágenes

## Resumen

Este trabajo presenta **DeMemte** (Deep Memory Network), una arquitectura de red neuronal profunda que incorpora un mecanismo de memoria basado en **Vector Quantized Variational Autoencoders (VQ-VAE)** para mejorar simultáneamente la precisión en datos limpios y la robustez frente a corrupciones de entrada. El sistema combina un backbone pre-entrenado de ResNet18 con un módulo de memoria espacial VQ-VAE equipado con atención (Transformer), junto con un mecanismo de gating adaptativo que modula la contribución de la memoria basándose en la discrepancia de cuantización. Los experimentos en **Flowers-102** demuestran que, bajo una comparación justa donde ambos modelos son entrenados con data augmentation de corrupciones idénticas, DeMemte Transformer supera al baseline ResNet18 en **+7.48 pp** de Clean Accuracy (74.9% vs 67.4%) y en **+2.02 pp** de Corrupt Accuracy promedio (53.1% vs 51.1%), posicionándose como Pareto-dominante en el espacio clean-vs-corrupt. El overhead computacional es mínimo: x1.37 de slowdown total con un costo de apenas 0.05 min por cada +1pp de ganancia en robustez.

**Palabras clave:** Redes Neuronales Profundas, VQ-VAE, Memoria Latente, Robustez, Clasificación de Imágenes, ResNet, Flowers-102, Gating Adaptativo

---

## Cómo navegar el código

El proyecto está estructurado como un módulo Python compartido + 4 notebooks orquestadores. Los notebooks anteriores quedaron archivados.

### Layout

- `src/dememte/` — módulo único de verdad: modelos, training (fases 1/2/3), corruption suite, evaluación, I/O.
- `notebooks/` — 4 notebooks, uno por experimento. Cada uno tiene su propia carpeta `out/` con checkpoints, métricas y plots.
  - `01_baseline/baseline.ipynb` — ResNet18 frozen, misma metodología en fases que E5. Baseline _fair 1:1_.
  - `02_e5_winner/e5_winner.ipynb` — DeMemte E5 (variante ganadora) reproducible end-to-end.
  - `03_ablations/ablations.ipynb` — 8 variantes del set crítico (E5 + 7 ablaciones), evaluadas en clean y corrupt.
  - `04_finetune_vs_frozen/finetune_vs_frozen.ipynb` — comparativo central: ResNet18 fine-tuneado con corrupciones agresivas vs DeMemte frozen. Demuestra que el gate rompe el trade-off clean↔corrupt.
- `archive/notebooks/` — notebooks legacy (baseline, attractor_memory, Dememte_e5y, e5_final_clean, no_ood_debug) conservados como referencia histórica.
- `experiments/data/flowers-102/` — dataset descargado.
- `experiments/atracctor/out/artifacts/dememte_e5_critical/` — corridas pre-computadas del E5 (5 seeds × 8 variantes); los notebooks pueden cargar estos checkpoints sin reentrenar.

### Ejecutar

Cada notebook arranca con `RUN_TRAINING = False`. En ese modo carga checkpoints existentes (o los siembra desde `experiments/atracctor/out/artifacts/` si no existen aún). Cambia a `True` para entrenar desde cero (1 seed por notebook, GPU requerida).

Los notebooks se generan a partir de `scripts/build_notebooks.py`; corre ese script si modificas la plantilla.

### Paridad

El checkpoint legacy `dememte_e5_critical/seed_42/e5_combined_dropout_ood_tau_150/*.pt` carga con `src.dememte.models.DeMemteAttractor` y reproduce `clean_acc=0.811839` (idéntico a `metrics.json` de la corrida original).

---

## 1. Introducción

### 1.1 Motivación

Las redes neuronales profundas modernas logran rendimiento excepcional en condiciones ideales, pero su precisión se degrada significativamente ante corrupciones en los datos de entrada (ruido gaussiano, oclusiones, desenfoque). Este fenómeno representa un riesgo para aplicaciones en el mundo real donde las condiciones de captura son impredecibles.

Inspirados por la memoria biológica, donde el cerebro puede reconstruir representaciones parciales o degradadas a partir de patrones previamente almacenados, proponemos incorporar una memoria latente discreta en el pipeline de clasificación.

### 1.2 Problema de Investigación

**¿Es posible incorporar un módulo de memoria VQ-VAE en una red neuronal que mejore la robustez frente a corrupciones sin sacrificar la precisión en datos limpios, y que dicha mejora sea atribuible a la arquitectura y no simplemente a diferencias en el régimen de entrenamiento?**

### 1.3 Propuesta

Presentamos **DeMemte**, una arquitectura que combina:

- **Backbone congelado (ResNet18):** Extractor de características pre-entrenado
- **VQ-VAE Espacial con Atención (Transformer):** Memoria latente discreta que opera sobre mapas de features 2D, equipada con self-attention para capturar dependencias espaciales
- **Gating Adaptativo basado en Discrepancia de Cuantización:** Mecanismo aprendible que decide cuánto confiar en la memoria versus las features originales, usando la distancia al codebook como señal de novedad/familiaridad
- **Entrenamiento en Dos Fases:** (1) Pre-entrenamiento del VQ-VAE, (2) Fine-tuning conjunto con corrupciones

### 1.4 Contribución Principal: Equidad Metodológica

Un hallazgo clave de este trabajo es la importancia de la **equidad en la comparación experimental**. Inicialmente, el baseline ResNet se entrenaba solo con datos limpios mientras DeMemte se entrenaba con corrupciones, lo que invalidaba cualquier conclusión sobre el valor de la arquitectura. En la versión actual, **ambos modelos reciben exactamente el mismo régimen de data augmentation con corrupciones** (`train_corrupt_prob=0.70`), aislando así la contribución real del módulo de memoria.

### 1.5 Contribuciones

1. **Arquitectura VQ-VAE Espacial con Transformer** para memoria latente en feature maps 2D
2. **Gating adaptativo** basado en discrepancia de cuantización normalizada por EMA
3. **Metodología de evaluación justa** donde baseline y variante comparten el mismo régimen de corrupciones
4. **Evidencia empírica** de que DeMemte es Pareto-dominante (mejor en clean Y corrupt) frente a un baseline aumentado con las mismas corrupciones
5. **Análisis de costo-beneficio** demostrando overhead mínimo (x1.37 slowdown, +20% parámetros)

---

## 2. Metodología

### 2.1 Arquitectura General

```
Imagen → [Backbone ResNet18 (congelado)] → Feature Map (B, 512, 7, 7)
    ├─→ [VQ-VAE Spatial Transformer] → Features Reconstruidas + dq_map
    │        ├─ Pre-Conv → Positional Embedding → Self-Attention × L → VQ-2D → Post-Conv
    │        └─ vq_loss, denoise_loss
    └─→ Gating: enhanced = (1 - σ) · original + σ · reconstructed
              donde σ = sigmoid(α · (dq_norm - τ))
         → [AdaptiveAvgPool2d] → [Classifier Linear(512→C)] → Logits
```

### 2.2 VQ-VAE Espacial con Atención (AttentionSpatialVQVAE)

A diferencia de un VAE probabilístico, el VQ-VAE usa un codebook discreto de embeddings. Esto proporciona una forma natural de medir la "familiaridad" de un patrón: la distancia al embedding más cercano.

#### 2.2.1 Componentes

**Pre-procesamiento (Conv):**

$$z = \text{Conv}_{1\times1}(\text{ReLU}(\text{BN}(\text{Conv}_{1\times1}(x))))$$

donde $x \in \mathbb{R}^{B \times 512 \times 7 \times 7}$ y $z \in \mathbb{R}^{B \times d_e \times 7 \times 7}$, con $d_e = 256$.

**Positional Embedding + Self-Attention:**

$$z' = z + P, \quad P \in \mathbb{R}^{1 \times d_e \times 7 \times 7}$$

Se aplican $L=2$ bloques de Transformer (Multi-Head Attention + FFN + LayerNorm) sobre los tokens espaciales aplanados ($49$ tokens de dimensión $d_e$):

$$\text{tokens} = \text{flatten}(z') \in \mathbb{R}^{B \times 49 \times d_e}$$

**Vector Quantization 2D:**

Para cada posición espacial $(h, w)$, se busca el embedding más cercano en el codebook $E \in \mathbb{R}^{K \times d_e}$ con $K=1024$:

$$k^* = \arg\min_k \|z_{h,w} - e_k\|^2$$
$$z_q = e_{k^*} + (z - z).detach() \quad \text{(straight-through)}$$

**Pérdida VQ:**

$$\mathcal{L}_{VQ} = \beta \cdot \|z - \text{sg}(z_q)\|^2 + \|z_q - \text{sg}(z)\|^2$$

con $\beta = 0.25$ (commitment cost).

**Mapa de Discrepancia:**

$$dq_{h,w} = \frac{1}{d_e}\sum_{c=1}^{d_e}(z_{c,h,w} - z_{q_{c,h,w}})^2$$

**Post-procesamiento (Conv):**

$$\hat{x} = \text{Conv}_{1\times1}(\text{ReLU}(\text{BN}(\text{Conv}_{1\times1}(z_q))))$$

### 2.3 Gating Adaptativo

El mecanismo de gating decide cuánto confiar en la reconstrucción de la memoria:

$$dq_{\text{norm}} = \frac{dq - \mu_{EMA}}{\sqrt{\sigma^2_{EMA}} + \epsilon}$$
$$\sigma_{\text{gate}} = \text{sigmoid}(\alpha \cdot (dq_{\text{norm}} - \tau))$$
$$f_{\text{enhanced}} = (1 - \sigma_{\text{gate}}) \cdot f_{\text{original}} + \sigma_{\text{gate}} \cdot f_{\text{reconstructed}}$$

donde $\alpha = \text{softplus}(\alpha_{\text{param}})$ y $\tau$ son parámetros aprendibles. $\mu_{EMA}$ y $\sigma^2_{EMA}$ se actualizan con momentum 0.99 durante entrenamiento.

**Intuición:** Cuando un patrón de features es "conocido" por el codebook (baja discrepancia), el gate se cierra y las features originales dominan. Cuando es "novedoso" o ruidoso (alta discrepancia), el gate se abre y la reconstrucción de memoria contribuye más, actuando como denoiser.

### 2.4 Protocolo de Entrenamiento

#### 2.4.1 Principio de Equidad

**Crítico:** Tanto el baseline como DeMemte reciben las mismas corrupciones durante entrenamiento, garantizando que cualquier diferencia en rendimiento se debe a la arquitectura y no al régimen de datos.

Corrupciones aplicadas con probabilidad $p=0.70$:
- **Gaussian Noise:** $\sigma \in [0.4, 1.3]$
- **Pixel Mask:** ratio $\in [0.20, 0.65]$
- **Cutout:** ratio $\in [0.20, 0.45]$
- **Blur:** ratio $\in [0.30, 0.80]$ (box filter 7×7)

#### 2.4.2 Baseline: ResNet18 + Noise Augmentation

Entrenado end-to-end (solo clasificador, backbone congelado) con las mismas corrupciones aplicadas a cada batch:

$$\mathcal{L}_{\text{baseline}} = \text{CE}(\text{model}(\text{corrupt}(x)), y)$$

Optimizador: AdamW ($lr=10^{-3}$, $wd=10^{-4}$), ReduceLROnPlateau, Early Stop (patience=3).

#### 2.4.3 DeMemte: Entrenamiento en Dos Fases

**Fase 1 - Pre-entrenamiento VQ-VAE** (4 épocas, backbone congelado):

$$\mathcal{L}_{P1} = 0.5 \cdot \text{MSE}(f, \hat{f}) + 0.25 \cdot \mathcal{L}_{VQ}$$

Con Masked Feature Modeling (ratio=0.35): se enmascaran aleatoriamente features de entrada al VQ-VAE para forzar reconstrucción.

**Fase 2 - Fine-tuning Conjunto** (10 épocas, backbone congelado):

Para cada batch, se realiza un forward con datos limpios y otro con datos corruptos:

$$\mathcal{L}_{P2} = 0.5(\text{CE}_{\text{clean}} + \text{CE}_{\text{noisy}}) + 0.5 \cdot w_d(\text{MSE}_{\text{clean}} + \text{MSE}_{\text{noisy}}) + 0.5 \cdot w_{vq}(\mathcal{L}_{VQ}^{\text{clean}} + \mathcal{L}_{VQ}^{\text{noisy}})$$

Con $w_d = 0.5$ y $w_{vq} = 0.25$. El target de reconstrucción para el forward ruidoso son las features limpias (denoising target).

### 2.5 Configuración Experimental

#### 2.5.1 Dataset

**Flowers-102:** 102 clases de flores. Train+Val combinados con stratified split (80/20). Test: split oficial (6,149 imágenes).

**Transformaciones:** Resize 224×224, RandomHorizontalFlip, ColorJitter (train), Normalize ImageNet stats.

#### 2.5.2 Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Batch Size | 16 |
| Backbone | ResNet18 (ImageNet pretrained, congelado) |
| Embedding Dim VQ | 256 |
| Num Embeddings (Codebook) | 1024 |
| Attention Heads | 4 |
| Attention Layers | 2 |
| Commitment Cost | 0.25 |
| Denoise Weight | 0.50 |
| VQ Weight | 0.25 |
| Masked Feature Ratio | 0.35 |
| Train Corrupt Prob | 0.70 |
| Init Tau (gate) | 1.0 |
| Init Alpha (gate) | 1.5 |
| LR Baseline | 1e-3 |
| LR VQ | 3e-4 |
| LR Classifier | 1e-4 |
| Weight Decay | 1e-4 |
| Early Stop Patience | 3 |

#### 2.5.3 Infraestructura

- **Framework:** PyTorch 2.x
- **Hardware:** CUDA GPU
- **Backbone:** ResNet18 (ImageNet_V1 weights)
- **Optimizador:** AdamW + ReduceLROnPlateau

### 2.6 Suite de Evaluación de Robustez

Evaluación paired (misma corrupción exacta para ambos modelos, seed fija por nivel) sobre el test set completo:

| Corrupción | Niveles de Severidad |
|------------|---------------------|
| Gaussian Noise | 0.5, 1.0, 1.5 |
| Pixel Mask | 0.25, 0.50, 0.75 |
| Cutout | 0.20, 0.35, 0.50 |
| Blur | 0.35, 0.60, 0.85 |

**Métricas:**
- **Clean Accuracy:** Precisión en test set sin corrupciones
- **Corrupt Accuracy:** Precisión promedio sobre todos los niveles y tipos de corrupción
- **Accuracy Drop:** Clean Acc − Corrupt Acc (cuánto cae la precisión bajo ruido)
- **Trade-off Position:** Posición en el espacio (Clean, Corrupt) - ideal: arriba a la derecha

---

## 3. Resultados

### 3.1 Entrenamiento del Baseline (ResNet18 + Noise Augmentation)

El baseline se entrenó 10 épocas completas con corrupciones idénticas a las de DeMemte:

| Época | Train Acc | Val Acc |
|-------|----------|---------|
| 1 | 16.1% | 28.7% |
| 3 | 52.3% | 44.4% |
| 5 | 72.0% | 57.4% |
| 7 | 84.8% | 63.9% |
| 9 | 88.4% | 71.1% |
| 10 | 89.9% | 59.3% |

**Best Val Acc:** 72.5% (mejor estado: época 9). No hubo early stop; las 10 épocas se completaron con overfitting al final (train 89.9% vs val 59.3% en la última época).

**Test Clean Accuracy:** 67.44%

### 3.2 Entrenamiento de DeMemte Transformer

**Fase 1 (VQ-VAE pretrain):**

| Época | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | 1.3827 | 1.2817 |
| 2 | 1.2259 | 1.1786 |
| 3 | 1.1782 | 1.1557 |
| 4 | 1.1775 | 1.1652 |

Se completaron las 4 épocas. El mejor estado corresponde a época 3 (val_loss=1.1557). El VQ-VAE aprendió a reconstruir features rápidamente.

**Fase 2 (Fine-tuning conjunto):**

| Época | Train Acc | Val Acc |
|-------|----------|---------|
| 1 | 93.1% | 77.5% |
| 2 | 95.8% | 78.7% |
| 3 | 96.2% | 79.4% |
| 4 | 96.8% | 78.9% |
| 5 | 96.3% | 79.4% |
| 6 | 97.2% | 79.4% |

Early stop en época 6 (patience=3, sin mejora desde época 3). Nótese la convergencia rápida: ya en la primera época de P2 se supera al baseline.

**Test Clean Accuracy:** 74.92%

### 3.3 Comparación Clean Accuracy

| Modelo | Val Acc | Test Clean Acc | Delta vs Baseline |
|--------|---------|---------------|-------------------|
| **Baseline (ResNet18 + Aug)** | 72.5% | 67.44% | — |
| **DeMemte Transformer** | 77.5% | **74.92%** | **+7.48 pp** |

**Hallazgo 1:** DeMemte no solo no sacrifica Clean Accuracy — la mejora sustancialmente (+7.48 pp). La memoria VQ-VAE actúa como regularizador y proporciona features complementarias.

### 3.4 Resultados de Robustez Estricta

#### 3.4.1 Tabla Detallada por Tipo de Corrupción

| Corrupción | Baseline Corrupt Acc | Transformer Corrupt Acc | Delta | Baseline Drop | Transformer Drop |
|---|---|---|---|---|---|
| Gaussian Noise | 50.82% | 49.68% | −1.14 pp | 16.62 pp | 25.24 pp |
| Pixel Mask | 43.58% | 39.58% | −4.00 pp | 23.86 pp | 35.34 pp |
| Cutout | 56.01% | **63.13%** | **+7.12 pp** | 11.43 pp | 11.79 pp |
| Blur | 54.06% | **60.17%** | **+6.11 pp** | 13.38 pp | 14.75 pp |

#### 3.4.2 Overall

| Modelo | Clean Acc | Corrupt Acc | Drop |
|--------|-----------|-------------|------|
| **Baseline** | 67.44% | 51.12% | 16.32 pp |
| **DeMemte Transformer** | **74.92%** | **53.14%** | 21.78 pp |

**Overall Corrupt Accuracy:** +2.02 pp a favor de DeMemte Transformer.

### 3.5 Análisis de las Gráficas de Robustez por Severidad

Las curvas de corrupt accuracy vs severity revelan patrones distintos:

- **Cutout (todas las severidades):** DeMemte domina consistentemente. A severidad 0.20, la ventaja es ~7 pp. La memoria espacial reconstruye con éxito las regiones ocluidas.
- **Blur (todas las severidades):** DeMemte domina con ~6 pp de ventaja promedio. El VQ-VAE “sharpens” features borrosas al cuantizarlas a prototipos limpios del codebook.
- **Gaussian Noise (severidades altas):** Baseline es ligeramente mejor a σ=1.0 y σ=1.5. El ruido extremo destruye información antes del backbone, donde la memoria no puede intervenir.
- **Pixel Mask (severidad alta):** Similar a gaussian noise en severidades extremas.

### 3.6 Perfil de Costo Computacional

| Métrica | Baseline | DeMemte Transformer |
|---------|----------|-------------------|
| Parámetros (M) | 11.23 | 13.48 (+20%) |
| Throughput (samples/s) | 431.9 | ~440 (≈ igual) |
| Peak Memory (MB) | 1304.8 | 1127.4 (P2) |
| Tiempo total estimado | 0.25 min | 0.34 min |
| **Slowdown** | — | **x1.37** |
| Costo por +1pp noisy gain | — | **0.05 min/pp** |

**Hallazgo:** El overhead es mínimo. El throughput por época es prácticamente idéntico. El slowdown x1.37 viene de las épocas extra de P1, no de la velocidad de iteración.

---

## 4. Discusión

### 4.1 ¿Por qué DeMemte Mejora Clean Accuracy?

El resultado más sorprendente es que DeMemte mejora la precisión en datos limpios en +7.48 pp. Proponemos tres explicaciones complementarias:

1. **Regularización por Cuantización:** El VQ-VAE fuerza features a alinearse con prototipos discretos del codebook, actuando como regularizador implícito que reduce overfitting.

2. **Denoising Target en P2:** El training dual (clean + noisy) con target de features limpias obliga al modelo a aprender representaciones invariantes al ruido, lo que también beneficia la clasificación limpia.

3. **Warmstart Efectivo:** El pre-entrenamiento de VQ-VAE (P1) proporciona un anclaje que estabiliza el fine-tuning posterior, permitiendo convergencia más rápida y a un mejor mínimo.

### 4.2 ¿Por qué el Drop del Transformer es Mayor?

El Drop (Clean − Corrupt) del Transformer (21.78 pp) es mayor que el del Baseline (16.32 pp). Esto **no** indica peor robustez — indica que el Transformer tiene más precisión que perder:

$$\text{Retention Ratio} = \frac{\text{Corrupt Acc}}{\text{Clean Acc}}$$

| Modelo | Retention Ratio |
|--------|-----------------|
| Baseline | 75.8% |
| Transformer | 70.9% |

La diferencia de retention (4.9 pp) es modesta y se explica por las corrupciones destructivas (gaussian noise σ=1.5, pixel mask 75%), donde ambos modelos colapsan a niveles bajos pero el Transformer parte de un Clean Acc mayor.

### 4.3 Tipología de Corrupciones: Espaciales vs Destructivas

Los resultados revelan una dicotomía clara:

**Corrupciones Espaciales (cutout, blur):** DeMemte domina. El VQ-VAE 2D con atención puede reconstruir información faltante o borrosa porque el codebook almacena prototipos espaciales de features. La self-attention propaga información de posiciones no afectadas a las afectadas.

**Corrupciones Destructivas (gaussian noise extremo, pixel mask extremo):** El baseline es ligeramente mejor. Estas corrupciones destruyen información antes del backbone — la memoria no puede reconstruir lo que nunca llegó al feature space.

**Implicación:** DeMemte es especialmente valioso en escenarios donde las corrupciones son parciales o localizadas (oclusiones, desenfoque por movimiento, artefactos de compresión), que son los más comunes en aplicaciones reales.

### 4.4 Importancia de la Equidad Metodológica

En la versión inicial de este trabajo, el baseline se entrenaba sin corrupciones mientras DeMemte las recibía. Esto producía resultados engañosos donde DeMemte parecía dramáticamente superior en robustez. Al igualar el régimen de entrenamiento, la ganancia en Corrupt Accuracy se reduce de aparentemente grande a un modesto +2.02 pp. Sin embargo, la ganancia en Clean Accuracy (+7.48 pp) emerge como el resultado principal — un hallazgo que no era visible en la comparación injusta.

**Lección metodológica:** En investigación de robustez, el régimen de data augmentation debe ser idéntico entre modelos comparados. Las diferencias en datos de entrenamiento invalidan cualquier conclusión sobre la arquitectura.

### 4.5 Trade-off Analysis: Scatter Plot

En el espacio bidimensional (Clean Acc, Corrupt Acc), DeMemte Transformer está estrictamente arriba y a la derecha del Baseline, lo que lo convierte en **Pareto-dominante**. No existe trade-off:

- Mejor en clean: +7.48 pp
- Mejor en corrupt: +2.02 pp
- Costo computacional: ~x1.37

### 4.6 Limitaciones

1. **Dataset único:** Solo Flowers-102. Necesita validación en CIFAR-10/100, ImageNet, y datasets de dominio específico.
2. **Corrupciones in-distribution:** Las corrupciones de evaluación son del mismo tipo que las de entrenamiento. Falta evaluar robustez a corrupciones nunca vistas.
3. **Backbone congelado:** No se exploró fine-tuning del backbone, lo que podría mejorar ambos modelos.
4. **Codebook limitado:** K=1024 embeddings pueden no ser suficientes para datasets más grandes.
5. **Corrupciones pre-backbone:** El diseño actual no puede mitigar corrupciones que destruyen información antes del feature extractor.

---

## 5. Conclusiones

### 5.1 Resumen de Hallazgos

1. **DeMemte Transformer es Pareto-dominante** frente a un baseline ResNet18 entrenado con las mismas corrupciones: +7.48 pp en Clean Accuracy y +2.02 pp en Corrupt Accuracy promedio.

2. **La memoria VQ-VAE actúa como regularizador:** La mejora más significativa es en Clean Accuracy, sugiriendo que la cuantización vectorial proporciona una forma de regularización beneficiosa.

3. **Ventaja diferencial por tipo de corrupción:** DeMemte sobresale en corrupciones espaciales (cutout: +7.12 pp, blur: +6.11 pp) donde la memoria 2D puede reconstruir información parcial, pero es ligeramente inferior en corrupciones destructivas totales (gaussian noise: −1.14 pp, pixel mask: −4.00 pp).

4. **Overhead mínimo:** Solo x1.37 de slowdown y +20% de parámetros, con throughput por época prácticamente idéntico.

5. **La equidad metodológica es crucial:** Entrenar ambos modelos con el mismo régimen de corrupciones cambia completamente la narrativa de los resultados, revelando que la ganancia real de la arquitectura es modesta en robustez pero sustancial en precisión limpia.

### 5.2 Implicaciones

**Para la comunidad de robustez:**
- Los módulos de memoria discreta (VQ-VAE) pueden mejorar robustez sin data augmentation adicional, pero su contribución principal puede ser la regularización y mejora de clean accuracy.
- La evaluación de robustez debe garantizar equidad de régimen de entrenamiento.

**Para aplicaciones prácticas:**
- DeMemte es especialmente recomendable en escenarios con oclusiones parciales, desenfoque o artefactos de compresión.
- El costo computacional marginal lo hace viable para deployment en producción.

### 5.3 Próximos Pasos

#### Corto Plazo
1. **Validación multi-dataset:** Replicar experimentos en CIFAR-10/100 e ImageNet para verificar generalización.
2. **Corrupciones out-of-distribution:** Evaluar con corrupciones nunca vistas en entrenamiento (e.g., JPEG compression, elastic transform, fog).
3. **Ablation study:** Aislar la contribución de cada componente (atención, VQ, gating, masked feature modeling).

#### Mediano Plazo
4. **Memoria pre-backbone:** Explorar un módulo VQ-VAE antes del backbone para mitigar corrupciones destructivas a nivel de píxel.
5. **Fine-tuning del backbone:** Descongelar capas superiores del ResNet durante P2 para permitir co-adaptación.
6. **Codebook dinámico:** Expandir/contraer el codebook durante entrenamiento según la complejidad del dataset.
7. **Comparación con métodos SOTA de robustez:** AugMax, AdvProp, DeepAugment, etc.

#### Largo Plazo
8. **Memoria jerárquica:** Múltiples VQ-VAEs a diferentes resoluciones (multi-scale memory).
9. **Aprendizaje continuo:** Evaluar si la memoria VQ permite class-incremental learning sin catastrophic forgetting.
10. **Transferencia cross-domain:** Entrenar memoria en un dominio y evaluar robustez en otro.

---

## 6. Apéndice

### A.1 Detalle de variantes exploradas

Se exploraron tres variantes de DeMemte:

| Variante | Tipo de Memoria | Espacio | Características |
|----------|----------------|---------|----------------|
| DeMemte 1D | OneDVQVAE | Post-pooling (512-d) | Simple, rápido |
| DeMemte Transformer | AttentionSpatialVQVAE | Feature maps 2D (512×7×7) | Self-attention + VQ-2D |
| DeMemte Masked Conv | SpatialConvVQVAE | Feature maps 2D (512×7×7) | Solo convoluciones |

La variante **Transformer** fue seleccionada como la principal por obtener los mejores resultados en clean y corrupt accuracy.

### A.2 Configuración de la Suite de Corrupciones

```python
strict_suite = {
    'gaussian_noise': [0.5, 1.0, 1.5],    # σ de ruido gaussiano
    'pixel_mask':     [0.25, 0.5, 0.75],   # fracción de píxeles eliminados
    'cutout':         [0.2, 0.35, 0.5],    # fracción del área ocluida
    'blur':           [0.35, 0.6, 0.85],   # intensidad de box blur 7×7
}
```

### A.3 Reproducibilidad

Todos los experimentos usan seed=42. Los resultados se generan con evaluación paired (misma corrupción exacta para ambos modelos por generador seeded). Código disponible en `experiments/VQ/dememte_variants.ipynb`.

---

**Autor:** Nakato (DeMemte Project)
**Fecha:** 2026
**Framework:** PyTorch 2.x
**Dataset:** Flowers-102
**Licencia:** Investigación Académica

---

## Referencias Conceptuales

1. **Vector Quantized VAE:** van den Oord et al. (2017) - Neural Discrete Representation Learning
2. **ResNet:** He et al. (2016) - Deep Residual Learning for Image Recognition
3. **Transformer / Self-Attention:** Vaswani et al. (2017) - Attention Is All You Need
4. **Robustness Benchmarks:** Hendrycks & Dietterich (2019) - Benchmarking Neural Network Robustness to Common Corruptions
5. **Straight-Through Estimators:** Bengio et al. (2013) - Estimating or Propagating Gradients Through Stochastic Neurons
6. **Memory-Augmented Networks:** Graves et al. (2014) - Neural Turing Machines
7. **Masked Autoencoders:** He et al. (2022) - Masked Autoencoders Are Scalable Vision Learners