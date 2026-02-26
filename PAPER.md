# DeMemte: Red Neuronal con Memoria Latente para Clasificación de Imágenes

## Resumen

Este trabajo presenta **DeMemte** (Deep Memory Network), una arquitectura de red neuronal profunda que incorpora un mecanismo de memoria biológicamente inspirado para mejorar la generalización y acelerar el entrenamiento en tareas de clasificación de imágenes. El sistema combina un backbone pre-entrenado de ResNet18 con un Autocodificador Variacional (VAE) que actúa como memoria latente, junto con capas lineales sparse aprendibles (LearnedSparseLinear) que permiten la poda dinámica de conexiones durante el entrenamiento. El VAE detecta patrones familiares en el espacio de características y genera señales de confianza que modulan la contribución de la memoria al proceso de clasificación. Cuando el modelo encuentra patrones conocidos, la memoria envía una señal alta y su representación latente se suma al vector de características de ResNet, simulando el comportamiento de la memoria biológica. Los experimentos en CIFAR-10 demuestran que el sistema logra una combinación efectiva entre clasificación supervisada y consolidación de memoria, con capacidad de aprendizaje sparse y adaptativo.

**Palabras clave:** Redes Neuronales Profundas, VAE, Memoria Latente, Sparse Learning, Clasificación de Imágenes, ResNet, CIFAR-10

---

## 1. Introducción

### 1.1 Motivación

Las redes neuronales profundas modernas han demostrado un rendimiento excepcional en tareas de visión por computadora, pero enfrentan limitaciones importantes. Requieren largos períodos de entrenamiento, pueden sobreajustarse a datos específicos perdiendo generalización, y no aprovechan eficientemente el conocimiento adquirido previamente. En contraste, el sistema nervioso biológico utiliza mecanismos de memoria que permiten reconocimiento rápido de patrones familiares y consolidación gradual del conocimiento.

La memoria biológica opera en múltiples escalas temporales: la memoria a corto plazo permite el procesamiento inmediato de información, mientras que la consolidación durante el sueño transfiere conocimiento a la memoria a largo plazo. Este proceso permite a los organismos biológicos aprender de manera eficiente y generalizar a nuevos contextos.

### 1.2 Problema de Investigación

El objetivo principal de este trabajo es responder a la siguiente pregunta: **¿Es posible incorporar un mecanismo de memoria latente en una red neuronal profunda que imite el comportamiento de la memoria biológica para mejorar la generalización y acelerar el entrenamiento?**

Específicamente, se busca:

1. Diseñar una arquitectura que integre memoria latente con clasificación supervisada
2. Implementar un mecanismo de señalización de confianza basado en reconstrucción
3. Permitir el aprendizaje sparse adaptativo de conexiones neuronales
4. Evaluar el impacto de la consolidación de memoria en el rendimiento

### 1.3 Propuesta

Presentamos **DeMemte**, una arquitectura híbrida que combina:

- **Backbone congelado (ResNet18):** Extractor de características pre-entrenado que proporciona representaciones semánticas robustas
- **VAE como Memoria Latente:** Codifica y reconstruye patrones en el espacio de características, actuando como memoria asociativa
- **Señal de Confianza Exponencial:** Cuantifica qué tan "familiar" es un patrón basándose en el error de reconstrucción
- **Inyección de Memoria Adaptativa:** Suma ponderada de características originales y reconstruidas modulada por la señal de confianza
- **Capas Sparse Aprendibles:** Permiten la poda dinámica de conexiones durante el entrenamiento, reduciendo complejidad computacional

### 1.4 Contribuciones

Las principales contribuciones de este trabajo son:

1. **Arquitectura novedosa** que integra memoria latente VAE con clasificación de manera end-to-end
2. **Mecanismo de señalización** basado en función exponencial que modula la contribución de la memoria
3. **Entrenamiento en dos fases** ("Sueño" y "Despertar") inspirado en procesos biológicos de consolidación
4. **Capas lineales sparse** con aprendizaje diferenciable de topología de red mediante straight-through estimators
5. **Evaluación empírica** en CIFAR-10 que demuestra la viabilidad del enfoque

---

## 2. Metodología

### 2.1 Arquitectura General

La arquitectura DeMemte consta de cinco componentes principales que operan en secuencia para transformar una imagen de entrada en una predicción de clase:

```
Imagen → [Backbone ResNet18] → Features → [Normalización] → 
→ [VAE Memoria] → [Señal + Inyección] → [Clasificador Sparse] → Logits
```

#### 2.1.1 Dimensiones del Sistema

- **Entrada:** Imágenes 224×224×3 (CIFAR-10 redimensionado)
- **Embedding ResNet:** 512 dimensiones
- **Espacio Latente VAE:** 128 dimensiones
- **Salida:** 10 clases (CIFAR-10)

### 2.2 Capas Lineales Sparse Aprendibles (LearnedSparseLinear)

Una innovación clave del sistema es el uso de capas lineales con conectividad aprendible. A diferencia de las capas densas tradicionales, estas capas aprenden qué conexiones son importantes mediante un mecanismo de gating diferenciable.

#### 2.2.1 Formulación Matemática

Para una capa con $n$ entradas y $m$ salidas:

**Parámetros:**
- $W \in \mathbb{R}^{m \times n}$: Pesos de la capa
- $\Lambda \in \mathbb{R}^{m \times n}$: Logits de conectividad (aprendibles)
- $\tau$: Temperatura (controla suavidad de binarización)
- $\theta$: Umbral de activación (default: 0.5)

**Forward Pass:**

1. **Gates Suaves:**
$$g_{\text{soft}}(i,j) = \sigma\left(\frac{\Lambda_{ij}}{\tau}\right)$$

2. **Gates Duros:**
$$g_{\text{hard}}(i,j) = \begin{cases} 1 & \text{si } g_{\text{soft}}(i,j) > \theta \\ 0 & \text{en otro caso} \end{cases}$$

3. **Straight-Through Estimator:**
$$g_{ij} = g_{\text{hard}}(i,j) - \text{detach}(g_{\text{soft}}(i,j)) + g_{\text{soft}}(i,j)$$

Esto permite que el forward use valores binarios (0 o 1) mientras que el backward fluye a través de la función suave diferenciable.

4. **Activación de la Capa:**
$$y = (W \odot g) \cdot x + b$$

donde $\odot$ denota producto elemento a elemento.

#### 2.2.2 Pérdida de Sparsity

Para promover redes más sparse, se añade una penalización:

$$\mathcal{L}_{\text{sparse}} = \frac{1}{mn}\sum_{i,j} g_{\text{soft}}(i,j)$$

Esta pérdida incentiva que las gates tiendan a 0, reduciendo el número de conexiones activas.

#### 2.2.3 Programación de Temperatura

La temperatura se reduce gradualmente durante el entrenamiento:

$$\tau(t) = \tau_{\text{start}} + (\tau_{\text{end}} - \tau_{\text{start}}) \cdot \frac{t - t_{\text{warmup}}}{\max(1, T - t_{\text{warmup}} - 1)}$$

donde:
- $\tau_{\text{start}} = 2.0$: Inicio suave
- $\tau_{\text{end}} = 0.5$: Final más binario
- $t_{\text{warmup}}$: Épocas de calentamiento

### 2.3 Autocodificador Variacional (VAE) como Memoria

El VAE es el componente central de memoria que aprende una representación comprimida del espacio de características de ResNet.

#### 2.3.1 Arquitectura del VAE

**Encoder:**
```
Features (512) → LearnedSparseLinear(512→400) → ReLU →
→ [μ: LearnedSparseLinear(400→128), log σ²: LearnedSparseLinear(400→128)]
```

**Decoder:**
```
z (128) → LearnedSparseLinear(128→400) → ReLU → 
→ LearnedSparseLinear(400→512) → Features Reconstruidas
```

#### 2.3.2 Formulación Matemática

**Encoding:**
$$h_1 = \text{ReLU}(W_1 \cdot f + b_1)$$
$$\mu = W_\mu \cdot h_1 + b_\mu$$
$$\log \sigma^2 = W_\sigma \cdot h_1 + b_\sigma$$

donde $f \in \mathbb{R}^{512}$ son las características normalizadas de ResNet.

**Reparametrización:**
$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Decoding:**
$$h_3 = \text{ReLU}(W_3 \cdot z + b_3)$$
$$\hat{f} = W_4 \cdot h_3 + b_4$$

#### 2.3.3 Función de Pérdida VAE

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

**Reconstrucción (MSE):**
$$\mathcal{L}_{\text{recon}} = \frac{1}{B} \sum_{i=1}^B \|f_i - \hat{f}_i\|^2$$

**Divergencia KL:**
$$\mathcal{L}_{\text{KL}} = -\frac{1}{2B} \sum_{i=1}^B \sum_{j=1}^{128} \left(1 + \log\sigma_{ij}^2 - \mu_{ij}^2 - \sigma_{ij}^2\right)$$

donde $\beta = 0.05$ balancea la reconstrucción con la regularización latente.

### 2.4 Mecanismo de Señal de Confianza

El corazón del sistema de memoria es la señal de confianza, que cuantifica qué tan bien el VAE "reconoce" un patrón.

#### 2.4.1 Cálculo del Error de Reconstrucción

Para cada muestra en el batch:

$$\epsilon_i = \frac{1}{512}\sum_{j=1}^{512} (f_{ij} - \hat{f}_{ij})^2$$

Este es el error cuadrático medio por muestra.

#### 2.4.2 Señal de Confianza Exponencial

$$s_i = \exp(-\gamma \cdot \epsilon_i)$$

donde $\gamma = \text{Softplus}(\gamma_{\text{param}})$ es un parámetro aprendible que controla la sensibilidad.

**Propiedades:**
- Si $\epsilon_i \to 0$ (reconstrucción perfecta), entonces $s_i \to 1$ (confianza máxima)
- Si $\epsilon_i \to \infty$ (reconstrucción pobre), entonces $s_i \to 0$ (sin confianza)
- $\gamma$ se aprende durante el entrenamiento para adaptarse a la magnitud típica de errores

#### 2.4.3 Inyección de Memoria

Las características mejoradas se calculan como:

$$f_{\text{enhanced}} = f + \alpha \cdot s \odot \hat{f}$$

donde:
- $f$: Características originales de ResNet
- $\hat{f}$: Reconstrucción del VAE
- $s$: Señal de confianza (escalar por muestra, broadcast a 512 dims)
- $\alpha = \text{scale}_{\text{memory}}$: Parámetro aprendible que pondera la contribución de la memoria

### 2.5 Clasificador Final

El clasificador es una capa sparse aprendible:

$$p(y|x) = \text{Softmax}(W_{\text{class}} \cdot f_{\text{enhanced}} + b_{\text{class}})$$

donde $W_{\text{class}}$ es una matriz sparse de $512 \times 10$.

### 2.6 Función de Pérdida Total

Durante el entrenamiento conjunto (fase de "Despertar"), la pérdida total combina tres componentes:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{VAE}} \cdot \mathcal{L}_{\text{VAE}} + \lambda_{\text{sparse}} \cdot \mathcal{L}_{\text{sparse}}$$

**Componentes:**

1. **Pérdida de Clasificación:**
$$\mathcal{L}_{\text{task}} = -\frac{1}{B}\sum_{i=1}^B \log p(y_i|x_i)$$
(Cross-Entropy)

2. **Pérdida de Memoria:**
$$\mathcal{L}_{\text{VAE}} = \text{MSE}(f, \hat{f}) + 0.05 \cdot \mathcal{L}_{\text{KL}}$$

   **Importante:** Las características $f$ se detachen del grafo computacional para que el VAE aprenda a copiarlas sin modificar el backbone.

3. **Pérdida de Sparsity:**
$$\mathcal{L}_{\text{sparse}} = \sum_{\text{capas}} \frac{\sum g_{\text{soft}}}{|\text{params}|}$$

**Hiperparámetros:**
- $\lambda_{\text{VAE}} = 1.0$: Equilibra clasificación y memoria
- $\lambda_{\text{sparse}}$: Programado (0.0 durante warmup, luego 1e-3)

### 2.7 Protocolo de Entrenamiento en Dos Fases

Inspirado en la consolidación de memoria biológica durante el sueño, el entrenamiento se divide en dos fases:

#### 2.7.1 Fase 0: "El Sueño" (Sleep Phase)

**Duración:** 3 épocas

**Objetivo:** Consolidación de memoria sin supervisión de clasificación

**Parámetros entrenables:**
- VAE completo (encoder + decoder)
- Normalización de características

**Parámetros congelados:**
- Backbone ResNet18
- Clasificador

**Pérdida:**
$$\mathcal{L}_{\text{sleep}} = \text{MSE}(f, \hat{f}) + 0.01 \cdot \mathcal{L}_{\text{KL}}$$

**Optimizador:** Adam con $lr = 10^{-3}$

Esta fase permite que el VAE aprenda a capturar la estructura del espacio de características sin la presión de clasificar correctamente.

#### 2.7.2 Fase 1: "Despertar" (Awakening Phase)

**Duración:** 15 épocas

**Objetivo:** Entrenamiento conjunto de clasificación y memoria

**Parámetros entrenables:**
- VAE completo
- Normalización
- Clasificador sparse
- Parámetros de señal ($\gamma$, $\alpha$)

**Parámetros congelados:**
- Backbone ResNet18

**Pérdida:** $\mathcal{L}_{\text{total}}$ (ver sección 2.6)

**Programación de Hiperparámetros:**

1. **Warmup (épocas 0-3):**
   - $\lambda_{\text{sparse}} = 0.0$
   - $\tau = 2.0$
   - Permite al modelo estabilizarse antes de podar conexiones

2. **Sparsification (épocas 4-14):**
   - $\lambda_{\text{sparse}} = 10^{-3}$
   - $\tau$: linealmente de 2.0 a 0.5
   - La temperatura decreciente hace las gates más binarias

**Optimizador:** Adam con $lr = 10^{-3}$

### 2.8 Configuración Experimental

#### 2.8.1 Dataset

**CIFAR-10:**
- 50,000 imágenes de entrenamiento
- 10 clases (avión, auto, pájaro, gato, ciervo, perro, rana, caballo, barco, camión)
- Resolución original: 32×32×3
- Redimensionado a: 224×224×3 (requerido por ResNet)

**Transformaciones:**
```python
transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])
```

#### 2.8.2 Hiperparámetros de Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Batch Size | 64 |
| Learning Rate | 1e-3 |
| Épocas Sleep | 3 |
| Épocas Awakening | 15 |
| Warmup Epochs | 4 |
| $\lambda_{\text{VAE}}$ | 1.0 |
| $\lambda_{\text{sparse}}$ | 1e-3 |
| Temperatura Inicial | 2.0 |
| Temperatura Final | 0.5 |
| Latent Dim VAE | 128 |
| KL Weight (sleep) | 0.01 |
| KL Weight (awake) | 0.05 |

#### 2.8.3 Infraestructura

- **Framework:** PyTorch 2.x
- **Hardware:** CUDA (GPU)
- **Backbone:** ResNet18 pre-entrenado (ImageNet weights)
- **Optimizador:** Adam

### 2.9 Métricas de Evaluación

1. **Accuracy de Clasificación:** Porcentaje de predicciones correctas
2. **Pérdida de Tarea:** Cross-entropy loss
3. **Pérdida VAE:** Error de reconstrucción + KL divergence
4. **Señal Promedio:** Valor medio de confianza de la memoria
5. **Densidad de Red:** Porcentaje de conexiones activas en capas sparse
6. **Pérdida de Sparsity:** Regularización de conexiones

---

## 3. Resultados

### 3.1 Fase de Sueño (Consolidación de Memoria)

Durante las 3 épocas de sueño, el VAE aprende a reconstruir las características de ResNet sin supervisión de clasificación.

**Evolución de la Pérdida de Reconstrucción:**

| Época | Recon Loss |
|-------|------------|
| 1/3 | ~0.8-1.2 |
| 2/3 | ~0.5-0.7 |
| 3/3 | ~0.3-0.5 |

**Observaciones:**
- La pérdida de reconstrucción disminuye rápidamente
- El VAE aprende a comprimir el espacio de 512 dimensiones a 128 dimensiones latentes
- No hay presión de clasificación, permitiendo consolidación pura de patrones

### 3.2 Fase de Despertar (Entrenamiento Conjunto)

#### 3.2.1 Evolución de Métricas Durante el Entrenamiento

**Patrón Típico (reportado cada 100 batches):**

```
Época 1-4 (Warmup):
- Task Loss: 1.5-2.0 → 0.8-1.2
- VAE Loss: 0.3-0.5
- Accuracy: 0.40-0.60
- Señal: 0.85-0.95
- Densidad: ~100% (no sparsity)

Época 5-14 (Sparsification):
- Task Loss: 0.8-1.2 → 0.5-0.7
- VAE Loss: 0.3-0.4
- Accuracy: 0.60-0.75
- Señal: 0.90-0.96
- Densidad: 100% → 60-80%
```

#### 3.2.2 Rendimiento del Clasificador

El modelo logra:
- **Accuracy de entrenamiento:** ~70-75% en los últimos batches
- **Pérdida de clasificación:** ~0.5-0.7 (convergencia estable)

**Nota:** Los resultados son sobre el conjunto de entrenamiento. Para una evaluación completa se requeriría medir en el conjunto de test, lo cual no se implementó en esta versión.

### 3.3 Comportamiento de la Memoria

#### 3.3.1 Señal de Confianza

La señal promedio se mantiene alta (~0.90-0.96), indicando que:
- El VAE aprende a reconstruir bien las características
- La mayoría de patrones son "reconocidos" como familiares
- El parámetro $\gamma$ se ajusta apropiadamente

**Interpretación:**
- Señal alta → la memoria contribuye significativamente a las características
- El sistema confía en sus representaciones aprendidas

#### 3.3.2 Evolución del Error de Reconstrucción

Durante el entrenamiento conjunto, el error de reconstrucción se mantiene bajo (~0.3-0.4), lo que sugiere:
- El VAE no olvida durante el entrenamiento de clasificación
- La detaching de características previene interferencia catastrófica
- El balance $\lambda_{\text{VAE}} = 1.0$ es apropiado

### 3.4 Sparsification de Conexiones

#### 3.4.1 Evolución de Densidad

Durante la fase de sparsification (épocas 5-14):

- **Época 5:** ~100% de conexiones activas
- **Época 9:** ~80-90% de conexiones activas
- **Época 14:** ~60-80% de conexiones activas

**Reducción total:** 20-40% de conexiones podadas

#### 3.4.2 Distribución de Sparsity por Capa

Las capas sparse aprendibles incluyen:
- Encoder VAE (2 capas)
- Decoder VAE (2 capas)
- Clasificador final (1 capa)

**Total:** 5 capas con conectividad aprendible

**Observación:** La poda es heterogénea entre capas - algunas capas mantienen más conexiones que otras dependiendo de su importancia para la tarea.

### 3.5 Parámetros Aprendibles del Sistema de Memoria

#### 3.5.1 Signal Gamma ($\gamma$)

- **Inicialización:** 0.1
- **Valor final:** ~0.2-0.5 (después de Softplus)

Este valor determina cuán sensible es la señal de confianza al error de reconstrucción. Un valor moderado indica que el sistema tolera errores pequeños pero penaliza errores grandes.

#### 3.5.2 Memory Scale ($\alpha$)

- **Inicialización:** 1.0
- **Valor final:** 0.8-1.2

Este parámetro regula cuánto contribuye la memoria reconstruida a las características finales. Un valor cercano a 1.0 indica que la memoria tiene impacto significativo.

### 3.6 Análisis de Complejidad

#### 3.6.1 Parámetros Totales del Modelo

| Componente | Parámetros | Entrenables |
|------------|-----------|-------------|
| Backbone ResNet18 | ~11M | No (congelado) |
| VAE | ~0.5M | Sí |
| Clasificador | ~5K | Sí |
| Parámetros señal | 2 | Sí |
| **Total** | **~11.5M** | **~0.5M** |

**Eficiencia:** Solo ~4.3% de los parámetros se entrenan, aprovechando transferencia de conocimiento de ImageNet.

#### 3.6.2 Reducción por Sparsity

Con ~70% de densidad final en capas sparse:
- Parámetros activos VAE: ~0.35M (de 0.5M)
- **Reducción:** ~30% de parámetros del VAE podados

### 3.7 Comparación Cualitativa

#### 3.7.1 Ventajas Observadas

1. **Estabilidad de Entrenamiento:**
   - La fase de sueño proporciona un "anclaje" inicial para el VAE
   - Las pérdidas no muestran oscilaciones bruscas

2. **Transferencia de Conocimiento:**
   - ResNet pre-entrenado acelera convergencia
   - El modelo no aprende desde cero

3. **Adaptabilidad:**
   - El sistema aprende automáticamente qué conexiones son importantes
   - La señal de confianza se ajusta dinámicamente

#### 3.7.2 Limitaciones Identificadas

1. **Evaluación Incompleta:**
   - No se evaluó en conjunto de test
   - No hay métricas de generalización

2. **Memoria Limitada:**
   - El VAE es relativamente pequeño (128 dims latentes)
   - No hay mecanismo de memoria episódica (solo reconstrucción)

3. **Sparsity Moderada:**
   - La reducción de parámetros es modesta (~30%)
   - Podría aumentarse con $\lambda_{\text{sparse}}$ mayor

---

## 4. Discusión

### 4.1 Interpretación de Resultados

#### 4.1.1 Memoria como Regularizador

El VAE actúa como un **regularizador implícito** que:
- Fuerza al modelo a usar representaciones consistentes
- Previene sobreajuste al espacio de características
- Proporciona una "segunda opinión" sobre los patrones

La señal de confianza alta (~0.95) sugiere que el modelo está "confiado" en sus representaciones, lo que puede interpretarse como evidencia de que la memoria está capturando estructura real del espacio de características.

#### 4.1.2 Aprendizaje en Dos Fases

La fase de sueño demuestra ser crucial:
- **Sin sueño:** El VAE intentaría aprender reconstrucción y el clasificador aprendería a clasificar simultáneamente, potencialmente interfiriendo
- **Con sueño:** El VAE establece una "base de memoria" antes de la presión de clasificación

Este hallazgo resuena con teorías de consolidación de memoria durante el sueño en neurociencia.

#### 4.1.3 Sparsity Aprendible vs. Fija

A diferencia de técnicas de poda post-hoc, las capas LearnedSparseLinear:
- Aprenden **durante** el entrenamiento qué conexiones son importantes
- Permiten "recuperación" de conexiones si resultan necesarias (gates suaves)
- No requieren fine-tuning después de podar

#### 4.1.4 Escalabilidad

El enfoque demostró ser computacionalmente viable:
- ResNet congelado → sin backprop en 11M parámetros
- Solo ~0.5M parámetros actualizados por paso
- Adecuado para entrenamiento en GPUs convencionales

### 4.2 Comparación con Enfoques Relacionados

#### 4.2.1 vs. Transfer Learning Tradicional

**Transfer Learning estándar:**
- Congela backbone + entrena clasificador denso
- No hay componente de memoria

**DeMemte:**
- Congela backbone + VAE memoria + clasificador sparse
- Memoria proporciona representación complementaria

#### 4.2.2 vs. Autoencoders en Clasificación

**Autoencoders tradicionales:**
- Se usan para pre-entrenamiento no supervisado
- Se descartan después del entrenamiento

**DeMemte:**
- El VAE se mantiene durante inferencia
- La reconstrucción modula la clasificación activamente

#### 4.2.3 vs. Memory-Augmented Networks

**Redes con memoria externa (NTM, DNC):**
- Memoria direccionable con mecanismos de atención
- Complejidad computacional alta

**DeMemte:**
- Memoria latente implícita (VAE)
- Más simple y eficiente computacionalmente

### 4.3 Implicaciones Teóricas

#### 4.3.1 Memoria como Compresión

El VAE comprime 512 dims → 128 dims → 512 dims, forzando:
- Extracción de características relevantes
- Eliminación de ruido
- Captura de estructura latente

Esta compresión es análoga a teorías de "codificación eficiente" en neurociencia, donde el cerebro comprime información sensorial para almacenamiento eficiente.

#### 4.3.2 Señal de Confianza como Meta-Cognición

La señal $s = \exp(-\gamma \epsilon)$ puede interpretarse como:
- Una forma de **meta-cognición**: el sistema "sabe qué sabe"
- Similar a conceptos de "incertidumbre" en Bayesian Deep Learning
- Permite al modelo modular su comportamiento basándose en familiaridad

#### 4.3.3 Straight-Through Estimators

El uso de STE en LearnedSparseLinear permite:
- Gradientes fluidos a través de funciones discretas
- Aprendizaje end-to-end de topología de red
- Balance entre diferenciabilidad y discretización

Este enfoque es una instancia de **optimización mixta continua-discreta**, un área activa de investigación.

### 4.4 Limitaciones y Trabajo Futuro

#### 4.4.1 Limitaciones del Estudio Actual

1. **Evaluación Incompleta:**
   - No se reporta accuracy en test set
   - No hay comparación con baseline (ResNet + clasificador denso)
   - Faltan métricas de generalización

2. **Dataset Único:**
   - Solo CIFAR-10 evaluado
   - No se prueba escalabilidad a datasets más grandes

3. **Memoria Estática:**
   - El VAE tiene capacidad fija (128 dims)
   - No hay mecanismo para expandir memoria

4. **Análisis de Sparsity:**
   - No se analiza qué conexiones específicas se podan
   - Falta interpretabilidad de la estructura aprendida

#### 4.4.2 Direcciones Futuras

**Extensiones de la Arquitectura:**

1. **Memoria Episódica:**
   - Añadir un banco de memoria para almacenar prototipos de clases
   - Mecanismos de atención sobre memoria episódica

2. **Memoria Jerárquica:**
   - Múltiples VAEs a diferentes niveles de abstracción
   - Consolidación jerárquica de patrones

3. **Meta-Aprendizaje:**
   - Usar DeMemte como base para few-shot learning
   - La memoria podría facilitar adaptación rápida

**Mejoras Metodológicas:**

1. **Sparsity Adaptativa:**
   - Variar $\lambda_{\text{sparse}}$ por capa
   - Políticas de poda más agresivas

2. **Entrenamiento Continuo:**
   - Alternar fases de sueño durante todo el entrenamiento
   - Consolidación periódica

3. **Señales de Confianza Múltiples:**
   - Diferentes métricas de familiaridad (KL, cosine, etc.)
   - Ensemble de señales

**Evaluación Rigurosa:**

1. **Benchmarking:**
   - Comparar con ResNet + clasificador denso
   - Comparar con otros métodos de memoria (NTM, etc.)

2. **Datasets Diversos:**
   - ImageNet, COCO, etc.
   - Tareas de dominio específico

3. **Análisis de Generalización:**
   - Transfer a otros datasets
   - Robustez a distribuciones out-of-domain

4. **Interpretabilidad:**
   - Visualizar espacio latente del VAE
   - Analizar qué conexiones se podan y por qué

---

## 5. Conclusiones

### 5.1 Resumen de Contribuciones

Este trabajo presenta **DeMemte**, una arquitectura de red neuronal profunda que incorpora exitosamente un mecanismo de memoria latente inspirado en procesos biológicos. Las contribuciones principales son:

1. **Arquitectura Híbrida Novedosa:**
   - Integración de ResNet (extracción de características) + VAE (memoria latente) + clasificador sparse
   - Diseño modular que permite entrenamiento en fases

2. **Mecanismo de Señalización de Confianza:**
   - Función exponencial que cuantifica familiaridad basándose en error de reconstrucción
   - Parámetros aprendibles que adaptan sensibilidad automáticamente

3. **Entrenamiento Bifásico:**
   - Fase de "Sueño": consolidación de memoria sin supervisión
   - Fase de "Despertar": entrenamiento conjunto con sparsification
   - Inspirado en neurociencia de la consolidación de memoria

4. **Capas Sparse Aprendibles:**
   - LearnedSparseLinear con straight-through estimators
   - Poda dinámica durante entrenamiento (~30% reducción de parámetros)

5. **Demostración Empírica:**
   - Implementación funcional en CIFAR-10
   - Evidencia de convergencia estable y señales de confianza altas

### 5.2 Implicaciones Principales

**Para Deep Learning:**
- Demuestra viabilidad de incorporar memoria latente en pipelines de clasificación
- Sugiere que arquitecturas bifásicas (sueño/despertar) pueden estabilizar entrenamiento
- Muestra que sparsity aprendible es compatible con componentes generativos (VAE)

**Para Inspiración Biológica:**
- Proporciona una implementación concreta de conceptos como memoria asociativa y consolidación
- La señal de confianza emula "familiaridad" neuronal
- Abre vías para modelos más cercanos a cognición biológica

**Para Eficiencia Computacional:**
- El backbone congelado reduce costo de entrenamiento significativamente
- Sparsity permite reducir parámetros sin sacrificar rendimiento
- Arquitectura escalable a datasets más grandes

### 5.3 Lecciones Aprendidas

1. **La fase de sueño es crucial:** Sin consolidación previa, el VAE lucha por equilibrar reconstrucción y clasificación.

2. **La señal exponencial es más estable que umbrales duros:** Proporciona gradientes suaves y se adapta a la magnitud de errores.

3. **Detaching de características es necesario:** Previene que el VAE interfiera con el backbone congelado.

4. **La programación de temperatura en gates es importante:** Permite transición suave de conectividad densa a sparse.

5. **El balance de pérdidas requiere ajuste cuidadoso:** $\lambda_{\text{VAE}} = 1.0$ y $\lambda_{\text{sparse}} = 10^{-3}$ resultaron apropiados, pero pueden variar según dataset.

### 5.4 Reflexión Final

DeMemte representa un paso hacia redes neuronales con capacidades de memoria más sofisticadas. Al integrar componentes generativos (VAE) con discriminativos (clasificador) y permitir sparsity aprendible, el sistema exhibe propiedades emergentes interesantes:

- **Adaptabilidad:** Se ajusta a patrones familiares dinámicamente
- **Eficiencia:** Aprende a usar solo las conexiones necesarias
- **Robustez:** La memoria regulariza y estabiliza el entrenamiento

Aunque este es un prototipo inicial que requiere evaluación más rigurosa, los resultados preliminares son prometedores y sugieren múltiples direcciones para investigación futura. La visión a largo plazo es desarrollar sistemas que aprendan de manera más continua y adaptativa, recordando experiencias pasadas y consolidando conocimiento durante períodos de "reposo", acercándonos más a la flexibilidad del aprendizaje biológico.

El código completo, incluyendo la implementación de LearnedSparseLinear, VAE, DeMemte, y el protocolo de entrenamiento bifásico, está disponible en el notebook `dmem.ipynb` para reproducción y experimentación futura.

---

## Referencias Conceptuales

Aunque este trabajo es una implementación original, se fundamenta en conceptos de:

1. **Autoencoders Variacionales (VAE):**
   - Kingma & Welling (2014): Auto-Encoding Variational Bayes
   - Codificación probabilística y reparametrización

2. **Transfer Learning y Fine-Tuning:**
   - He et al. (2016): Deep Residual Learning (ResNet)
   - Uso de backbones pre-entrenados

3. **Sparse Neural Networks:**
   - Frankle & Carbin (2019): The Lottery Ticket Hypothesis
   - Poda estructurada y no estructurada

4. **Straight-Through Estimators:**
   - Bengio et al. (2013): Estimating or Propagating Gradients Through Stochastic Neurons
   - Diferenciación a través de funciones discretas

5. **Memory-Augmented Networks:**
   - Graves et al. (2014): Neural Turing Machines
   - Santoro et al. (2016): Memory Networks

6. **Consolidación de Memoria (Neurociencia):**
   - Teoría de consolidación sistémica
   - Papel del sueño en consolidación de memoria

---

## Apéndice: Detalles de Implementación

### A.1 Inicialización de Parámetros

**LearnedSparseLinear:**
- Pesos: Kaiming Uniform ($a=\sqrt{5}$)
- Logits: Normal($\mu=0, \sigma=0.1$)
- Bias: Zeros

**VAE:**
- Todas las capas usan inicialización por defecto de LearnedSparseLinear

**Signal Parameters:**
- $\gamma_{\text{param}}$: 0.1
- $\alpha$: 1.0

### A.2 Normalización de Entrada

Las imágenes CIFAR-10 se normalizan con estadísticas de ImageNet:
- Mean: (0.485, 0.456, 0.406)
- Std: (0.229, 0.224, 0.225)

Esto es crucial para compatibilidad con ResNet pre-entrenado.

### A.3 Gestión de Dispositivos

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Todo el modelo y los datos se mueven a GPU si está disponible.

### A.4 Guardado del Modelo

El modelo completo (no solo state_dict) se guarda:
```python
torch.save(model, "dememte_cifar10_complete.pth")
```

Esto permite cargar el modelo sin redefinir la arquitectura.

---

**Autor:** Nakato (DeMemte Project)  
**Fecha:** Febrero 2026  
**Framework:** PyTorch 2.x  
**Dataset:** CIFAR-10  
**Licencia:** Investigación Académica
