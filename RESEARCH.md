## ¿Qué condiciones hacen que las ganancias de TTA reportadas sean frágiles o artefactos de hiperparámetros, y cómo se distingue una ganancia real de una ilusoria?
Las ganancias reportadas en la Adaptación en Tiempo de Prueba (TTA, por sus siglas en inglés) a menudo resultan frágiles o son simples artefactos de hiperparámetros debido a la falta de estandarización en las evaluaciones y a las condiciones en las que se despliegan. 

Las principales condiciones que provocan esta fragilidad son:

*   **Dependencia del lote (Batch Dependency):** En la configuración de adaptación en línea, los modelos acumulan conocimiento del historial de lotes de prueba, lo que crea una dependencia. Una elección inadecuada de hiperparámetros (como la tasa de aprendizaje o múltiples pasos de adaptación) puede provocar una degradación considerable del rendimiento a medida que avanza la TTA. Sorprendentemente, utilizar una selección de modelos "oráculo" (que asume acceso a las etiquetas verdaderas para elegir el mejor modelo) puede exacerbar este problema al sobreajustar el modelo a los lotes ya vistos.
*   **Sensibilidad extrema a los hiperparámetros:** La eficacia de los métodos TTA depende en gran medida de sus hiperparámetros. En la práctica, es extremadamente difícil ajustarlos correctamente porque no se dispone de etiquetas de validación ni se conoce la estructura del cambio de distribución (distribution shift) de antemano. Una mala elección puede reducir la precisión de manera drástica.
*   **Calidad y entrenamiento del modelo base:** El grado de mejora depende fuertemente de la calidad del modelo pre-entrenado. Paradójicamente, **las prácticas recomendadas de aumento de datos (data augmentation) utilizadas para mejorar la generalización fuera de distribución (OOD) pueden tener un efecto inverso o marginal** cuando se combinan con TTA.
*   **Vulnerabilidad a cambios de distribución específicos:** Incluso con hiperparámetros óptimos, los métodos TTA fracasan sistemáticamente o empeoran el rendimiento frente a ciertos cambios de distribución que son comunes en el mundo real, tales como los cambios de correlación espuria, los cambios en la distribución de etiquetas (label shifts) o los flujos de datos no estacionarios.

**Para distinguir una ganancia real de una ilusoria**, se pueden utilizar las siguientes estrategias prácticas y de evaluación:

*   **Aprovechar los fenómenos "Agreement-on-the-Line" (AGL) y "Accuracy-on-the-Line" (ACL):** Se ha descubierto que los modelos sometidos a TTA muestran una correlación lineal extremadamente fuerte entre su precisión y nivel de acuerdo (agreement) en datos dentro de la distribución (ID) frente a datos fuera de la distribución (OOD). Esta tendencia se mantiene constante a través de distintos hiperparámetros y diferentes arquitecturas de red.
*   **Estimación de la precisión OOD sin etiquetas:** Al explotar los fenómenos AGL y ACL mediante técnicas de estimación (como ALine-S y ALine-D), **es posible predecir con alta precisión el rendimiento OOD de un modelo adaptado sin necesidad de utilizar datos de prueba etiquetados**. Las tendencias de estas estimaciones reflejan de manera fiel cuándo la TTA realmente mejora la generalización y cuándo la degrada, permitiendo desenmascarar ganancias ilusorias.
*   **Selección confiable de hiperparámetros:** En lugar de afinar hiperparámetros a ciegas en el dominio de prueba (lo cual genera artefactos), se puede identificar una ganancia genuina seleccionando los hiperparámetros que producen el modelo con **el mejor rendimiento en los datos ID** (donde sí hay etiquetas). Gracias al fenómeno de correlación lineal, los modelos seleccionados de esta manera logran una precisión OOD casi óptima, comparable a la que se obtendría si se tuvieran las etiquetas reales.
*   **Evaluación rigurosa en benchmarks estandarizados:** Para validar que las mejoras no son un artefacto, los algoritmos deben probarse en plataformas amplias y transparentes (como TTAB), que evalúan los modelos de manera justa sobre diversos tipos de cambios de distribución bajo protocolos estrictos, sin la trampa de la selección de modelos oráculo.


## ¿Qué métricas, además de accuracy, se usan para mostrar que un mecanismo de adaptación se activa aunque no cambie la predicción? (busco vocabulario para reportar mi "hito mecánico vs numérico")

Para reportar un "hito mecánico vs. numérico", donde necesitas demostrar que el engranaje interno del modelo se está ajustando a pesar de que la predicción final (la clase elegida) siga siendo la misma, las investigaciones en Adaptación en Tiempo de Prueba (TTA) utilizan diversas métricas "bajo el capó". 

Estas métricas demuestran que las probabilidades subyacentes, la estructura del espacio latente o las dinámicas de los gradientes están reaccionando de manera efectiva. A continuación, te presento el vocabulario y las métricas estructuradas por categorías:

### 1. Métricas de Calibración, Confianza y Probabilidad
Incluso si el modelo sigue prediciendo "gato" (no hay cambio numérico en el *accuracy*), la fuerza o forma de esa predicción puede estar mejorando (hito mecánico).
*   **Cambio en la Entropía ($\Delta H$) y Minimización de Entropía:** Mide la reducción de la incertidumbre en la salida del modelo. Una caída en la entropía indica que el mecanismo se activó y el modelo se volvió más seguro de sus predicciones (reduciendo la ambigüedad en el vector *softmax*), incluso si la clase ganadora no cambia.
*   **Brier Score:** Es una regla de puntuación estrictamente propia que mide la precisión de las probabilidades pronosticadas (no solo la clase final). Combina tanto el error de clasificación como la calibración del modelo en una sola métrica.
*   **Error de Calibración Esperado (ECE - Expected Calibration Error) y Error Máximo de Calibración (MCE):** Miden qué tan bien las probabilidades predichas por el modelo reflejan la verdadera probabilidad de acierto. TTA a menudo ajusta severamente estas métricas y reduce la sobreconfianza (o infra-confianza) del modelo.
*   **Puntuación de Energía (Energy Score) o Probabilidad Máxima de Clase:** Sirven como "proxies" de la pérdida (loss) o incertidumbre del modelo sin necesidad de conocer las etiquetas reales, observando solo cómo se comportan los logits.

### 2. Métricas de Dinámica de Optimización y Gradientes
Estas métricas prueban empíricamente que los pesos del modelo están recibiendo retroalimentación activa del nuevo dominio.
*   **Norma del Gradiente (Gradient Norm - $\| \nabla_{\theta}\mathcal{L} \|$):** Cuantifica qué tan "sensible" o inestable está siendo el modelo frente a las muestras de prueba. Una fluctuación o estabilización de las normas de los gradientes indica que la red está reaccionando a los datos entrantes o a los valores atípicos (outliers).
*   **Efectividad de Optimización (Similitud del Coseno de Gradientes):** Mide la alineación geométrica (similitud del coseno) entre el gradiente que el modelo está usando para adaptarse y un "gradiente ideal" u oráculo. Demuestra empíricamente que la dirección de la adaptación es matemáticamente correcta.
*   **Divergencia Kullback-Leibler (KL Drift):** Evalúa cómo la distribución de creencias predictivas del modelo ($p_t$) se ha desplazado o alejado de una distribución de referencia histórica o previa a la perturbación ($q_t$).

### 3. Métricas de Estabilidad y Consistencia de Predicción
Estas evalúan qué tan resistente se vuelve la representación interna del modelo ante perturbaciones de la misma imagen, lo cual ocurre antes de que mejore el *accuracy* general.
*   **Agreement (Nivel de Acuerdo):** Mide la proporción de veces que dos versiones del modelo (por ejemplo, antes y después de TTA, o bajo distintas configuraciones) toman exactamente la misma decisión sobre un conjunto de datos. Este concepto está ligado a fenómenos estructurales muy fuertes llamados *Agreement-on-the-Line (AGL)*, los cuales permiten estimar el rendimiento de la adaptación sin usar etiquetas reales.
*   **Diferencia de Probabilidad de Pseudo-Etiquetas (PLPD - Pseudo Label Probability Difference):** Mide la variación en la probabilidad de una predicción antes y después de aplicarle pequeñas perturbaciones o transformaciones (por ejemplo, alterar los píxeles o cambiar parches espaciales). Un PLPD estable muestra que el modelo aprendió a ignorar el ruido (hito mecánico).
*   **Diferencia bajo Transformaciones Semánticas ($Diff(\hat{y}, \hat{y}^{sp})$):** Similar al anterior, compara la salida de los vectores de probabilidad de la imagen original frente a versiones con transformaciones que preservan la semántica o la alteran, para determinar si una muestra es "confiable" para activar la adaptación.

### 4. Métricas a Nivel de Características (Espacio Latente)
Prueban que las representaciones internas de la red se están separando de manera más limpia, aunque el clasificador final en la última capa aún cometa el mismo error.
*   **Métricas de Clustering (Calidad del Espacio Latente):** Se usan índices como el **Silhouette Score**, el **Índice de Davies-Bouldin**, y el **Índice de Calinski-Harabasz** para evaluar qué tan densos y bien separados están los clústeres de características internamente. Un incremento en el Silhouette Score significa que el modelo está organizando mejor la información bajo el capó.
*   **Distancia al Prototipo de Clase:** Es la distancia en el espacio latente entre las características de la muestra de prueba y el vector representativo (prototipo) de la clase más cercana ($\min_c \|f(x) - w_c\|_2^2$).

**Vocabulario sugerido para tu reporte:**
Puedes enmarcar tu hito bajo la frase: *"Aunque la precisión nominal (Numerical Accuracy) se mantiene constante, observamos un **ajuste mecánico significativo (Mechanical Alignment)** evidenciado por la compresión en la métrica de [Brier Score / Entropía / Norma del Gradiente], indicando que el mecanismo de adaptación ha entrado en régimen operativo y está reestructurando activamente la confianza del espacio latente."*

## ¿Cómo se evalúa TTA de forma rigurosa para evitar conclusiones engañosas (resets, semillas, orden de batches)?
Para evaluar la Adaptación en Tiempo de Prueba (TTA) de forma rigurosa y evitar conclusiones engañosas o lo que la literatura llama "ilusión de progreso", las evaluaciones deben aislar y estandarizar cuidadosamente múltiples factores operacionales. Los métodos TTA son extremadamente sensibles a las condiciones experimentales, y obviar estas configuraciones invalida las comparaciones entre diferentes algoritmos.

A continuación, se detallan las prácticas de evaluación rigurosa para abordar resets, semillas, orden de datos y otros factores críticos:

### 1. Transparencia en los "Resets" (Protocolo Episódico vs. En Línea/Continuo)
La literatura actual clasifica la evaluación en diferentes protocolos dependiendo de si el modelo borra o acumula su memoria a corto plazo. Es metodológicamente incorrecto comparar un método evaluado con *resets* frente a uno sin ellos.
*   **Adaptación Episódica (con resets):** El modelo se adapta a un solo lote de prueba o muestra, emite la predicción y luego **se reinicia a los pesos del modelo preentrenado original** antes de recibir el siguiente lote. Esto elimina la "dependencia del lote" y aísla la evaluación, garantizando mejoras estables pero limitadas.
*   **Adaptación En Línea o Continua (sin resets):** El modelo se adapta secuencialmente lote tras lote y acumula actualizaciones a lo largo del tiempo, sin volver a su estado base original. Esta configuración ("lifelong" o TTA continuo) es mucho más realista, pero sufre de olvido catastrófico y acumulación de errores al enfrentar largos flujos de datos o cambios bruscos de dominio. En algunos benchmarks intermedios, el modelo solo se reinicia al transicionar entre diferentes dominios de prueba (ej. de ruido gaussiano a lluvia), mientras que una evaluación estricta continua nunca lo reinicia.

### 2. Control del Orden de los Batches y Semillas Aleatorias (Seeds)
La dependencia del orden en el que llegan los datos es un problema profundo en TTA. Para evaluar robustez, se aplican dos estrategias clave respecto a las semillas y el orden de los datos:
*   **Múltiples semillas para el barajado:** Un algoritmo riguroso debe probarse bajo múltiples semillas aleatorias (random seeds) que mezclen los lotes de prueba en distintos órdenes, reportando la media y desviación estándar de su rendimiento. Esto asegura que la mejora no sea un artefacto de una secuencia afortunada y demuestra la estabilidad estadística del método.
*   **Desplazamientos no estacionarios y desequilibrados:** Las evaluaciones rigurosas o "Wild TTA" modifican intencionalmente el orden de los lotes para simular **distribuciones de etiquetas desequilibradas en línea**. En lugar de datos barajados uniformemente, introducen los datos ordenados por clases o con proporciones fuertemente sesgadas (ej. presentando secuencialmente primero muestras de una clase y luego de otra) para comprobar si la optimización direcciona al modelo hacia un colapso.

### 3. Evaluación frente a Tamaños de Lote (Batch Sizes) Reducidos
Muchos métodos (especialmente los que actualizan estadísticas de Batch Normalization) reportan ganancias usando tamaños de lote muy grandes (ej. 64, 128 o 200) para estabilizar la adaptación. 
*   **Pruebas de estrés de Batch Size:** Una evaluación rigurosa expone intencionalmente al modelo a lotes pequeños (ej. tamaño de lote 1, 2 o 4) porque revela la inestabilidad oculta de la actualización y es más fiel a configuraciones de latencia en tiempo real o de dispositivos de borde (edge). TTA suele colapsar catastróficamente bajo tamaños de lote pequeños a menos que use capas de normalización agnósticas al lote (como Group Norm o Layer Norm) o incorpore regularización robusta.

### 4. Selección de Hiperparámetros y Eliminación del "Oráculo"
Es común el error de afinar hiperparámetros de TTA usando métricas que en la práctica estarían ocultas. 
*   **El peligro del Oráculo:** Intentar seleccionar el modelo u optimizar la tasa de aprendizaje mediante un "oráculo" (que mira las etiquetas verdaderas del dominio de prueba durante la adaptación en línea) es una trampa. Paradójicamente, en una configuración en línea continua, usar el oráculo exacerba la dependencia de lotes pasados y hace que el rendimiento del modelo se desplome rápidamente.
*   **Protocolos sin etiquetas:** Una buena metodología debe dictar un esquema claro y ciego de ajuste de hiperparámetros. Las soluciones validadas incluyen seleccionar aquellos hiperparámetros que logren el mejor rendimiento en los **datos fuente originales / dentro de la distribución (ID)**, o usar cuadrículas de búsqueda estandarizadas compartidas entre todos los métodos sin trampa.

### 5. Estandarización de la Calidad del Modelo Base
La eficacia de los métodos de adaptación no solo depende del algoritmo TTA, sino también de la calidad del extractor de características preentrenado. Algunos métodos parecen mejores simplemente porque en su artículo original modificaron su esquema de preentrenamiento. Por ello, librerías estandarizadas como *TTAB (Test-Time Adaptation Benchmark)* exigen que las evaluaciones se realicen **forzando a todos los algoritmos a adaptarse sobre los mismos modelos preentrenados, utilizando exactamente las mismas arquitecturas y optimizadores**. Además, advierten que trucos populares de generalización en entrenamiento (como AugMix o PixMix) pueden limitar o generar efectos marginales en el paso de adaptación durante TTA.


## ¿Qué condiciones hacen que la adaptación en test-time no mejore o degrade la accuracy aunque la pérdida de adaptación sí cambie los parámetros? Lista los modos de fallo documentados y a qué los atribuyen (batch size, número de pasos, hiperparámetros).

La adaptación en tiempo de prueba (TTA) puede reducir matemáticamente la función de pérdida (como la minimización de entropía) mientras degrada catastróficamente la precisión real. Esto ocurre porque la optimización no supervisada, al carecer de etiquetas reales, a menudo encuentra **soluciones degeneradas o triviales**: el modelo altera sus parámetros para estar 100% "seguro" de sus predicciones (entropía cero), pero de forma incorrecta, destruyendo la representación semántica subyacente.

A continuación, se detallan los modos de fallo documentados y las condiciones a las que se atribuyen:

**1. Colapso del Modelo (Model Collapse)**
El modelo comienza a predecir una única clase constante para todas las entradas, independientemente de la imagen, logrando una pérdida de entropía mínima pero una precisión cercana a cero.
*   **Atribuido a (Hiperparámetros - Tasa de aprendizaje alta):** Una tasa de aprendizaje demasiado alta (ej. $\eta=0.1$) provoca un sobreajuste rápido al lote actual, destruyendo el conocimiento previo y forzando al modelo a colapsar en una sola clase.
*   **Atribuido a (Gradientes ruidosos):** Muestras de prueba atípicas o ruidosas generan picos masivos en la norma de los gradientes que desestabilizan los pesos. Inmediatamente después de este pico, el gradiente cae a casi cero, confirmando que el modelo ha colapsado en un mínimo trivial.

**2. Inestabilidad por Tamaños de Lote Pequeños (Small Batch Size)**
La adaptación fluctúa bruscamente o empeora el rendimiento del modelo base cuando los datos llegan en lotes muy pequeños (ej. 1, 2 o 4 muestras).
*   **Atribuido a (Batch Size / Normalización):** Los métodos TTA tradicionales dependen de actualizar las estadísticas (media y varianza) de las capas de Normalización por Lotes (Batch Normalization). Con lotes pequeños, estas estimaciones son extremadamente inexactas y sesgadas, destrozando la representación de las características. Aunque usar normas independientes del lote (como Group Norm o Layer Norm) mitiga esto, la optimización por entropía en estos modelos sigue siendo propensa al colapso bajo desplazamientos severos.

**3. Acumulación de Errores y Dependencia del Lote**
El modelo mejora inicialmente, pero su rendimiento cae en picada a medida que avanza el flujo de datos.
*   **Atribuido a (Número de Pasos):** Aumentar excesivamente el número de pasos de adaptación por cada lote genera un sobreajuste a ese lote específico (dependencia del lote), impidiendo que el modelo generalice a los siguientes.
*   **Atribuido a (Falta de Resets):** En horizontes largos, las actualizaciones continuas acumulan sesgos y errores. Sin un mecanismo adecuado para reiniciar (resetear) los parámetros al estado original o regularizarlos, el modelo sufre de "olvido catastrófico" (catastrophic forgetting), perdiendo su precisión tanto en los datos nuevos como en los originales (in-distribution).

**4. Fracaso ante Desplazamientos de Etiquetas (Label Shifts) y Lotes Desequilibrados**
El rendimiento de TTA cae por debajo del modelo sin adaptar cuando la proporción de clases en el dominio de prueba cambia drásticamente o cuando un lote está dominado por una sola clase.
*   **Atribuido a (Batch Size y Distribución):** Al recalcular las estadísticas de normalización sobre un lote desequilibrado, el modelo sesga artificialmente sus predicciones hacia la clase mayoritaria de ese lote. 

**5. Degradación por Optimización Ciega de Alta Entropía**
Actualizar el modelo usando todas las muestras disponibles suele perjudicar la precisión global.
*   **Atribuido a (Calidad de la Muestra / Hiperparámetros de filtrado):** Las muestras con una entropía muy alta son inherentemente inciertas. Obligar al modelo a minimizar la entropía sobre estas muestras produce gradientes sesgados y poco confiables que desorientan la optimización direccional de los pesos. 

**6. Efectos Inversos con Aumentos de Datos Robustos (Data Augmentation)**
Paradójicamente, pre-entrenar el modelo con técnicas avanzadas de aumento de datos diseñadas para mejorar la robustez (como AugMix o PixMix) limita severamente las ganancias de TTA.
*   **Atribuido a (Calidad del modelo base):** Estos modelos ya poseen una generalización fuera de distribución (OOD) muy alta. Como resultado, las señales de adaptación no supervisada no logran encontrar direcciones de optimización útiles, produciendo efectos marginales o incluso perjudiciales durante el test-time en comparación con modelos entrenados sin aumentos.


## ¿Qué parámetros o capas eligen adaptar los distintos métodos de test-time adaptation (normalización, prompts, features, codebook) y qué justificación dan para que esa elección propague gradiente hacia la salida?

La elección de qué parámetros o capas adaptar durante la Adaptación en Tiempo de Prueba (TTA) depende fundamentalmente de un compromiso entre la eficiencia computacional, la estabilidad para evitar el "olvido catastrófico" (catastrophic forgetting) y el nivel de acceso que se tenga al modelo (caja blanca, gris o negra). 

A continuación, detallo qué parámetros eligen adaptar los distintos métodos y la justificación mecánica de cómo esta elección permite una correcta optimización y propagación del gradiente:

### 1. Capas de Normalización (BatchNorm, LayerNorm, GroupNorm)
Esta es la estrategia más clásica y predominante en TTA (usada por métodos como TENT, EATA, SAR y SoTTA).
*   **Qué adaptan:** Únicamente los parámetros afines (escala $\gamma$ y desplazamiento $\beta$) y las estadísticas de normalización (media y varianza) de las capas de normalización, manteniendo congelado el resto del modelo.
*   **Justificación para el gradiente:** 
    *   Se asume que la mayor parte del "conocimiento de la tarea" reside en los pesos de las capas convolucionales o densas, mientras que el "conocimiento del dominio" o estilo se captura en las estadísticas de normalización. 
    *   Desde la perspectiva de la optimización, las transformaciones afines son lineales y operan canal por canal (channel-wise feature modulation). Esto crea una ruta de gradiente directa y de muy baja dimensionalidad (los parámetros afines representan menos del 1% del total del modelo), lo que permite que el gradiente retropropague de forma extremadamente eficiente.
    *   Limitar la propagación del gradiente solo a estas capas previene que la optimización basada en métricas no supervisadas (como la entropía) cause que el modelo diverja catastróficamente de su entrenamiento original.

### 2. Prompts de Entrada (Input Prompts)
Usado por métodos para modelos Visión-Lenguaje (VLM) como TPT, y métodos de caja negra o caja gris como BETA y FOA.
*   **Qué adaptan:** En lugar de modificar los pesos internos, añaden un "prompt" visual (un patrón de píxeles o borde añadido a la imagen) o un prompt textual (vectores de contexto en el espacio de embedding) que es optimizable.
*   **Justificación para el gradiente:**
    *   **En modelos de Caja Blanca (White-Box):** El gradiente calculado a partir de la pérdida de entropía se propaga por toda la red hasta llegar al espacio de entrada (los píxeles del prompt) para actualizarlos, dejando el modelo general intacto y preservando su conocimiento general (como hace TPT).
    *   **En modelos de Caja Negra (Black-Box APIs):** Dado que la retropropagación no es posible a través de un sistema opaco, métodos como BETA introducen un **modelo guía (steering model) local y ligero**. El gradiente se calcula utilizando la ruta diferenciable del modelo guía (que estima el error respecto a la predicción del modelo de caja negra) y se usa para optimizar el prompt visual antes de enviarlo a la API. Otros enfoques usan **Optimización de Orden Cero (ZOO)**, estimando gradientes direccionales perturbando aleatoriamente el prompt y midiendo cómo reacciona la salida del modelo sin usar retropropagación real.

### 3. Adaptadores Intermedios (Meta-networks) y Clasificador Final
Para situaciones que requieren eficiencia extrema de memoria o configuraciones sin retropropagación.
*   **Qué adaptan:** Redes auxiliares ("meta networks") inyectadas en las capas tempranas del modelo (ej. EcoTTA), o simplemente los prototipos del clasificador final lineal (ej. T3A, STAD).
*   **Justificación para el gradiente:** 
    *   En métodos como EcoTTA, actualizar toda la red requiere guardar mapas de activación masivos durante el *forward pass* para calcular los gradientes en el *backward pass*, lo cual agota la memoria de dispositivos pequeños. Al adaptar solo pequeñas meta-redes en las primeras capas, la retropropagación se acorta drásticamente, ahorrando memoria y previniendo la acumulación de errores.
    *   Para el clasificador final (T3A), el método es directamente libre de retropropagación (backpropagation-free). Modifican la decisión ajustando "pseudo-prototipos" basados en la media de las características extraídas, en lugar de calcular gradientes.

### 4. Codebook (Quantización de Vectores - VQ)
Aunque usualmente visto en arquitecturas de representación y generativas, en adaptaciones VQ la selección del parámetro a optimizar es muy particular.
*   **Qué adaptan:** Los vectores discretos dentro del diccionario de códigos (codebook) a los cuales las representaciones latentes continuas son asignadas.
*   **Justificación para el gradiente:** La operación de asignar un vector a su vecino más cercano en el codebook (usando `argmin`) corta abruptamente el flujo computacional porque es una función escalonada (no diferenciable). Para permitir que el gradiente se propague hacia la salida (o de vuelta al codificador), estos modelos utilizan el **Straight-Through Estimator (STE)**. Esta técnica es un "truco" matemático que copia directamente los gradientes desde el decodificador, pasa por encima de la capa discreta no diferenciable, y los entrega intactos al codificador. Además, para actualizar los propios vectores del codebook, los gradientes a veces se manipulan inyectando estimaciones suavizadas de las variaciones del codificador (como en NS-VQ), forzando a que códigos no usados reciban información del gradiente basándose en su distancia a la muestra y evitando el colapso del codebook.

## ¿Cómo construyen los métodos de clasificación aumentada por recuperación (cache models, kNN, memory banks) su predicción a partir de la memoria, y cómo combinan el score de retrieval con el logit del clasificador base?

Los métodos de clasificación aumentada por recuperación (como los *cache models*, kNN y bancos de memoria) operan almacenando representaciones de datos conocidos y utilizando métricas de similitud para mejorar o adaptar las predicciones del modelo base. 

A continuación se detalla cómo construyen su predicción y cómo la combinan con el clasificador base:

### 1. Construcción de la predicción a partir de la memoria
El proceso de recuperación se basa en un sistema de claves y valores (*key-value*) que sigue estos pasos:

*   **Construcción de la memoria:** El modelo procesa un conjunto de datos (por ejemplo, ejemplos de entrenamiento *few-shot*) utilizando un codificador preentrenado. Las características extraídas (vectores continuos) se almacenan como **claves (*keys*)**, y sus etiquetas correspondientes (generalmente codificadas en formato *one-hot*) se almacenan como **valores (*values*)**.
*   **Consulta (*Querying*):** Durante la inferencia, la imagen o dato de prueba pasa por el mismo codificador para extraer su vector de características, el cual actúa como una **consulta (*query*)**.
*   **Cálculo de similitud (Retrieval):** El modelo calcula la distancia o similitud entre la consulta y las claves en la memoria para encontrar los $K$ vecinos más cercanos. Las métricas más comunes son la similitud del coseno o la distancia euclidiana.
*   **Agregación de la predicción:** Una vez recuperados los vecinos, la predicción de la memoria se puede calcular de diferentes maneras:
    *   **Ponderación por afinidad:** Las similitudes se transforman en pesos. Por ejemplo, *Tip-Adapter* utiliza una función exponencial sobre la similitud del coseno para calcular la afinidad ($A = \exp(-\beta(1 - \text{similitud}))$), modulando su nitidez con un hiperparámetro $\beta$. La predicción de la memoria se obtiene combinando linealmente los valores (etiquetas *one-hot*) ponderados por estas afinidades. Otro ejemplo es usar una función kernel sobre la distancia euclidiana para asignar mayor peso a las memorias más cercanas.
    *   **Votación mayoritaria:** En enfoques más directos (como *SPARK-IL*), se recuperan los $K$ vecinos más cercanos y la predicción final se determina simplemente por un voto mayoritario sobre las etiquetas de dichos vecinos.

### 2. Combinación del *score* de retrieval con el clasificador base
Dependiendo de la arquitectura, la predicción obtenida de la memoria se fusiona con las predicciones del modelo base (paramétrico) utilizando distintas estrategias:

*   **Conexión residual (Suma de Logits):** En enfoques como *Tip-Adapter*, los *logits* generados por el modelo de caché se combinan de forma lineal con los *logits* del clasificador base original (conocimiento previo). La fórmula típicamente se define como $\text{logits} = \alpha \cdot \text{predicción\_memoria} + \text{logits\_base}$, donde $\alpha$ es un ratio residual que controla cuánto peso se le da al conocimiento recuperado frente al conocimiento preentrenado del clasificador. Ajustar $\alpha$ permite equilibrar la importancia de ambas fuentes.
*   **Modelos de Mezcla (*Mixture Models* a nivel de probabilidad):** Otros métodos combinan las decisiones después de que los *logits* se han convertido en distribuciones de probabilidad. La predicción final se interpola linealmente combinando la probabilidad del modelo base y la del modelo de memoria: $p(y|q) = \lambda p_{param}(y|q) + (1-\lambda) p_{mem}(y|q)$, donde $\lambda$ controla la contribución de cada modelo.
*   **Adaptación local de parámetros (Alternativa indirecta):** En lugar de sumar puntuaciones directamente, métodos como *Memory-based Parameter Adaptation (MbPA)* utilizan el contexto recuperado de la memoria (los $K$ vecinos y sus pesos) para definir una función de pérdida temporal. El modelo base da pasos de gradiente rápido (*gradient descent*) para adaptar localmente sus propios parámetros a los ejemplos recuperados antes de emitir los *logits* finales. Una vez hecha la predicción, los cambios en los pesos se descartan para evitar el sobreajuste y el olvido catastrófico.


## Según la teoría de las Modern Hopfield Networks, ¿cuál es la capacidad de almacenamiento y bajo qué condiciones la recuperación converge a un patrón único frente a una mezcla metaestable de patrones? Da la ecuación de actualización y el rol de la temperatura β.
**Capacidad de Almacenamiento:**
Las Modern Hopfield Networks rompen la limitación de escalado lineal de la red clásica de Hopfield introduciendo no linealidades más fuertes que otorgan a la red una capacidad de memoria inmensamente mayor:
*   **Capacidad Polinomial:** Cuando se utiliza una función de energía polinomial $F(x) = x^n$, la capacidad máxima de almacenamiento de memorias sin error crece de forma polinomial de manera proporcional a $d^{n-1}$ (donde $d$ o $N_f$ es la dimensión del espacio de patrones).
*   **Capacidad Exponencial:** Al emplear una función de energía exponencial $F(x) = e^x$ en variables discretas, la capacidad de memoria se vuelve exponencial, permitiendo almacenar hasta $\approx 2^{d/2}$ patrones. 
*   Para las Modern Hopfield Networks de **estados continuos** operando con patrones aleatorios en una esfera, está teóricamente probado que el número de patrones que pueden almacenarse escala de manera proporcional a $c^{\frac{d-1}{4}}$ (donde $c$ es una constante dependiente de los hiperparámetros del sistema), manteniendo su capacidad de almacenamiento exponencial respecto a la dimensión.

**Ecuación de Actualización:**
Para una red de Hopfield moderna con estados continuos, la actualización del vector de estado $ξ$ (la consulta o *query*) frente a los patrones almacenados en una matriz $X$ se define globalmente para minimizar la energía con la siguiente ecuación:
**$$ξ^{new} = X \text{softmax}(\beta X^T ξ)$$**.
Sorprendentemente, esta regla de actualización matemática es exacta y funcionalmente **equivalente al mecanismo de atención (*key-value attention*) de los modelos Transformer**.

**Condiciones de Convergencia (Patrón Único vs. Mezcla Metaestable):**
La convergencia a un punto u otro depende fundamentalmente de la **separación $\Delta_i$**, que mide la diferencia entre el producto punto de un patrón consigo mismo y su producto punto máximo con cualquier otro patrón en la memoria ($\Delta_i = x_i^T x_i - \max_{j \neq i} x_i^T x_j$).
*   **Convergencia a un patrón único:** Si un patrón está "bien separado" de los demás (es decir, $\Delta_i$ es suficientemente grande frente a un umbral teórico), la regla de actualización converge a un punto fijo fuertemente aislado que representa a ese único patrón. Típicamente esto ocurre en una sola iteración de la actualización, logrando un error de recuperación exponencialmente pequeño.
*   **Convergencia a una mezcla metaestable:** Si un grupo de patrones son muy similares entre sí (no están bien separados internamente) pero el grupo en su conjunto está bien separado del resto de patrones externos, se genera un **estado metaestable**. Si el estado inicial inicia cerca de esta vecindad, la actualización convergerá a este estado, el cual es equivalente a una mezcla o promedio aritmético local de dichos patrones similares.
*   *Nota adicional:* Si ningún patrón en absoluto está bien separado, el sistema converge a un punto fijo global equivalente a la media aritmética de *todos* los vectores almacenados.

**El Rol de la Temperatura $\beta$:**
El hiperparámetro $\beta$ modula la agudeza (*sharpness*) de la función softmax, y sirve como factor de escala o parámetro de temperatura.
*   Su rol principal es **gobernar la dinámica de los puntos fijos** y determinar el "tamaño" o naturaleza de los estados metaestables.
*   **Valores bajos de $\beta$** inducen una distribución más uniforme en la función softmax, lo cual fuerza al sistema a realizar un promediado global sobre una gran cantidad de patrones (favoreciendo el punto fijo global).
*   **Valores altos de $\beta$** permiten que la red cree estados metaestables pequeños o aísle patrones únicos de forma nítida, ya que controlan exactamente sobre cuántos patrones similares se debe promediar.
*   A diferencia de los Transformers convencionales, donde suele estar bloqueado a $\beta = 1/\sqrt{d_k}$, en las capas basadas en redes de Hopfield, $\beta$ puede configurarse libremente o aprenderse para dictar la cantidad deseada de recuperación o mezcla sin cambiar la arquitectura de la red.

## ¿Cómo se gestionan el tamaño, la escritura y el olvido de un buffer episódico de plasticidad rápida en sistemas de memoria dual (rápido/lento)? ¿Qué regla de consolidación usan?

En los sistemas de memoria dual, que se inspiran en la teoría de los Sistemas de Aprendizaje Complementarios (CLS), coexisten un sistema lento encargado de adquirir gradualmente conocimiento estructurado y un sistema rápido (el búfer episódico) que permite el aprendizaje veloz de experiencias individuales específicas. 

El manejo de este búfer rápido y la consolidación del conocimiento se gestionan de la siguiente manera:

### Gestión del tamaño, escritura y olvido

*   **Escritura (Almacenamiento):** El búfer episódico funciona como un módulo que almacena experiencias pasadas utilizando un formato de diccionarios con **pares de claves y valores**. Las claves (*keys*) suelen ser las representaciones latentes o incrustaciones (*embeddings*) extraídas por la red ante una entrada, mientras que los valores (*values*) corresponden a los objetivos o etiquetas de la tarea. La escritura se realiza mediante un proceso de **agregación continua**: cada vez que el sistema percibe un nuevo ejemplo, extrae sus características y añade el nuevo par directamente a la memoria.
*   **Tamaño:** Aunque la cantidad de elementos crece dinámicamente conforme llegan datos, el búfer episódico está diseñado con una **capacidad máxima fija**. 
*   **Olvido (Reemplazo de datos):** La gestión primaria del olvido en el modelo básico opera como un **búfer circular**. Cuando la memoria alcanza su límite de capacidad, el sistema **sobrescribe primero los datos más antiguos** para dejar espacio a las nuevas entradas. En variantes más complejas adaptadas a flujos de datos cambiantes (como en la adaptación continua en tiempo de prueba), la memoria emplea un mecanismo basado en la equidad de clases: si el búfer se llena, el sistema descarta de forma aleatoria una muestra que pertenezca a la categoría con mayor presencia, evitando así el sesgo hacia las clases dominantes.

### Reglas de consolidación

El proceso de transferir o integrar la información desde la memoria episódica rápida hacia el sistema paramétrico lento utiliza varias "reglas" o mecanismos principales:

*   **Autorregulación por ajuste paramétrico:** En sistemas de adaptación de parámetros basados en memoria (MbPA), la consolidación es un proceso de autorregulación. A medida que el modelo paramétrico (lento) se entrena y se vuelve más apto para modelar los datos, la magnitud de la corrección local que extrae de la memoria episódica disminuye automáticamente. El sistema logra la "consolidación" cuando el modelo paramétrico puede realizar predicciones fiables de manera independiente, sin necesitar el contexto de los recuerdos episódicos.
*   **Olvido Predictivo (*Predictive Forgetting*):** Desde un enfoque de la teoría de la información y la generalización, la consolidación no solo estabiliza recuerdos, sino que transforma la memoria mediante una regla de "olvido predictivo". Este principio dicta que el sistema debe **retener selectivamente la información que predice resultados futuros y eliminar progresivamente los detalles incidentales** o el ruido sensorial de la experiencia original que no tienen utilidad predictiva.
*   **Refinamiento y compresión fuera de línea (*Replay*):** Para lograr el olvido predictivo de manera segura sin alterar el aprendizaje rápido inicial (que requiere alta fidelidad), la consolidación se ejecuta en una fase separada "fuera de línea" (una fase análoga al sueño o al descanso). Mediante reactivaciones o bucles generativos (*replay*), los trazos episódicos almacenados se reevalúan e iterativamente **se comprimen hacia un estado latente más esencial, sin volver a acceder a la entrada sensorial bruta**. 

Como ejemplo de esta última regla en arquitecturas modernas de IA (como los Transformers), la caché *Key-Value* actúa como este búfer episódico; durante la consolidación fuera de línea, la estructura de direccionamiento (las Claves) se mantiene casi inalterada por estabilidad, mientras que el contenido (los Valores) experimenta la compresión semántica descrita.

## Cómo se ve afectada la calibración (ECE, NLL) bajo distribution shift, y qué técnicas la preservan durante la adaptación en test-time?
Bajo cambios de distribución (*distribution shifts*), las probabilidades estimadas por los modelos pierden fiabilidad, volviéndose mal calibradas. Sorprendentemente, **la aplicación de muchos métodos estándar de adaptación en tiempo de prueba (TTA) puede empeorar aún más la calibración del modelo**, incrementando métricas como el Error de Calibración Esperado (ECE). 

Esto ocurre principalmente en métodos basados en la minimización de la entropía (como TENT, ETA o ConjPL), ya que estos fuerzan al modelo a aumentar su nivel de confianza (reducir la entropía) en todas las muestras por igual, independientemente de si la predicción es correcta o no. Como resultado, los modelos se vuelven sobreconfiados. En contraste, métodos más simples que únicamente actualizan las estadísticas de las capas de normalización (como *BN_Adapt*) tienden a retener una mejor calibración en comparación con la minimización de entropía pura.

Para preservar y mejorar la calibración durante la adaptación sin disponer de etiquetas reales, existen técnicas específicas:

**1. Escalado de Temperatura No Supervisado (Basado en *Agreement-on-the-Line*)**
El escalado de temperatura clásico requiere un conjunto de validación etiquetado, lo cual es inviable en TTA. Para resolver esto, se ha desarrollado una variante no supervisada que aprovecha el fenómeno de *Agreement-on-the-Line* (AGL) y *Accuracy-on-the-Line* (ACL), el cual demuestra que existe una fuerte correlación lineal entre el acuerdo de las predicciones de distintos modelos y su precisión, permitiendo **estimar la precisión del modelo en el dominio objetivo sin usar etiquetas**. 
Una vez estimada esta precisión (Acc_est), el método utiliza un algoritmo de búsqueda de raíces (método de Newton) para encontrar un valor de temperatura óptimo $\tau$. Este valor **escala la confianza promedio del modelo para que coincida exactamente con la precisión estimada**. Aplicar esta técnica reduce drásticamente el ECE en modelos adaptados con TENT o ETA, acercándolos a los niveles ideales (*oracle bounds*) que se obtendrían si se tuvieran las verdaderas etiquetas.

**2. Filtrado de muestras ruidosas y gradientes (Método ETAGE)**
Técnicas como ETAGE integran la minimización de entropía con el control de la norma de los gradientes y la Diferencia de Probabilidad de Pseudo-Etiquetas (PLPD). Al calcular métricas de calibración y discriminación (ECE, MCE, Brier Score), ETAGE supera a métodos como TENT, SAR o EATA. Esto se logra porque **filtra y descarta activamente las muestras inestables o ruidosas que presentan una alta entropía combinada con normas de gradiente elevadas**. Al evitar que el modelo se sobreajuste a este ruido durante la adaptación, sus estimaciones de probabilidad se mantienen mucho más fiables (reflejado en un Brier Score y ECE más bajos) y su capacidad de calibración se preserva bajo el *distribution shift*.

## Qué evidencia hay de que la cuantización vectorial / representaciones discretas mejoren la robustez frente a corrupciones, y qué causa el colapso del codebook y lo mitiga (EMA, k-means init, reinicio de códigos muertos, SimVQ)?

## ¿En qué condiciones (severidad del shift, gradual vs. abrupto, batches correlacionados online) reporta la literatura los mayores beneficios de TTA, y cuándo son marginales? ¿Qué protocolos evitan sobreestimar ese beneficio?
**Evidencia de la robustez de la cuantización vectorial frente a corrupciones**

La cuantización vectorial (VQ) y las representaciones discretas mejoran significativamente la robustez de los modelos ante ruido y corrupciones porque actúan como un mecanismo intrínseco de filtrado y eliminación de ruido:
*   **Efecto de eliminación de ruido (*Denoising*):** Al forzar que una representación continua y ruidosa se asigne al vector más cercano de un *codebook* previamente aprendido, el sistema "arrastra" la representación interna hacia patrones limpios conocidos. En experimentos con imágenes ruidosas (como MNIST), los modelos con cuantización mantienen una precisión de clasificación mucho más alta frente al ruido en comparación con los autoencoders estándar.
*   **Representación independiente de la calidad:** La cuantización vectorial permite mapear tanto las características de alta calidad como las degradadas hacia un mismo espacio discreto. Esto debilita la información irrelevante o corrupta, permitiendo aprender una representación esencial e independiente de la calidad de la entrada.
*   **Defensa contra ataques adversarios:** Debido a que el proceso de cuantización utiliza la operación *argmin* (que no es diferenciable), crea un cuello de botella que interrumpe el flujo de gradientes (*stop-gradient*). Esta incompatibilidad estructural neutraliza las optimizaciones de los ataques adversarios basados en gradientes (como los *jailbreaks* o perturbaciones adversarias), operando como un preprocesador robusto sin necesidad de reentrenar el modelo amenazado.

---

**Causas del colapso del *codebook***

El colapso del *codebook* es un problema frecuente donde una gran proporción de los vectores discretos queda inactiva o "muerta", limitando la capacidad del modelo. Sus causas principales son:
*   **No estacionariedad del codificador:** A medida que el codificador se actualiza durante el entrenamiento, la distribución de sus características ("*latents*") cambia o se desplaza en el espacio. Los vectores del *codebook* que no son seleccionados en ese momento fallan en recibir actualizaciones y se vuelven obsoletos frente a la nueva distribución de los datos, quedando inactivos permanentemente.
*   **Gradientes dispersos por el estimador *Straight-Through* (STE):** Durante la retropropagación, solo el vector "ganador" (el seleccionado por el *argmin*) recibe una señal de gradiente. Los vectores no seleccionados obtienen un gradiente nulo y no se optimizan.
*   **Bucle de retroalimentación positiva:** Los vectores que reciben más asignaciones se actualizan con más frecuencia, ajustándose cada vez mejor a la distribución de los datos. Esto aumenta su probabilidad de seguir siendo seleccionados, monopolizando el uso del *codebook* y marginando a los demás.
*   **Decaimiento de la tasa de aprendizaje (*Annealing*):** Al reducirse la tasa de aprendizaje, las actualizaciones se vuelven tan pequeñas que resultan insuficientes para que un código inactivo logre cruzar las fronteras de Voronoi y vuelva a recibir asignaciones, consolidando el colapso.

---

**Mecanismos para mitigar el colapso**

Para mantener una alta utilización del *codebook*, se emplean diversas estrategias que abordan la falta de actualización de los vectores:
*   **EMA (Media Móvil Exponencial):** En lugar de optimizar el *codebook* con descenso de gradiente directo, los vectores se actualizan utilizando un promedio ponderado histórico de las salidas del codificador que se les han asignado. Esto proporciona una dinámica de entrenamiento mucho más estable frente a las fluctuaciones y reduce el colapso del *codebook* entre un 40% y un 60%.
*   **Inicialización basada en K-means:** En lugar de inicializar los vectores de forma aleatoria (lo que genera entre un 30% y un 50% más de códigos muertos), se utiliza el algoritmo K-means sobre el primer lote de entrenamiento. Esto asegura que el *codebook* comience con una cobertura razonable y representativa del espacio de características.
*   **Reinicio de códigos muertos:** Métodos como CVQ-VAE identifican activamente los vectores inactivos y los reinicializan utilizando características extraídas (anclajes o *anchors*) de la distribución actual de los datos. Esto los reubica en zonas relevantes del espacio latente, aumentando sus posibilidades de ser seleccionados y optimizados en iteraciones futuras.
*   **SimVQ y proyecciones globales (TransVQ / VQBridge):** Para evitar el problema de los gradientes dispersos, **SimVQ** reparametriza los vectores a través de una transformación lineal, optimizando todo el espacio lineal en lugar de vectores individuales. Sin embargo, como una capa lineal puede ser frágil, métodos más avanzados como **TransVQ** o **VQBridge** utilizan bloques Transformer (mecanismos de autoatención) para modelar interacciones entre todos los códigos. Esto permite que las señales de gradiente fluyan densamente a través de toda la matriz de atención hacia todos los vectores (incluso los inactivos), alineando dinámicamente todo el *codebook* con el codificador y logrando tasas de utilización cercanas al 100%.


## ¿En qué condiciones (severidad del shift, gradual vs. abrupto, batches correlacionados online) reporta la literatura los mayores beneficios de TTA, y cuándo son marginales? ¿Qué protocolos evitan sobreestimar ese beneficio?

La literatura sobre Adaptación en Tiempo de Prueba (TTA, por sus siglas en inglés) revela que el éxito de estos métodos depende críticamente de las características del entorno en el que se despliegan. A continuación, se detallan las condiciones donde TTA brilla, dónde fracasa y qué protocolos son necesarios para evaluar su beneficio real.

### 1. Condiciones para los mayores beneficios de TTA
Los métodos de TTA reportan sus mayores mejoras sobre el modelo original cuando se cumplen ciertas condiciones "benignas" o controladas:
*   **Severidad del shift (leve a moderada):** TTA es altamente efectivo para mitigar desajustes moderados causados por corrupciones comunes (como ruido de sensores, cambios de clima o variaciones de dominio leves).
*   **Shifts graduales y estacionarios:** Los métodos de TTA logran mejores resultados cuando el cambio de distribución es estacionario o evoluciona lentamente, permitiendo que el modelo ajuste sus parámetros estadísticos de forma progresiva. 
*   **Batches grandes e independientes (i.i.d.):** El mayor rendimiento de TTA (especialmente en métodos basados en *Batch Normalization* como Tent) se da cuando los datos de prueba llegan en batches relativamente grandes (ej. 64 a 200 muestras) y las clases dentro del batch están distribuidas de manera uniforme y aleatoria.

### 2. Condiciones donde el beneficio es marginal o perjudicial
El rendimiento de TTA cae drásticamente, llegando a ser peor que no adaptar el modelo en absoluto, cuando se enfrenta a escenarios del "mundo real salvaje" (*wild scenarios*):
*   **Shifts severos:** Ante cambios drásticos de distribución (por ejemplo, adaptar un modelo de dígitos SVHN a MNIST), la adaptación puede fracasar por completo y aumentar significativamente la tasa de error.
*   **Shifts abruptos y largos (Long-horizon):** Cuando la distribución cambia de forma brusca a lo largo del tiempo, los métodos de TTA que no cuentan con mecanismos de reinicio (*resets*) sufren de "colapso de entropía". El modelo se vuelve sobreconfiado, acumula errores y termina prediciendo una única clase para todas las muestras (solución trivial).
*   **Batches pequeños o temporalmente correlacionados:** Si los datos llegan de a una muestra (tamaño de batch = 1), tienen dependencias temporales o presentan un desbalance de etiquetas temporal (ej. una ráfaga de imágenes de la misma clase), las estimaciones estadísticas colapsan. Métodos populares experimentan una degradación severa bajo este "label shift".
*   **Correlaciones espurias:** Frente a distribuciones donde existen atajos engañosos o correlaciones espurias (como en los datasets Waterbirds o ColoredMNIST), la mayoría de los métodos TTA fracasan en mejorar el rendimiento o empeoran al modelo original.

### 3. Protocolos para evitar sobreestimar el beneficio de TTA
La literatura reciente ha detectado que muchos beneficios reportados previamente estaban inflados por malas prácticas de evaluación. Para evitar sobreestimar los beneficios de TTA, se exigen los siguientes protocolos:

*   **Evitar la "Selección por Oráculo" (Oracle Model Selection):** Los algoritmos TTA son extremadamente sensibles a sus hiperparámetros (tasa de aprendizaje, pasos de adaptación). En la práctica, es imposible ajustar estos parámetros porque no hay etiquetas en la fase de prueba. Evaluar buscando los mejores parámetros post-hoc usando las etiquetas reales de prueba genera una ilusión de eficacia y agrava el problema de dependencia del batch. Los protocolos justos exigen fijar los hiperparámetros de antemano o usar sintonización no supervisada.
*   **Evaluación bajo restricciones computacionales realistas:** En entornos de streaming real (ej. APIs), los modelos TTA requieren retropropagación continua, lo que añade latencia. Los protocolos realistas penalizan o descartan métodos de TTA que no pueden procesar los datos a la velocidad a la que llegan, revelando que muchos métodos costosos (que usan múltiples aumentos de datos) fracasan por omisión de muestras (adaptation lag).
*   **Diferenciar claramente adaptación "Episódica" vs. "Online Continua":** Muchos beneficios se sobreestiman reseteando el modelo a su estado original después de cada batch (episódico). Un protocolo realista debe someter al modelo a una evaluación **online continua (lifelong)** donde los pesos no se reinician por defecto, obligando al algoritmo a demostrar que no sufre de olvido catastrófico (*catastrophic forgetting*) al transitar por múltiples dominios.
*   **Estandarizar la calidad del modelo pre-entrenado (Benchmarks como TTAB):** El éxito de TTA depende enormemente de las características del modelo base. Irónicamente, el uso de ciertas aumentaciones de datos robustas (como PixMix o AugMix) durante el entrenamiento mejora tanto la generalización que los beneficios adicionales del TTA se vuelven mínimos. Protocolos como el banco de pruebas **TTAB** obligan a comparar los métodos de TTA sobre la misma arquitectura, pre-entrenamiento y optimizador para medir su progreso real.

## A través de todo el corpus, ¿qué denominador común comparten los métodos que sí logran mejoras grandes y robustas en test-time, y qué problemas abiertos señalan los autores en memoria asociativa para inferencia robusta?

A través del corpus analizado, los métodos de Adaptación en Tiempo de Prueba (TTA) que consiguen mejoras significativas y no colapsan en escenarios reales comparten varios principios fundamentales de diseño. Por otro lado, en el terreno de la memoria asociativa, los autores señalan cuellos de botella computacionales y teóricos que aún dificultan la inferencia robusta.

### Denominador común de los métodos TTA exitosos y robustos

Las arquitecturas que superan las vulnerabilidades del TTA clásico (como el colapso por entropía o el olvido catastrófico) comparten las siguientes estrategias:

*   **Evitan la adaptación "ciega" mediante filtrado y selección de muestras:** Los métodos más robustos rechazan la idea de adaptar el modelo con todos los datos entrantes. Utilizan estrategias de selección para ignorar muestras ruidosas, redundantes o con entropía poco fiable. Por ejemplo, **EATA** filtra muestras con alta entropía o excesivamente similares, **SAR** elimina muestras ruidosas que generan gradientes grandes, y **STAMP** emplea un banco de memoria que filtra las muestras basándose en la consistencia de sus predicciones. **DualTTA** va más allá y divide los datos en "probablemente correctos" (donde minimiza la entropía) y "probablemente incorrectos" (donde la maximiza para suprimir la sobreconfianza), evaluándolos bajo transformaciones que alteran o preservan la semántica.
*   **Optimizan hacia mínimos planos (Estabilidad Geométrica):** Para evitar que el modelo descarrile ante gradientes ruidosos producidos por muestras atípicas, algoritmos como **SAR** y **STAMP** integran técnicas de *Sharpness-Aware Minimization (SAM)*. Esto fuerza al modelo a encontrar regiones planas en el paisaje de pérdida, haciéndolo mucho menos sensible a perturbaciones o etiquetas ruidosas.
*   **Mecanismos explícitos contra el olvido catastrófico:** La adaptación continua a nuevas distribuciones destruye el conocimiento original del modelo. Los métodos robustos lo solucionan congelando la mayoría de los pesos y usando regularizadores. **EcoTTA** utiliza redes "meta" ligeras combinadas con regularización auto-destilada desde la red original congelada. **EATA** implementa un regularizador basado en información de Fisher para evitar que los parámetros vitales para la distribución original cambien drásticamente, y **SPARNet** emplea términos similares de regularización anti-olvido.
*   **Validación de consistencia mediante aumentaciones:** En lugar de fiarse únicamente de cuán "seguro" está el modelo de una predicción (entropía), métodos robustos como los basados en arquitecturas *Mean-Teacher* (ej. **CoTTA**, **RoTTA**) o esquemas de promedios de aumentaciones, obligan al modelo a mantener predicciones consistentes bajo diferentes vistas o alteraciones de la misma imagen, lo que reduce drásticamente la adaptación hacia errores.

### Problemas abiertos en memoria asociativa para inferencia robusta

En el ámbito de las memorias asociativas y episódicas (como las Redes de Hopfield modernas o autoencoders generativos), la literatura señala los siguientes obstáculos para lograr sistemas robustos:

*   **Complejidad computacional y sobrecarga de memoria GPU:** Modelos como las Redes de Hopfield modernas (basadas en atención) son inmensamente costosas durante la inferencia y la selección de hiperparámetros. Recuperar secuencias requiere propagar miles de instancias a través de la red y mantener estas cantidades en la memoria de la GPU, lo cual limita drásticamente la escalabilidad de estas arquitecturas y su capacidad para usar mecanismos de atención más avanzados. 
*   **Propiedades dinámicas y "cuencas de atracción" poco claras:** Mientras que la capacidad de almacenamiento de memorias densas asociativas ha sido muy estudiada en su estado de equilibrio, existe un vacío histórico sobre sus propiedades dinámicas. Hasta investigaciones muy recientes, era un problema abierto entender cuantitativamente el tamaño de su cuenca de atracción, es decir, qué tan lejos puede estar el estado inicial (o qué tan corrupta puede estar una entrada) para que el sistema aún logre recuperar la memoria correcta de forma estable.
*   **Limitaciones en la flexibilidad de selección atencional y jerarquía:** En modelos de memoria episódica generativa (que reconstruyen recuerdos a partir de trazos incompletos), los mecanismos actuales de selección de atención son demasiado rígidos. Los autores señalan que se necesitan redes más flexibles (como Transformers en lugar de PixelCNN) y que los espacios de representación latente (matrices de índices en modelos VQ-VAE) deben volverse verdaderamente jerárquicos para completar la información semántica de manera plausible.
*   **Falta de modelado secuencial y "One-shot":** La inferencia episódica robusta debe poder capturar secuencias temporales en tiempo real ("one-shot storage"), pero los modelos asociativos actuales a menudo se limitan a "instantáneas" estáticas sin aprovechar adecuadamente la naturaleza secuencial intrínseca a la memoria episódica o enfrentar el riesgo de sobreajuste crítico ante datasets grandes.

## Bibliografía
1. *Addressing Representation Collapse in Vector Quantized Models with One Linear Layer*. (s.f.). 
2. *Adversarial Defenses via Vector Quantization*. (s.f.). arXiv.
3. *Balance of Number of Embedding and their Dimensions in Vector Quantization*. (s.f.). arXiv.
4. *Contrastive Deep Learning for Variant Detection in Wastewater Genomic Sequencing*. (s.f.). arXiv.
5. *Dual Strategies for Test-Time Adaptation*. (s.f.). arXiv.
6. *ETAGE: ENHANCED TEST TIME ADAPTATION WITH INTEGRATED ENTROPY AND GRADIENT NORMS FOR ROBUST MODEL PERFORMANCE*. (s.f.). arXiv.
7. *EcoTTA: Memory-Efficient Continual Test-Time Adaptation via Self-Distilled Regularization*. (s.f.). CVF Open Access.
8. *Efficient and Stable Test-Time Adaptation for Black-Box Models*. (s.f.). arXiv.
9. *Enhancing Adversarial Robustness via Score-Based Optimization*. (s.f.). arXiv.
10. Fayyaz, Z., Altamimi, A., Zoellner, C., Klein, N., & Wolf, O. T. (2022). A Model of Semantic Completion in Generative Episodic Memory. *Cognitive Psychology*.
11. Gong, T., Kim, Y., Lee, T., Chottananurak, S., & Lee, S. J. (2023). *SoTTA: Robust Test-Time Adaptation on Noisy Data Streams*.
12. Gui, S., et al. (2024). Active Test-Time Adaptation: Theoretical Analyses and An Algorithm. *arXiv*.
13. Hari, S. S. (2025). Enhancing Recommender Systems with RQ-VAE-Based Hierarchical Clustering. *International Journal of Advanced Research in Science, Communication and Technology (IJARSCT)*, 5(9).
14. *Implementation of various Vector Quantized Variational Autoencoders (VQVAEs)*. (s.f.). GitHub.
15. Lu, H., et al. (2026). Beyond Stationarity: Rethinking Codebook Collapse in Vector Quantization. *arXiv*. *(Nota: Este trabajo aparece listado dos veces en tus fuentes, como documento PDF y como enlace web).*
16. Maharana, S. K., Mishra, S., Zhang, Y., Niu, S., Rafi, T. H., Hamm, J., Pedersoli, M., Dolz, J., & Guo, Y. (2026). *Continual Test-Time Adaptation: A Comprehensive Survey*. Zenodo.
17. *MEMORY-BASED PARAMETER ADAPTATION*. (s.f.). OpenReview.
18. Mimura, K., Takeuchi, J., Sumikawa, Y., & Kabashima, Y. (s.f.). *Dynamical Properties of Dense Associative Memory*. arXiv.
19. *Model-Based Monaural Source Separation Using a Vector-Quantized Phase-Vocoder Representation*. (s.f.). Columbia Academic Commons.
20. *Modern Hopfield Networks and Attention for Immune Repertoire Classification*. (s.f.). arXiv.
21. *Monitoring Risks in Test-Time Adaptation*. (s.f.). arXiv.
22. Niu, S., Wu, J., Zhang, Y., Chen, Y., Zheng, S., Zhao, P., & Tan, M. (2022). Efficient Test-Time Model Adaptation without Forgetting. *Proceedings of Machine Learning Research*.
23. *On Pitfalls of Test-time Adaptation*. (s.f.). OpenReview.
24. *On the Adversarial Robustness of Discrete Image Tokenizers*. (s.f.). arXiv.
25. *Prototype–based continual learning for single-cell annotation*. (s.f.). bioRxiv.
26. *Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security*. (s.f.). NDSS Symposium.
27. Ramsauer, H., et al. (2021). *Hopfield Networks is All You Need*. OpenReview.
28. *Ranked Entropy Minimization for Continual Test-Time Adaptation*. (s.f.). arXiv.
29. *RDumb++: Drift-Aware Continual Test-Time Adaptation*. (s.f.). arXiv.
30. *Reliable Test-Time Adaptation via Agreement-on-the-Line*. (s.f.). OpenReview.
31. *Robust Residual Finite Scalar Quantization for Neural Compression*. (s.f.). arXiv.
32. *SCALABLE TRAINING FOR VECTOR-QUANTIZED NETWORKS WITH 100% CODEBOOK UTILIZATION*. (s.f.). OpenReview.
33. Sheng, L., et al. (2025). The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models. *arXiv*.
34. *SPARK-IL: Spectral Retrieval-Augmented RAG for Knowledge-driven Deepfake Detection via Incremental Learning*. (s.f.). arXiv.
35. *SPARNet: Continual Test-Time Adaptation via Sample Partitioning Strategy and Anti-Forgetting Regularization*. (s.f.). arXiv.
36. *STAMP: Outlier-Aware Test-Time Adaptation with Stable Memory Replay*. (s.f.). ECVA.
37. *Scaling the Codebook Size of VQGAN to 100,000 with a Utilization Rate of 99%*. (s.f.). NIPS.
38. *TARO: Temporal Adversarial Rectification Optimization Using Diffusion Models as Purifiers*. (s.f.). arXiv.
39. *Towards stable test-time adaptation in dynamic wild world*. (s.f.). arXiv.
40. Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2020). *Tent: Adaptación en Tiempo de Prueba Mediante Minimización de Entropía* [PDF]. arXiv.
41. Wang, Z., Luo, Y., Zheng, L., Chen, Z., Wang, S., & Huang, Z. (2025). In Search of Lost Online Test-Time Adaptation: A Survey. *International Journal of Computer Vision*, 133, 1106–1139.
42. Wikipedia contributors. (2025). Modern Hopfield network. En *Wikipedia, The Free Encyclopedia*.
43. Yang, S., Wang, Y., van de Weijer, J., Herranz, L., & Jui, S. (2021). Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation. *OpenReview*.
44. Yang, Z., Xu, Z., Zhang, J., Hartley, R., & Tu, P. (2023). Adversarial Purification with the Manifold Hypothesis. *arXiv*.
45. *Vector Quantization With Self-Attention for Quality-Independent Representation Learning*. (2023). CVPR.
46. Zhang, R., et al. (s.f.). Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification. *ECVA*.
47. Zhou, D., Noh, S. M., Harhen, N. C., Banavar, N. V., Kirwan, C. B., Yassa, M. A., & Bornstein, A. M. (2025). A compressed code for memory discrimination. *bioRxiv*. National Library of Medicine.