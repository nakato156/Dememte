# E7b — Insights

TTA conservador con LayerNorm + preservación de memoria latente, sobre el checkpoint
ganador de E6 (`e6_ema_kmeans_restart`). Resultados en
`notebooks/08_e7b_tta/out/{e7b_results.csv, e7b_curves.csv}`.

## Tabla cabecera (test, `historical_trainval_resplit`, seed 42)

| variante | clean | corrupt avg | ECE corrupt | NLL corrupt | Δcorrupt vs source |
|---|---:|---:|---:|---:|---:|
| **source** | 0.7523 | 0.5030 | 0.0903 | 2.022 | — |
| **bn_stats_no_update** | 0.0317 | 0.0284 | 0.4896 | 8.740 | **−0.475 (colapso)** |
| tent_ln | 0.7520 | 0.5021 | 0.0948 | 2.033 | −0.0009 |
| eata_ln | 0.7517 | **0.5037** | 0.0912 | 2.021 | +0.0007 |
| eata_ln_srcfilter | 0.7520 | 0.5036 | 0.0913 | 2.021 | +0.0006 |
| tent_ln_memreg | 0.7520 | 0.5021 | 0.0948 | 2.033 | −0.0009 |
| eata_ln_srcfilter_memreg | 0.7520 | 0.5036 | 0.0913 | 2.021 | +0.0006 |

Desglose por corrupción (acc media sobre severidades):

| corrupción | source | tent_ln | eata_ln | Δ(eata_ln−source) |
|---|---:|---:|---:|---:|
| blur | 0.6708 | 0.6710 | 0.6718 | +0.0010 |
| cutout | 0.6389 | 0.6393 | 0.6395 | +0.0006 |
| gaussian_noise | 0.3534 | 0.3524 | 0.3534 | +0.0000 |
| pixel_mask | 0.3487 | 0.3455 | 0.3500 | +0.0014 |

## Insights

### 1. El gate técnico se pasó: la causa del colapso de E7 v1 era la superficie BatchNorm
`bn_stats_no_update` —que reproduce la *test-time normalization* de
[Schneider et al. (2020)](https://arxiv.org/abs/2006.16971)— colapsa idéntico a E7 v1
(clean 0.032 / corrupt 0.028), pero **ninguna** variante LayerNorm colapsa: todas se
quedan pegadas a `source`. Esto valida la lectura de RESPONSES Q1/Q3: según
[SAR (Niu et al., ICLR 2023)](https://arxiv.org/abs/2302.12400), BatchNorm es el factor
que desestabiliza TTA y conviene migrar a normas agnósticas al batch (GroupNorm/
LayerNorm). Siguiendo esa recomendación se implementó `configure_tta_layernorm`
(adaptar solo LayerNorm manteniendo BN en running stats), y el resultado confirma que
el problema nunca fue TENT ni el `lr` sino cambiar BatchNorm a estadísticas por batch.
`bn_stats_no_update` se reporta como baseline y no como compuerta (Q4):
[EcoTTA (Song et al., CVPR 2023)](https://arxiv.org/abs/2303.01904) incluye un baseline
explícito "BN Stats Adapt" que se desploma con batch chico (su Tabla 5: batch size 1 →
99.1% error vs 69.7% del source), y [SoTTA (Gong et al., NeurIPS 2023)](https://arxiv.org/abs/2310.10074)
documenta que las TTA basadas en BatchNorm caen por debajo de `source` en streams
ruidosos (p.ej. TENT 81.0% → 52.1%). Reportar el no-update junto a `source` es la
práctica establecida desde Schneider et al.

### 2. La adaptación es inerte: no mueve la aguja
Todas las variantes LN caen dentro de ±0.003 de `source`. Se probó la minimización de
entropía de [TENT (Wang et al., ICLR 2021)](https://arxiv.org/abs/2006.10726) sobre
LayerNorm (`tent_ln`) y el filtro fiabilidad/diversidad de
[EATA (Niu et al., ICML 2022)](https://arxiv.org/abs/2204.02610) sobre la misma
superficie (`eata_ln`). `tent_ln` incluso **empeora** levemente corrupt (−0.0009) y la
calibración (ECE 0.0948 vs 0.0903): minimizar entropía vuelve al modelo más
sobreconfiado. El único positivo marginal es `eata_ln` (+0.0007, traccionado por
pixel_mask +0.0014 y blur +0.0010), pero está dentro del ruido. `tent_ln` actualiza el
100% de los samples (385 updates, selection_rate 1.0) y aun así no cambia nada;
`eata_ln` filtra al ~8% (147.6 updates, selection_rate 0.0805) y básicamente **evita el
daño** que hace TENT en vez de aportar ganancia —consistente con el rol del filtrado de
muestras de EATA. Conclusión: el affine de LayerNorm tiene **leverage casi nulo** sobre
un clasificador de features congeladas.

### 3. Hallazgo central — el regularizador de memoria es un no-op estructural
`tent_ln` y `tent_ln_memreg` son **bit-a-bit idénticos** (mismas 16 cifras en
acc/ECE/NLL); igual `eata_ln_srcfilter` vs su `_memreg`:

```
tent_ln.corrupt_acc_avg            = 0.5020735078874614
tent_ln_memreg.corrupt_acc_avg     = 0.5020735078874614   (idéntico)
eata_ln_srcfilter.corrupt_acc_avg        = 0.5035778175313059
eata_ln_srcfilter_memreg.corrupt_acc_avg = 0.5035778175313059   (idéntico)
```

El regularizador `latent_memory_loss` se implementó siguiendo la regularización
auto-destilada de [EcoTTA (Song et al., CVPR 2023)](https://arxiv.org/abs/2303.01904),
que mantiene la salida adaptada cerca de la de la red source **congelada** mediante
`‖x̃−x‖₁`; acá el análogo es MSE sobre `z`/`zq` y KL sobre las asignaciones del codebook.
(El anti-olvido de [EATA (Niu et al., ICML 2022)](https://arxiv.org/abs/2204.02610)
persigue un objetivo similar pero por otra vía —una penalización L2 ponderada por Fisher
sobre los **pesos**, no sobre la representación—, por eso no es la base de este
regularizador.) Pero acá no muerde por una razón arquitectónica: en
`src/dememte/models/dememte.py` (`VQSAFusion.forward`),
`z`, `zq` y `soft_assign` se calculan **aguas arriba** de los bloques de self-attention;
los LayerNorm solo tocan los tokens *después* del pooling y de la VQ. Por lo tanto el
gradiente de `latent_memory_loss` respecto a los parámetros LayerNorm es **exactamente
cero**. La memoria latente (el codebook, en el sentido de
[VQ-VAE (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937)) vive antes de la
única superficie adaptable agnóstica al batch.

### 4. Corolario — la "preservación de memoria" es trivial, no efectiva
`hard_usage` y `dead_code_fraction` son **idénticos a source** en todas las variantes
LN (0.7383 / 0.2617 clean; 0.4792 / 0.5208 corrupt). La memoria se "preserva"
trivialmente porque es **inalcanzable** desde LayerNorm, no porque el regularizador
haya hecho algo. El objetivo Q5 (preservar/adaptar el espacio latente compartido
limpio/corrupto) **no se puede testear con esta superficie de adaptación**.

## Implicación / próximo paso (E7c)

E7b cierra limpio la pregunta metodológica como **resultado negativo útil**:
*conservative TTA = seguro pero inerte, y el regularizador de memoria está
estructuralmente desactivado por dónde se sitúa el codebook.*

Para que la tesis de DeMemte (pattern completion tipo memoria humana, Q5) sea
siquiera comprobable, la superficie de adaptación debe **influir sobre `z`/`zq`**, que
es justo lo congelado/dependiente de BN. Dos rutas que harían "morder" el
regularizador, manteniendo la disciplina de E7b (ancla a teacher source congelado, à la
[EcoTTA](https://arxiv.org/abs/2303.01904), y reporte del baseline BN-Stats):

- **Adaptar el codebook VQ en test-time** — los embeddings son `nn.Parameter`. Usar
  `latent_memory_loss` como ancla anti-deriva. Es lo más cercano a plasticidad sináptica
  y, a diferencia de la actualización EMA del codebook (introducida en el
  [VQ-VAE original, Apéndice A.1 (van den Oord et al., 2017)](https://arxiv.org/abs/1711.00937)),
  se haría guiada por gradiente de entropía con preservación; el regularizador deja de
  ser no-op. Mantener BN del projector en running stats (como acá).
- **Adaptar el projector** (conv 1×1) con su BatchNorm congelada en running stats —o
  calibrando estadísticas al estilo [TTN (Lim et al., ICLR 2023)](https://arxiv.org/abs/2302.05155)
  / familia AdaptBN— para que `z` se mueva y el ancla `z`/`zq` tenga sentido.

Si la adaptación por gradiente sigue siendo inestable, la ruta de caja negra de
[FOA (Niu et al., ICML 2024)](https://arxiv.org/abs/2404.01650) (prompt de entrada/salida
sin retropropagación, red congelada) es el plan B que evita tocar BN por completo.

Ambas son "E7c": la **memoria como superficie adaptable con preservación**, en lugar de
LayerNorm.

## Reproducir

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest    # 23 passed (incluye tests E7b)
# Notebook (GPU + checkpoint E6): notebooks/08_e7b_tta/e7b_tta.ipynb
```

## Referencias (trabajos citados)

Citas verificadas vía búsqueda; cada entrada indica dónde se usa en el proyecto. Las
referencias `Q1`–`Q5` apuntan a las preguntas fundamentadas de `RESPONSES.md`.

### Literatura TTA que fundamenta E7b

- **TENT** — Wang, Shelhamer, Liu, Olshausen, Darrell. *Tent: Fully Test-Time
  Adaptation by Entropy Minimization.* ICLR 2021 (spotlight).
  [arXiv:2006.10726](https://arxiv.org/abs/2006.10726).
  → método base de E7 v1 (`tent_bn`) que colapsó; minimización de entropía sobre
  parámetros de normalización.
- **EATA** — Niu, Wu, Zhang, Chen, Zheng, Zhao, Tan. *Efficient Test-Time Model
  Adaptation without Forgetting.* ICML 2022.
  [arXiv:2204.02610](https://arxiv.org/abs/2204.02610).
  → filtro de fiabilidad (margen de entropía) y diversidad (coseno) de las variantes
  `eata_ln*`; actualiza solo el affine de las normas. Su anti-olvido es L2 ponderado por
  Fisher sobre los **pesos** —distinto de la preservación representacional de DeMemte
  (esa proviene de EcoTTA, no de EATA).
- **SAR** — Niu, Wu, Zhang, Wen, Chen, Zhao, Tan. *Towards Stable Test-Time
  Adaptation in Dynamic Wild World.* ICLR 2023 (oral).
  [arXiv:2302.12400](https://arxiv.org/abs/2302.12400).
  → Q1/Q2/Q3: identifica BatchNorm como obstáculo y recomienda normas agnósticas al
  batch (GN/LN); base de adaptar solo LayerNorm en E7b.
- **CoTTA** — Wang, Fink, Van Gool, Dai. *Continual Test-Time Domain Adaptation.*
  CVPR 2022. [arXiv:2203.13591](https://arxiv.org/abs/2203.13591).
  → Q2: teacher por EMA / restauración de pesos source contra el olvido.
- **EcoTTA** — Song, Lee, Kweon, Choi. *EcoTTA: Memory-Efficient Continual Test-Time
  Adaptation via Self-Distilled Regularization.* CVPR 2023.
  [arXiv:2303.01904](https://arxiv.org/abs/2303.01904).
  → Q2/Q5: regularización por destilación desde la red source congelada; sustento
  directo de `latent_memory_loss` anclada a un teacher source.
- **SoTTA** — Gong, Kim, Lee, Chottananurak, Lee. *SoTTA: Robust Test-Time Adaptation
  on Noisy Data Streams.* NeurIPS 2023.
  [arXiv:2310.10074](https://arxiv.org/abs/2310.10074).
  → Q4: documenta que el baseline "BN Stats" rinde peor que source en streams ruidosos;
  justifica reportar `bn_stats_no_update` sin invalidar el experimento.
- **FOA** — Niu, Miao, Chen, Wu, Zhao. *Test-Time Model Adaptation with Only Forward
  Passes.* ICML 2024 (oral). [arXiv:2404.01650](https://arxiv.org/abs/2404.01650).
  → Q3: alternativa libre de retropropagación (prompt de entrada/salida con red
  congelada); plan B "caja negra" para E7c.
- **Schneider et al.** — *Improving Robustness against Common Corruptions by Covariate
  Shift Adaptation.* NeurIPS 2020. [arXiv:2006.16971](https://arxiv.org/abs/2006.16971).
  → línea base de adaptación de estadísticas BN (el "BN Stats" / `bn_stats_no_update`).
- **TTN** — Lim, Kim, Choo, Choi. *TTN: A Domain-Shift Aware Batch Normalization in
  Test-Time Adaptation.* ICLR 2023. [arXiv:2302.05155](https://arxiv.org/abs/2302.05155).
  → Q1/Q3: interpolación CBN↔TBN; familia "AdaptBN" (mezclar stats source/target),
  plan B fuera del alcance de E7b. Relacionado: *Test-time Batch Statistics Calibration
  for Covariate Shift* (α-BN), [arXiv:2110.04065](https://arxiv.org/abs/2110.04065).
- **Hybrid-TTA** — Park, Park, Ko, Min. *Hybrid-TTA: Continual Test-time Adaptation via
  Dynamic Domain Shift Detection.* 2024.
  [arXiv:2409.08566](https://arxiv.org/abs/2409.08566).
  → Q2: marco teacher-student con teacher EMA para pseudo-etiquetas estables.

### Cuantizadores VQ (modelo DeMemte y variantes E6)

- **VQ-VAE** — van den Oord, Vinyals, Kavukcuoglu. *Neural Discrete Representation
  Learning.* NeurIPS 2017. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937).
  → codebook, commitment loss y straight-through del `VectorQuantizer2D`; además
  introduce (Apéndice A.1) la **actualización EMA del codebook** que usa
  `EMAVectorQuantizer2D`.
- **VQ-VAE-2** — Razavi, van den Oord, Vinyals. *Generating Diverse High-Fidelity
  Images with VQ-VAE-2.* NeurIPS 2019. [arXiv:1906.00446](https://arxiv.org/abs/1906.00446).
  → VQ jerárquico multiescala que consolida el uso del codebook con EMA (el EMA en sí
  proviene del VQ-VAE original); lineage del `EMAVectorQuantizer2D`.
- **Jukebox** — Dhariwal, Jun, Payne, Kim, Radford, Sutskever. *Jukebox: A Generative
  Model for Music.* 2020. [arXiv:2005.00341](https://arxiv.org/abs/2005.00341).
  → reinicio aleatorio de códigos muertos (*dead-code restart*) de la variante ganadora
  `e6_ema_kmeans_restart`.
- **SimVQ** — Zhu et al. *Addressing Representation Collapse in Vector Quantized Models
  with One Linear Layer.* ICCV 2025. [arXiv:2411.02038](https://arxiv.org/abs/2411.02038).
  → variante `e6_simvq_linear` (codebook reparametrizado por capa lineal).
- **FSQ** — Mentzer, Minnen, Agustsson, Tschannen. *Finite Scalar Quantization: VQ-VAE
  Made Simple.* ICLR 2024. [arXiv:2309.15505](https://arxiv.org/abs/2309.15505).
  → variante `e6_fsq` (cuantización escalar finita, *lookup-free*).

### Fundamentos (backbone, datos, normas, robustez)

- **ResNet** — He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition.*
  CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
  → backbone ResNet-18 congelado.
- **Common Corruptions** — Hendrycks, Dietterich. *Benchmarking Neural Network
  Robustness to Common Corruptions and Perturbations.* ICLR 2019.
  [arXiv:1903.12261](https://arxiv.org/abs/1903.12261).
  → inspiración del suite de corrupciones de evaluación.
- **Flowers-102** — Nilsback, Zisserman. *Automated Flower Classification over a Large
  Number of Classes.* ICVGIP 2008.
  [PDF](https://www.robots.ox.ac.uk/~men/papers/nilsback_icvgip08.pdf).
  → dataset (102 clases).
- **Layer Normalization** — Ba, Kiros, Hinton. 2016.
  [arXiv:1607.06450](https://arxiv.org/abs/1607.06450).
  → la norma agnóstica al batch que E7b adapta (en los bloques de self-attention).
- **Group Normalization** — Wu, He. ECCV 2018.
  [arXiv:1803.08494](https://arxiv.org/abs/1803.08494).
  → alternativa agnóstica al batch citada en Q1/Q3 (no usada en el checkpoint actual).

> Nota: `RESPONSES.md` menciona además "BETA" entre los métodos de caja negra; no se
> pudo resolver de forma unívoca a un arXiv canónico vía búsqueda, así que se cita
> **FOA** como representante verificado de esa dirección (libre de retropropagación,
> red congelada). Otra referencia *backprop-free* relacionada: *BaFTA*,
> [arXiv:2406.11309](https://arxiv.org/abs/2406.11309).
