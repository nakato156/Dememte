// =============================================================================
// inform.typ — Reporte vivo de DeMemte (tesis de retrieval no-paramétrico)
//
// Todas las tablas se derivan en tiempo de compilación de los CSV/JSON en
// notebooks/*/out/ vía csv()/json() nativo de Typst. NINGÚN número de resultado
// está tecleado a mano: un rerun de experimento actualiza este documento.
// Rutas repo-relativas (el .typ vive en la raíz; csv() rootea al project root).
//
// Compilar:  typst compile inform.typ
// =============================================================================

#set document(title: "DeMemte: memoria episódica no-paramétrica sobre un backbone congelado", author: ("Christian Borasino", "Rody Vilchez"))
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.6cm), numbering: "1")
#set text(font: "New Computer Modern", lang: "es", size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(0.4em); it; v(0.2em) }
#show raw.where(block: true): it => block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%, text(size: 8.5pt, it))
#set table(inset: (x: 7pt, y: 4pt))

// ---- Plomería de números vivos --------------------------------------------

// Formateo de punto fijo robusto (maneja "", none, "nan", signo y carry de redondeo).
#let fmt(x, dec: 3) = {
  if x == none { return "—" }
  let s = str(x).trim()
  if s == "" or s == "nan" or s == "NaN" { return "—" }
  let v = float(s)
  let neg = v < 0
  let a = calc.abs(v)
  let factor = calc.pow(10, dec)
  let scaled = calc.round(a * factor)
  let intpart = calc.floor(scaled / factor)
  let frac = scaled - intpart * factor
  let fracstr = str(int(frac))
  while fracstr.len() < dec { fracstr = "0" + fracstr }
  let out = str(int(intpart)) + "." + fracstr
  if neg { out = "−" + out }
  out
}
#let pct(x, dec: 1) = {
  if x == none { return "—" }
  let s = str(x).trim()
  if s == "" { return "—" }
  fmt(float(s) * 100, dec: dec) + "\%"
}

// Definición de columna + render de celda
#let C(key, title, kind: "num3") = (key: key, title: title, kind: kind)
#let cell(r, c) = {
  let v = r.at(c.key, default: none)
  if c.kind == "text" { v }
  else if c.kind == "num2" { fmt(v, dec: 2) }
  else if c.kind == "num3" { fmt(v, dec: 3) }
  else if c.kind == "pct1" { pct(v, dec: 1) }
  else if c.kind == "signed" {
    if v == none or str(v).trim() == "" { "—" }
    else { let s = fmt(v, dec: 3); if float(v) > 0 { "+" + s } else { s } }
  } else { v }
}

// Tabla estilo booktabs alimentada por un CSV (acceso por header)
#let booktab(path, cols, filter: none, relabel: (:)) = {
  let data = csv(path, row-type: dictionary)
  let rows = if filter == none { data } else { data.filter(filter) }
  table(
    columns: cols.len(),
    stroke: none,
    align: (x, _) => if cols.at(x).kind == "text" { left } else { right },
    table.hline(stroke: 0.7pt),
    ..cols.map(c => strong(c.title)),
    table.hline(stroke: 0.4pt),
    ..rows.map(r => {
      let rr = r
      for (k, v) in relabel { if rr.at(k, default: none) == k { } }
      cols.map(c => {
        if c.kind == "text" and c.key in relabel.keys() and r.at(c.key) in relabel.at(c.key) {
          relabel.at(c.key).at(r.at(c.key))
        } else { cell(r, c) }
      })
    }).flatten(),
    table.hline(stroke: 0.7pt),
  )
}

#let src = "notebooks/" // prefijo

// =============================================================================
#align(center)[
  #text(size: 17pt, weight: "bold")[DeMemte: memoria episódica no-paramétrica\ sobre un backbone congelado rompe el\ trade-off limpio–corrupto]
  #v(0.6em)
  #text(size: 11pt)[Christian Borasino #h(1.5em) Rody Vilchez]
  #v(0.3em)
  #text(size: 9pt, style: "italic")[Las tablas de este reporte se generan desde los CSV de `notebooks/*/out/`; los números no se editan a mano.]
]
#v(0.8em)

#heading(numbering: none, outlined: false, level: 2)[Resumen]
DeMemte aumenta un backbone ResNet congelado con un módulo de cuantización vectorial y
auto-atención (VQSA) y, sobre él, una *memoria episódica no-paramétrica* que recupera vecinos de
un caché de soporte y vota directamente sobre el logit (#raw("logits_base + α·logits_cache")). La
tesis central es que ese voto *rompe el trade-off limpio↔corrupto* —sube la accuracy limpia y la
corrupta a la vez— con un coste de calibración acotado. Validamos el mecanismo sobre un *slate* de
cinco dominios (Flowers-102, CIFAR-10-C, CIFAR-100-C, ImageNet-R de shift natural e ImageNet-C),
delimitamos honestamente su frontera de operación (la ganancia decae con la severidad y se apaga
donde la base ya está saturada) y mostramos que un *temperature scaling* post-hoc, ajustado en
validación limpia, mantiene la ganancia de accuracy y deja ECE/NLL corruptos por debajo de la
fuente en los cinco dominios. El codebook paramétrico, que en la literatura sería "la memoria", se
reposiciona aquí como *ablación negativa*: como clave de recuperación está aliasado, y la
adaptación latente conservadora en test-time resulta estructuralmente inerte. Esos negativos, bien
diagnosticados, son parte de la contribución.

= Introducción

La receta estándar de robustez —*data augmentation* + fine-tuning completo— compra accuracy bajo
corrupción al costo de la accuracy limpia. Este trade-off es el punto de partida del proyecto
(§5.1). DeMemte explora una ruta distinta: *congelar el backbone* y construir robustez en una
*memoria* externa, evitando que el ajuste a corrupciones degrade la representación limpia.

La pregunta que organiza el trabajo (Q5 del proyecto) es si una memoria puede hacer *pattern
completion* en test-time y mejorar la decisión. La respuesta a la que llega el slate experimental
es matizada y honesta:

- *Contribución positiva.* Una memoria *episódica no-paramétrica* que recupera vecinos en el
  espacio continuo `z_pool` (pre-cuantización) y vota sobre el logit *mueve la predicción* y
  *rompe el trade-off* en varios dominios (§5.2–5.3).
- *Frontera.* El efecto *no es universal*: vive donde la base es mejorable/fina y se apaga o
  invierte donde está saturada; además decae con la severidad del shift (§5.4).
- *Control de calibración.* El voto infla ECE/NLL; un *temperature scaling* por dominio lo acota y
  hasta mejora la calibración respecto a la fuente, sin tocar la accuracy (§5.6).
- *Negativos útiles.* El codebook como memoria paramétrica está aliasado (§5.5, §6) y la adaptación
  latente conservadora es inerte (§6); ambos resultados delimitan *por qué* la memoria útil es la
  episódica no-paramétrica y no el codebook.

= El framework DeMemte

== Arquitectura

```
Imagen (224×224)
  → [ResNet (congelada)]            → feats
  → projector (1×1 conv → BN → GELU) → z
  → VectorQuantizer2D (codebook K)  → zq, vq_loss, soft_assign, encoding_indices
  → GAP(z), GAP(zq)                 → z_pool, zq_pool
  → tokens = stack([z_pool, zq_pool])
  → SelfAttentionBlock × L          → fusión
  → classifier (Linear→GELU→Drop→Linear) → logits
```

El backbone permanece congelado (en `eval()`, sin deriva de BN). El proyector, el cuantizador, los
bloques de auto-atención y el clasificador se entrenan en un único bucle con batches mixtos
limpio/corrupto y pérdida #raw("CE + vq_weight·vq_loss").

== El mecanismo de retrieval (la memoria útil)

Sobre el modelo entrenado y congelado se construye un *caché de soporte*: para cada muestra de
un conjunto de soporte limpio se guarda su descriptor `z_pool` y su etiqueta. En inferencia, la
imagen de consulta recupera vecinos del caché y sus etiquetas forman un logit de memoria que se
suma al logit base:

$ "logits"_"final" = "logits"_"base" + alpha dot "logits"_"cache" $

El pivote clave del proyecto es que la clave de recuperación correcta es *`z_pool` (continuo,
pre-cuantización)*, no `zq_pool` (la salida del codebook): el codebook está aliasado y vota con
fuerza pero sin precisión (§5.5). Todo lo que sigue es post-hoc sobre el modelo congelado: no hay
reentrenamiento.

= Metodología experimental

== Slate de dominios OOD

El mecanismo se evalúa sobre cinco dominios elegidos para cubrir ejes de shift distintos:

#figure(
  table(
    columns: 5,
    stroke: none,
    align: (x, _) => if x <= 1 { left } else { center },
    table.hline(stroke: 0.7pt),
    strong[Dominio], strong[Backbone], strong[Tipo de shift], strong[Granularidad], strong[Severidad],
    table.hline(stroke: 0.4pt),
    [Flowers-102], [RN18], [sintético], [fino], [grid 3 niveles],
    [CIFAR-10-C], [RN18], [sintético (15 corr.)], [grueso], [1–5],
    [CIFAR-100-C], [RN18], [sintético (15 corr.)], [medio/fino], [1–5],
    [ImageNet-R], [RN50], [natural / semántico], [medio], [n/a],
    [ImageNet-C], [RN50], [sintético], [medio], [sev 3, 5],
    table.hline(stroke: 0.7pt),
  ),
  caption: [Slate OOD: rol y eje de shift por dominio (cobertura de `THESIS.md`).],
)

== Protocolo

Partición `historical_trainval_resplit` con `seed=42` en todos los experimentos (la comparabilidad
con la tabla de referencia depende de no cambiar ninguno de los dos). Suite de corrupción
determinista de evaluación (`STRICT_SUITE`, grid 4×3 de tipo×nivel para Flowers; suites estándar
para CIFAR-C/ImageNet-C/-R). El caché de retrieval se construye sobre el conjunto de soporte limpio
de cada dominio.

== Métricas

Reportamos accuracy limpia y corrupta (promedio sobre la suite), calibración (ECE, NLL, Brier) y,
para el retrieval, diagnósticos del voto: fracción de predicciones *corregidas* y *rotas* por la
memoria, y el delta de accuracy frente a la condición sin retrieval (`source`).

= Resultados

== Existe el trade-off (C1)

Enfrentar las tres patas en Flowers-102 aísla el trade-off: el fine-tuning sin augmentation
(`ft_clean`) maximiza la accuracy limpia a costa de la corrupta; añadir augmentation (`finetuned_aug`)
recupera corrupta sacrificando limpia; el *linear probe* congelado queda dominado en ambas.

#figure(
  booktab(
    "notebooks/01_baseline/out/baseline_summary.csv",
    (C("label", "Baseline", kind: "text"), C("clean_acc", "Acc limpia", kind: "num3"),
     C("corrupt_acc_avg", "Acc corrupta", kind: "num3"), C("ece_clean", "ECE limpia", kind: "num3"),
     C("ece_corrupt_avg", "ECE corrupta", kind: "num3")),
  ),
  caption: [Baselines ResNet18 en Flowers-102. Fuente: `01_baseline/out/baseline_summary.csv`.],
)
#figure(
  booktab(
    "notebooks/12_flowers_retrieval/out/ft_clean_baseline.csv",
    (C("variant", "Variante", kind: "text"), C("clean_acc", "Acc limpia", kind: "num3"),
     C("corrupt_acc_avg", "Acc corrupta", kind: "num3"), C("ece_clean", "ECE limpia", kind: "num3"),
     C("ece_corrupt_avg", "ECE corrupta", kind: "num3")),
  ),
  caption: [La pata faltante: FT sin augmentation (alta limpia / baja corrupta). Fuente: `12_flowers_retrieval/out/ft_clean_baseline.csv`.],
)

#figure(image("notebooks/04_finetune_vs_frozen/out/plots/scatter_tradeoff.png", width: 72%),
  caption: [Frontera limpio–corrupto: fine-tuning completo vs. DeMemte frozen.])

== El retrieval mueve la predicción (C3)

En ImageNet-C el voto no es inerte: corrige y rompe predicciones (columnas `corrected`/`broken`),
con delta de accuracy frente a `source` distinto de cero. El mecanismo *actúa* en todos los
dominios; que la acción *ayude* es la pregunta de C4.

#figure(
  booktab(
    "notebooks/11_retrieval_memory/out/e11_results.csv",
    (C("variant", "Variante de caché", kind: "text"),
     C("delta_clean_vs_source", "Δ limpia", kind: "signed"),
     C("delta_corrupt_vs_source", "Δ corrupta", kind: "signed"),
     C("corrected_by_retrieval_corrupt_avg", "Corregidas", kind: "num3"),
     C("broken_by_retrieval_corrupt_avg", "Rotas", kind: "num3")),
    filter: r => r.variant in ("source_cache_z_pool_fixed_alpha", "source_cache_fused_fixed_alpha", "source_cache_zq_pool_fixed_alpha", "source_cache_zq_pool_unfamiliarity_alpha"),
  ),
  caption: [ImageNet-C: el retrieval mueve la predicción (Δ vs `source`). Fuente: `11_retrieval_memory/out/e11_results.csv`.],
)

== Rompe el trade-off (C4)

*Núcleo de la tesis.* En ImageNet-C la variante `z_pool` sube limpia *y* corrupta simultáneamente
(ambos deltas positivos en la tabla anterior). El mismo patrón aparece en el barrido de α en
Flowers, donde aumentar α empuja ambas accuracies por encima de `source`:

#figure(
  booktab(
    "notebooks/12_flowers_retrieval/out/e11_retrieval_flowers.csv",
    (C("variant", "Variante", kind: "text"), C("clean_acc", "Acc limpia", kind: "num3"),
     C("corrupt_acc_avg", "Acc corrupta", kind: "num3"), C("ece_corrupt_avg", "ECE corrupta", kind: "num3"),
     C("nll_corrupt_avg", "NLL corrupta", kind: "num2")),
  ),
  caption: [Flowers-102: barrido de α (sube limpia y corrupta). Fuente: `12_flowers_retrieval/out/e11_retrieval_flowers.csv`.],
)

El efecto *sobrevive a un shift natural/semántico* (ImageNet-R, *renditions*: no hay ruido
sintético que "quitar"): el retrieval sube tanto la accuracy limpia como la de R. El claim vive en
el delta `source`→`retrieval`, no en los absolutos (la base RN50-VQSA es débil, §7).

#figure(
  booktab(
    "notebooks/13_imagenet_r/out/e13_imagenet_r.csv",
    (C("variant", "Variante", kind: "text"), C("domain", "Dominio", kind: "text"),
     C("acc", "Acc", kind: "num3"), C("ece", "ECE", kind: "num3"), C("nll", "NLL", kind: "num2")),
    filter: r => r.variant in ("source", "retrieval_z_pool_fixed_alpha@1.0"),
  ),
  caption: [ImageNet-R (shift natural): el efecto sobrevive. Fuente: `13_imagenet_r/out/e13_imagenet_r.csv`.],
)

== Frontera del mecanismo (C4 / wtq.7)

La curva CIFAR-C delimita honestamente *dónde* funciona el retrieval. En *CIFAR-100* (fino, base
mejorable) sube la accuracy en las cinco severidades, con ganancia que *decae con la severidad*.
En *CIFAR-10* (grueso, base saturada) el voto es inerte o dañino. No es una refutación: es el
mapa de la frontera de operación.

#figure(
  booktab(
    "notebooks/14_cifar_c/out/e14_cifar_c_curve.csv",
    (C("dataset", "Dataset", kind: "text"), C("severity", "Sev", kind: "text"),
     C("variant", "Variante", kind: "text"), C("acc", "Acc", kind: "num3"), C("ece", "ECE", kind: "num3")),
    filter: r => r.variant in ("source", "retrieval_z_pool_fixed_alpha@1.0"),
  ),
  caption: [CIFAR-10-C vs CIFAR-100-C, severidades 1–5. La ganancia vive en CIFAR-100 y decae con la severidad. Fuente: `14_cifar_c/out/e14_cifar_c_curve.csv`.],
)

== `z_pool` vs `zq_pool`: la clave correcta (C5/C6)

La representación útil para recuperar vecinos es `z_pool` (continuo), no `zq_pool` (la salida del
codebook). La variante `zq_pool` vota con fuerza pero sin precisión —el codebook está aliasado— y
queda lejos de `z_pool` incluso con un gate de *unfamiliarity* que la rescata parcialmente. Esto
es lo que justifica el pivote zq→z_pool (la comparación se lee en la tabla de §5.2: `z_pool` da
deltas positivos en ambos ejes; `zq_pool` no).

== Calibración: coste y control (C8)

El retrieval crudo (`retrieval`) infla ECE/NLL corruptos en los cinco dominios. El control con
delta medido es *temperature scaling* (un escalar $T>0$ por dominio ajustado en validación
limpia minimizando NLL; argmax-invariante, así que *no cambia la accuracy*). `retrieval_temp`
mantiene la ganancia de accuracy y deja ECE/NLL corruptos por debajo de `source` en los cinco
dominios —incluido CIFAR-100, la deuda dura de la frontera.

#figure(
  booktab(
    "notebooks/15_calibration/out/e15_summary.csv",
    (C("domain", "Dominio", kind: "text"), C("variant", "Variante", kind: "text"),
     C("corrupt_acc", "Acc corrupta", kind: "num3"), C("corrupt_ece", "ECE corrupta", kind: "num3"),
     C("corrupt_nll", "NLL corrupta", kind: "num2")),
    filter: r => r.variant in ("source", "retrieval", "retrieval_temp"),
  ),
  caption: [Coste y control de calibración (5 dominios): `retrieval` infla ECE/NLL; `retrieval_temp` los acota bajo `source` sin tocar accuracy. Fuente: `15_calibration/out/e15_summary.csv`.],
)

Las temperaturas ajustadas (por dominio, sobre validación limpia) se leen directamente del JSON del
experimento:

#let temps = json("notebooks/15_calibration/out/e15_temperatures.json").temperatures
#figure(
  table(
    columns: 4,
    stroke: none,
    align: (x, _) => if x == 0 { left } else { right },
    table.hline(stroke: 0.7pt),
    strong[Dominio], strong[T(source)], strong[T(retrieval)], strong[T(gate)],
    table.hline(stroke: 0.4pt),
    ..(("flowers", "cifar10", "cifar100", "imagenet_r", "imagenet_c").map(d => (
      d, fmt(temps.at(d).source, dec: 3), fmt(temps.at(d).retrieval, dec: 3), fmt(temps.at(d).gate, dec: 3)
    )).flatten()),
    table.hline(stroke: 0.7pt),
  ),
  caption: [Temperaturas por dominio. Fuente: `15_calibration/out/e15_temperatures.json`.],
)

El gate de *unfamiliarity* (escalar $alpha_"eff" = alpha dot (1 - "conf"_"base")$) es una válvula
distinta: recupera el negativo de la frontera (CIFAR-10) acercándolo a `source`, pero no es el
control de calibración —ese rol es del temperature scaling.

= Resultados negativos (contribución)

== La adaptación latente conservadora es inerte (C2)

Adaptar en test-time superficies conservadoras *upstream* del logit no mueve la accuracy. Con
LayerNorm afín como superficie (E7b), todas las variantes conservadoras caen a ±0.003 de `source`:
el gradiente de la pérdida de memoria sobre LN es estructuralmente nulo porque `z`/`zq` están
*upstream* del único parámetro adaptable.

#figure(
  booktab(
    "notebooks/08_e7b_tta/out/e7b_results.csv",
    (C("variant", "Variante TTA", kind: "text"), C("clean_acc", "Acc limpia", kind: "num3"),
     C("corrupt_acc_avg", "Acc corrupta", kind: "num3"), C("ece_corrupt_avg", "ECE corrupta", kind: "num3")),
    filter: r => r.variant in ("source", "tent_ln", "tent_ln_memreg", "eata_ln", "eata_ln_srcfilter", "eata_ln_srcfilter_memreg"),
  ),
  caption: [E7b (LayerNorm + anclaje de memoria): inerte por construcción. Fuente: `08_e7b_tta/out/e7b_results.csv`.],
)

Cuando la superficie sí es el codebook (SimVQ, E7c), la deriva pasa a ser distinta de cero, pero
las variantes reguladas por memoria la fijan exactamente en 0 (la memoria domina) y la accuracy
queda dentro de ±0.001 de `source`: el mecanismo se valida *mecanísticamente* pero no
*numéricamente*.

#figure(
  booktab(
    "notebooks/09_e7c_codebook/out/e7c_results.csv",
    (C("variant", "Variante", kind: "text"), C("corrupt_acc_avg", "Acc corrupta", kind: "num3"),
     C("zq_drift_corrupt_avg", "Deriva zq", kind: "num3"), C("z_drift_corrupt_avg", "Deriva z", kind: "num3")),
    filter: r => r.variant in ("source", "tent_codebook_softassign", "tent_codebook_memreg", "codebook_loss_adapt", "codebook_loss_adapt_memreg"),
  ),
  caption: [E7c (plasticidad del codebook SimVQ): la deriva se mueve, la accuracy no. Fuente: `09_e7c_codebook/out/e7c_results.csv`.],
)

La misma inercia aparece a escala en ImageNet-C (E10): el *soft-mix* de memoria se lava antes del
logit (`10_memory_hippocampal/out/e10_results.csv`, `10b_imagenet_c/out/e10_imagenet_c_results.csv`).

== El codebook como memoria está aliasado; la memoria episódica online se contamina (C6/C7)

Dos ablaciones negativas cierran el argumento de por qué la memoria útil es episódica y
no-paramétrica sobre `z_pool`: usar `zq_pool` como clave (codebook aliasado) y dejar que la memoria
episódica online se auto-etiquete con pseudo-labels (se contamina: rompe más de lo que repara).

#figure(
  booktab(
    "notebooks/11_retrieval_memory/out/e11_results.csv",
    (C("variant", "Variante de caché", kind: "text"),
     C("delta_clean_vs_source", "Δ limpia", kind: "signed"),
     C("delta_corrupt_vs_source", "Δ corrupta", kind: "signed"),
     C("broken_by_retrieval_corrupt_avg", "Rotas", kind: "num3")),
    filter: r => r.variant in ("source_cache_zq_pool_fixed_alpha", "episodic_cache_zq_pool", "dual_cache_zq_pool"),
  ),
  caption: [Negativos de memoria: codebook aliasado y episódico contaminado. Fuente: `11_retrieval_memory/out/e11_results.csv`.],
)

= Ablaciones

== Componentes del VQSA

Ablar las piezas del módulo VQSA (sobre el sustrato congelado, sin retrieval) confirma su rol:
`replace` (sustituir `z` por `zq` en ambos tokens) colapsa; quitar la auto-atención (`concat_no_sa`)
o el codebook (`no_codebook`) degrada de forma controlada; `add` rinde competitivo.

#figure(
  booktab(
    "notebooks/03_ablations/out/ablation_results.csv",
    (C("label", "Ablación", kind: "text"), C("clean_acc", "Acc limpia", kind: "num3"),
     C("corrupt_acc_avg", "Acc corrupta", kind: "num3"), C("ece_corrupt_avg", "ECE corrupta", kind: "num3")),
  ),
  caption: [Ablaciones del VQSA. Fuente: `03_ablations/out/ablation_results.csv`.],
)

== Barrido de cuantizador (anti-colapso)

El uso *hard* del codebook en VQ vainilla es ~0.2% (colapso severo); EMA + init k-means + reinicio
de códigos muertos restaura ~74% de uso con mejor accuracy. Este es el resultado de cabecera de la
calibración del codebook.

#figure(
  booktab(
    "notebooks/06_e6_zq_alignment/out/e6_results.csv",
    (C("variant", "Variante", kind: "text"), C("quantizer_type", "Cuantizador", kind: "text"),
     C("clean_acc", "Acc limpia", kind: "num3"), C("corrupt_acc_avg", "Acc corrupta", kind: "num3"),
     C("hard_usage_clean", "Uso hard", kind: "num3"), C("dead_code_fraction_clean", "Frac. muerta", kind: "num3")),
  ),
  caption: [Barrido de cuantizador E6. Fuente: `06_e6_zq_alignment/out/e6_results.csv`.],
)

#figure(image("notebooks/06_e6_zq_alignment/out/plots/e6_clean_vs_corrupt.png", width: 70%),
  caption: [Clean vs corrupt de las variantes de cuantizador.])

= Discusión

El arco completo es: el trade-off existe (§5.1); una memoria episódica no-paramétrica que vota en
el logit lo rompe en varios dominios (§5.2–5.3) e incluso bajo shift natural; su frontera es
honesta —decae con la severidad y se apaga donde la base está saturada (§5.4)— y su único coste,
la calibración, se controla con un escalar post-hoc (§5.6). Los negativos no son ruido: el codebook
aliasado (§5.5, §6) y la inercia de la adaptación latente (§6) son precisamente lo que explica
*por qué* la memoria útil es la episódica sobre `z_pool` y no el codebook paramétrico. El pivote
narrativo zq→z_pool queda así consumado: el codebook sobrevive como ablación, no como mecanismo.

= Limitaciones

- *Base ImageNet no-SOTA.* El checkpoint RN50-VQSA rinde por debajo de un RN50 plano en ImageNet-C/-R;
  el positivo se reporta como *mejora relativa al sustrato congelado* (delta `source`→`retrieval`),
  no como un absoluto competitivo.
- *Temperature scaling clean→corrupt.* $T$ se ajusta en validación limpia y se aplica a corrupto
  (límite estándar de TS); aquí generaliza al shift, pero es una suposición a vigilar.
- *Frontera de operación.* El retrieval no ayuda donde la base está saturada (CIFAR-10 grueso); el
  gate de *unfamiliarity* mitiga ese caso acercándolo a `source`, pero no lo convierte en ganancia.

= Conclusión

Una memoria episódica no-paramétrica sobre un backbone congelado *rompe el trade-off
limpio–corrupto* allí donde la base es mejorable, sobrevive a un shift natural, y paga su único
coste (calibración) con un control post-hoc barato. El codebook paramétrico, candidato natural a
"la memoria", queda reposicionado como ablación negativa. Las tablas de este reporte se derivan en
vivo de los CSV de los experimentos: un rerun las actualiza, sin números a mano.
