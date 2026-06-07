# E13 — ImageNet-R retrieval (wtq.6, shift natural)

Un output, un CSV (`out/e13_imagenet_r.csv`), que alimenta la fila ImageNet-R de la matriz OOD
y refuerza **C3/C4** con un 3er dominio de naturaleza distinta (no sintética). El ledger
(`THESIS.md`) solo cita la ruta; los números viven aquí.

Reproducir:
`PYTHONPATH=src uv run python notebooks/13_imagenet_r/e13_imagenet_r.py`
(checkpoint VQSA ImageNet RN50 `dememte_imagenet_resnet50_vqsa_best.pt`, cargado `strict=True`;
source cache desde `imagenet-clean-5k` train, `key_space="z_pool"`; eval con **máscara a las 200
clases de ImageNet-R**, protocolo Hendrycks; sin escala de severidad — R no tiene niveles).

## Setup

- **Shift**: ImageNet-R = 30k renditions (art, cartoon, sketch, toy, …) de 200 clases ImageNet.
  Shift **natural/semántico**, no corrupción de píxel — el punto del experimento.
- **Máscara-200**: logits puestos a `-inf` fuera de las 200 clases presentes en ImageNet-R, en
  clean-ref y en R por igual (manzanas con manzanas). ECE/NLL/brier salen sobre las 200 efectivas.
- **clean-ref**: subset de `imagenet-clean-5k` val restringido a las 200 clases de R (200 imgs) —
  referencia limpia fina, sirve de ancla del trade-off, no de baseline fuerte.
- **cache**: 5000 keys `z_pool` desde clean-5k train (cubre las 1000 clases; vota por las 200 de R).

## Resultados → `out/e13_imagenet_r.csv`

| variante | clean acc | R acc | clean ece | R ece | R nll | R brier |
|---|---:|---:|---:|---:|---:|---:|
| source (sin retrieval) | 0.855 | 0.232 | 0.068 | 0.334 | 6.60 | 1.018 |
| α=0.5 | 0.870 | 0.244 | 0.072 | 0.341 | 6.61 | 1.012 |
| **α=1.0** | **0.870** | **0.252** | 0.073 | 0.349 | 6.63 | **1.011** |
| α=2.0 | 0.880 | 0.263 | 0.081 | 0.367 | 6.72 | 1.016 |

**Insight C3/C4 (el efecto sobrevive a shift natural).** El retrieval sube clean **y** ImageNet-R
simultáneamente, monótono en α (clean 0.855→0.88; R 0.232→0.263). El mismo mecanismo que en
ImageNet-C (sintético) y Flowers (sintético sobre dominio benigno) **se replica sobre renditions
naturales**. Esto es lo que wtq.6 venía a blindar: refuta la objeción "tu retrieval solo hace
denoising de ruido sintético" — aquí no hay ruido que quitar, el shift es de estilo/semántica y
el voto episódico de vecinos limpios igual empuja la predicción en la dirección correcta. 3er
dominio, de naturaleza distinta a los dos previos.

**Insight C8 (coste de calibración acotado).** En R, ECE empeora suave y monótono con α
(0.334→0.367) pero **Brier mejora** (1.018→1.011 @α=1.0) y NLL queda casi plano hasta α=1.0
(6.60→6.63), encareciéndose recién en α=2.0 (6.72). El régimen de ganancia-con-coste-acotado es
el mismo que en Flowers: **α≈1.0** (R +2.0pp con ECE +0.015 / Brier −0.007 / NLL +0.03);
α≥2.0 da más accuracy (+3.1pp) pero paga ECE/NLL. Coherente con el gate C8/wtq.8.

**Matiz honesto (deuda C4-b, sin cerrar).** Los absolutos en R son bajos (source 0.232, mejor
0.263 aun con máscara-200): es la **base RN50 VQSA débil** ya anotada en el ledger, no SOTA de
ImageNet-R (RN50 plano ronda ~0.36 top-1 en R). Lo que sostiene el claim es el **delta**
source→retrieval (+3.1pp @α=2.0, +2.0pp @α=1.0) sobre el mismo sustrato congelado, no el
absoluto. El ECE alto en R (~0.33) es la base sobreconfiada bajo shift fuerte; el retrieval no lo
arregla pero tampoco lo rompe (Brier incluso baja).

## Estado de claims (propuesto; lo valida el dueño del ledger)

- **Matriz OOD, fila ImageNet-R**: PENDIENTE → hecho (retrieval E11 portado a R, máscara-200,
  barrido α; cita `out/e13_imagenet_r.csv`).
- **C3** (`sostenido`): gana 3er dominio, ahora **natural** además de sintético — el efecto de
  que el retrieval mueve la predicción ya no vive solo en corrupción sintética.
- **C4** (`parcial`): la tensión "robustez solo en dominios benignos/sintéticos" queda
  sustancialmente aliviada (shift natural cubierto); la deuda (b) no-SOTA **sigue abierta**
  (base débil). Candidato más fuerte a `sostenido` si el criterio acepta "mejora sobre sustrato
  frozen con ECE acotado", ahora en 3 dominios incl. uno natural.
- **C8** (`parcial`): 3er dominio confirma el patrón — ganancia con coste de calibración acotado
  a α≈1.0; gate transversal sigue vivo.
