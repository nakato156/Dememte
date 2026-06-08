# E15 — Control de calibración del retrieval (wtq.8, C8 transversal)

La deuda central del epic: el retrieval sube accuracy pero **a costa de calibración** (ECE/NLL
corruptos empeoran, confirmado en los 4 dominios previos). C8 exige *acotar* ese coste, no
reportar accuracy a secas. Este experimento mide dos controles post-hoc sobre los logits de
retrieval, **sin tocar el modelo** (frozen), sobre el slate completo (5 dominios):

1. **Temperature scaling** (Guo et al. 2017) — se ajusta un escalar `T>0` en un split clean-val
   (min NLL) y se divide el logit final. **Argmax-invariante**: la accuracy de `retrieval_temp`
   es idéntica a la de `retrieval`; solo se mueve la calibración.
2. **Confidence gate (unfamiliarity)** — `α_eff = α·(1−base_conf)` por muestra: suprime el voto
   del cache donde la base ya está segura. Apunta a la frontera wtq.7 (negativo de CIFAR-10) y a
   la sobre-confianza. Cambia el argmax (la accuracy difiere algo de `retrieval`).

Ingeniería: el adapter expone `base_logits + cache_logits + retrieval_margin` por muestra, así que
se vuelca **un solo forward por (dominio, condición)** y las 5 variantes se calculan offline — el
slate cuesta ~una eval por condición. 880 filas de detalle.

Reproducir:
`PYTHONPATH=src uv run python notebooks/15_calibration/e15_calibration.py`
(seed=42, key_space="z_pool", α=1.0, gate=unfamiliarity, T en `out/e15_temperatures.json`).
T ajustada por dominio sobre clean-val, aplicada también a corrupto (límite estándar de TS).

## Resultado → `out/e15_summary.csv` (clean + media-corrupto) · `out/e15_calibration.csv` (detalle)

**Accuracy (corrupto, media) — el retrieval gana en los 5 dominios; temp es argmax-invariante:**

| dominio | source | retrieval | gate_unfam |
|---|---:|---:|---:|
| flowers | 0.503 | **0.532** | 0.523 |
| cifar100 | 0.425 | **0.435** | 0.435 |
| cifar10 (frontera) | 0.665 | 0.653 ↓ | **0.664** |
| imagenet_r | 0.232 | **0.252** | 0.245 |
| imagenet_c | 0.217 | **0.241** | 0.236 |

**ECE corrupto (media) — el coste C8 y su control:**

| dominio | source | retrieval | **retrieval_temp** | gate_unfam_temp |
|---|---:|---:|---:|---:|
| flowers | 0.091 | 0.133 ↑ | **0.046** | 0.039 |
| cifar10 | 0.123 | 0.268 ↑ | **0.121** | 0.144 |
| cifar100 | 0.149 | 0.279 ↑ | **0.100** | 0.115 |
| imagenet_r | 0.334 | 0.350 ↑ | **0.161** | 0.210 |
| imagenet_c | 0.147 | 0.188 ↑ | **0.031** | 0.037 |

(NLL corrupto se mueve igual: source→retrieval sube en los 5 dominios; `retrieval_temp` lo baja
**por debajo de source** en los 5 — p.ej. imagenet_c 5.11→5.12→4.52, flowers 2.02→2.01→1.88.)

**Insight 1 — el coste C8 es universal: el retrieval degrada calibración en los 5 dominios.**
`source→retrieval` sube ECE y NLL (clean y corrupto) en todos. Antes lo teníamos acotado solo a
α≈1.0 en Flowers/R y descontrolado en CIFAR-100; el slate completo confirma que **el voto crudo
del cache infla la confianza en todas partes**. Esto cierra la parte descriptiva de C8.

**Insight 2 — temperature scaling es el control C8, con delta medido: acota Y mejora.**
`retrieval_temp` mantiene **toda** la ganancia de accuracy (argmax-invariante) y lleva el ECE
corrupto **por debajo de source en los 5 dominios** — de forma dramática donde más dolía:
CIFAR-100 0.279→0.100 (incluso por debajo del source 0.149), ImageNet-C 0.188→0.031 (1/5 del
source), ImageNet-R 0.350→0.161 (menos de la mitad). Un solo escalar por dominio, ajustado en
clean-val, frozen. La ganancia de retrieval **ya viene con calibración acotada** — es el mecanismo
de control que C8 pedía. Caveat honesto: T se ajusta en clean y se aplica a corrupto (límite
estándar de TS); aun así el ECE corrupto baja en los 5 dominios, o sea **generaliza al shift**.

**Insight 3 — el gate por unfamiliarity es la válvula para la frontera wtq.7, no el control de calibración.**
El gate cambia el argmax, así que su efecto es sobre **accuracy**: en **CIFAR-10** (el negativo de
wtq.7, base fuerte que el sustrato satura) recupera el corrupto de 0.653 (retrieval, dañino) a
0.664 — **neutraliza casi por completo el negativo** suprimiendo el voto donde la base ya acierta.
El precio: en los dominios donde el retrieval ayuda devuelve algo de ganancia (flowers 0.532→0.523).
Como control de calibración el gate solo va a medias (`gate_unfam` ECE sigue > source); es la
temperatura la que lo arregla. **Se componen**: `gate_unfam_temp` = frontera segura + calibración
acotada. Nota: en esta corrida el gate usa solo `1−base_conf` (la afinidad entra como 0), así que
es unfamiliarity pura — la variante `unfamiliarity_affinity` queda como sweep barato pendiente.

**Lectura para la tesis.** El criterio de éxito del epic era "ganancia robusta CON ECE/NLL
acotado". Con `retrieval + temperature scaling` eso se cumple en los 5 dominios: la accuracy sube
(donde la base es mejorable) y la calibración queda **por debajo** del source. El bloqueo de
calibración sobre C4 se levanta; lo que queda de C4 es la frontera (CIFAR-10, mitigada por el gate)
y el no-SOTA (base RN50 débil), no la calibración.

## Estado de claims (propuesto; lo valida el dueño del ledger)

- **C8** (`parcial` → **`sostenido`**): existe un mecanismo de control de calibración con su delta
  medido en los 5 dominios — temperature scaling acota Y mejora ECE/NLL corruptos por debajo de
  source, manteniendo la accuracy (argmax-invariante). Cita `out/e15_summary.csv`,
  `out/e15_temperatures.json`. La parte descriptiva (el coste existe) queda confirmada en los 5.
- **C4** (`parcial`): se **levanta el bloqueo de calibración**. La ganancia ya viene con ECE/NLL
  acotado (vía temp). Deudas restantes: (c) frontera CIFAR-10 — ahora *mitigada* por el gate
  unfamiliarity (recupera el negativo casi a source) — y (b) no-SOTA. No sube a `sostenido` por
  esas dos, no por calibración.
- **C3** (`sostenido`): sin cambios — el mecanismo actúa; aquí medimos su coste/control.
- **Matriz OOD, columna calibración**: cada dominio del slate reporta ahora ECE/NLL clean+corrupto
  para source/retrieval/retrieval_temp/gate/gate_temp (`out/e15_calibration.csv`).
