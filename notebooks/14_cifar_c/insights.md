# E14 — CIFAR-10/100-C retrieval (wtq.7, curva de severidad)

El eje que ni ImageNet-C (sev 3,5) ni Flowers (grid de 3) dan limpio: **cómo escala el efecto
del retrieval con la severidad 1–5**, sobre las 15 corrupciones canónicas Hendrycks. Complementa
a wtq.6 (ImageNet-R = *naturaleza* del shift); CIFAR-C = la *curva*. El ledger (`THESIS.md`) solo
cita rutas; los números viven aquí.

A diferencia de wtq.5/wtq.6, este experimento **entrena el sustrato** (no había checkpoint VQSA
de CIFAR y las clases no mapean a ImageNet): frozen RN18 + VQSA + cabeza de 10/100 clases, receta
ganadora E6 (`e6_ema_kmeans_restart`), idéntica a Flowers. Luego cache source (z_pool) desde el
train limpio y eval source vs retrieval@α en CIFAR-C.

Reproducir:
`PYTHONPATH=src CIFAR_DATA_ROOT=/home/r0sewt/data uv run python notebooks/14_cifar_c/e14_cifar_c.py`
(seed=42, key_space="z_pool", α∈{0.5,1.0,2.0}, sev 1–5, 15 corrupciones, test 10k/severidad).

Sustratos entrenados (val clean): **CIFAR-10 0.891**, **CIFAR-100 0.658** (codebook sano, usage
~0.71–0.75, sin colapso).

## Resultado → `out/e14_cifar_c_curve.csv` (media sobre 15 corrupciones) + `out/e14_cifar_c.csv` (detalle)

**Curva de accuracy source→retrieval, Δ vs source:**

| | sev | source | retr@0.5 | retr@1.0 | Δ@0.5 | Δ@1.0 |
|---|---|---:|---:|---:|---:|---:|
| **CIFAR-10** | 1 | 0.787 | 0.791 | 0.786 | +0.004 | −0.001 |
| (coarse) | 2 | 0.728 | 0.728 | 0.722 | −0.000 | −0.006 |
| | 3 | 0.675 | 0.672 | 0.664 | −0.003 | −0.011 |
| | 4 | 0.612 | 0.606 | 0.597 | −0.007 | −0.016 |
| | 5 | 0.524 | 0.513 | 0.502 | −0.011 | −0.022 |
| **CIFAR-100** | 1 | 0.545 | 0.560 | 0.562 | **+0.015** | **+0.016** |
| (fine) | 2 | 0.484 | 0.497 | 0.497 | **+0.013** | **+0.014** |
| | 3 | 0.431 | 0.442 | 0.442 | **+0.011** | **+0.011** |
| | 4 | 0.373 | 0.382 | 0.381 | **+0.010** | **+0.009** |
| | 5 | 0.290 | 0.294 | 0.293 | **+0.005** | **+0.003** |

**Insight 1 — el efecto NO es universal: depende de granularidad / fuerza de la base.**
En **CIFAR-100** (100 clases, base débil 0.66) el retrieval sube accuracy en las **5 severidades**
sobre el source. En **CIFAR-10** (10 clases, base fuerte 0.89) es **inerte o dañino**: apenas
+0.4pp en sev 1 a α=0.5, y negativo desde sev 2, peor con α. Lectura mecánica: el voto del cache
(keys limpias) ayuda cuando la cabeza base es débil y las clases finas (más señal discriminativa
que aportar); en una tarea gruesa que el sustrato congelado ya resuelve bien, el voto sobre una
query corrupta recupera vecinos limpios que no casan y **añade ruido** — efecto que crece con la
severidad (peor match) y con α (más peso al voto). Esto **mapea la frontera de operación de C4**:
el retrieval rompe el trade-off donde la base es mejorable, no donde ya está saturada.

**Insight 2 — la ganancia DECAE con la severidad (curva, CIFAR-100).** El delta source→retrieval
baja monótono: +1.6pp (sev 1) → +0.3pp (sev 5) a α=1.0; a α=0.5 algo más plano (+1.5pp → +0.5pp).
Sentido: a mayor corrupción, la query se aleja de las keys limpias del cache, el top-k recupera
peor y el voto ayuda menos. Es la primera vez que vemos la curva limpia — confirma que el aporte
del retrieval es máximo cerca del dominio limpio y se desvanece (no se invierte, en CIFAR-100) bajo
shift fuerte.

**Insight 3 — coste de calibración NO acotado (C8).** En CIFAR-100 el ECE empeora fuerte con α:
sev 1 source 0.102 → retr@0.5 0.168 → retr@1.0 0.217; sev 5 0.211 → 0.293 → 0.361. NLL y Brier
también suben (a diferencia de ImageNet-R, donde Brier mejoraba). El régimen menos malo es
**α≈0.5**, pero ni ahí el coste queda acotado como en Flowers/R. Refuerza C8 como la deuda central:
la ganancia de accuracy en CIFAR-100 viene con calibración degradada — el gate C8/wtq.8 no se pasa
limpio en este dominio.

## Estado de claims (propuesto; lo valida el dueño del ledger)

- **Matriz OOD, fila CIFAR-C**: PENDIENTE → hecho (curva sev 1–5 sobre 15 corrupciones, 2
  granularidades; cita `out/e14_cifar_c_curve.csv`).
- **C4** (`parcial`): resultado **mixto/condicionado**. Aporta (i) un positivo en CIFAR-100 con la
  curva de severidad y (ii) una **frontera**: el efecto es inerte/dañino en CIFAR-10 (coarse,
  base fuerte). No sube a `sostenido`; añade a la deuda (b) el matiz "no universal — depende de
  granularidad/fuerza de base", además de "no-SOTA". Honestamente fortalece la tesis al delimitar
  *dónde* funciona el mecanismo en vez de sobre-generalizar.
- **C8** (`parcial`): 4º dominio confirma el coste de calibración; aquí **no** queda acotado ni a
  α≈0.5 (ECE ~2× en CIFAR-100). Es la deuda que bloquea C4→`sostenido`.
- **C3** (`sostenido`): el retrieval mueve la predicción también en CIFAR-100 (en CIFAR-10 la
  mueve pero en contra) — el mecanismo *actúa*; su utilidad es lo condicionado.
