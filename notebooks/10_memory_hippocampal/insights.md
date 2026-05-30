# E10 - Memoria asociativa biologica TTA-only: hallazgos

## TL;DR

- **E10 funciona mecanicamente**: los mecanismos de memoria se activan sin
  gradiente ni reentrenamiento. Hay `completion_amount_corrupt > 0`, el buffer
  episodico escribe (`episodic_buffer_churn > 0`) y las trayectorias de
  completion son finitas/convergentes.
- **Pero el efecto sigue siendo numericamente inerte en accuracy**. En la base
  principal `e6_ema_kmeans_restart`, todas las variantes quedan en un rango de
  aproximadamente `-0.0009` a `+0.0021` en `corrupt_acc_avg` contra `source`.
- **La unica variante net-positiva es la via episodica**:
  `episodic_only` sube clean `+0.0034` y corrupt `+0.0021`, con mejor ECE/NLL
  corrupto que source. Es prometedora, pero esta dentro de ruido experimental.
- **La recuperacion semantica desde el codebook perjudica levemente**:
  `assoc_recall_const` es la peor variante semantica (`-0.0009` corrupt), y
  los gates biologicos de familiaridad/unfamiliaridad no revierten esa perdida.
- **SimVQ confirma el requisito de un codebook sano**: Phase 0 restringe
  `e6_simvq_linear` a `episodic_only` porque su `hard_usage=0.011 < 0.1`.
  La memoria asociativa basada en codebook no tiene sentido cuando el codebook
  esta colapsado.

## Setup

E10 prueba tres mecanismos biologicamente inspirados sobre checkpoints E6 ya
entrenados, sin actualizar parametros:

1. **Recall asociativo** del codebook como Modern Hopfield:
   `softmax(-||z - E||^2 / tau) E`.
2. **Pattern completion** iterativo sobre `z_pool`, con gate de familiaridad o
   unfamiliaridad.
3. **Doble via CLS**: codebook semantico mas buffer episodico EMA.

La integracion es deliberadamente conservadora: mezcla suave en `zq_pool` con
`lambda_max = 0.1`, para no sacar de distribucion al bloque de self-attention y
al clasificador congelado.

Resultados en:

- `notebooks/10_memory_hippocampal/out/e10_results.csv`
- `notebooks/10_memory_hippocampal/out/e10_curves.csv`
- `notebooks/10_memory_hippocampal/out/e10_phase0.json`
- `notebooks/10_memory_hippocampal/out/e10_summary.md`

## Phase 0

| base | check | resultado | decision |
|---|---:|---:|---|
| `e6_ema_kmeans_restart` | median `g_clean` | 0.497 | familiarity viable |
| `e6_ema_kmeans_restart` | median `g_gaussian_noise_1.5` | 0.395 | gate discrimina shift |
| `e6_ema_kmeans_restart` | `hard_usage` | 0.250 | all variants |
| `e6_ema_kmeans_restart` | clean floor con `assoc_recall_const` | -0.0026 | pasa |
| `e6_simvq_linear` | median `g_clean` | 0.500 | familiarity viable |
| `e6_simvq_linear` | median `g_gaussian_noise_1.5` | 0.407 | gate discrimina shift |
| `e6_simvq_linear` | `hard_usage` | 0.011 | solo `episodic_only` |
| `e6_simvq_linear` | clean floor con adapter | +0.0015 | pasa |

Phase 0 deja dos lecturas importantes. Primero, la senal de familiaridad no es
constante: cae de alrededor de 0.50 en clean a 0.39-0.41 bajo gaussian noise.
Segundo, la base SimVQ no puede evaluar recall semantico porque casi no usa el
codebook; esa restriccion refuerza el hallazgo de E6/E7c de que el mecanismo de
memoria depende de utilizacion real del codebook.

## Matriz principal

Base `e6_ema_kmeans_restart`, test `historical_trainval_resplit`, seed 42.
Source: clean `0.7523`, corrupt `0.5030`.

| variante | clean | corrupt avg | Delta clean | Delta corrupt | ECE corrupt | NLL corrupt | completion | epi churn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `episodic_only` | **0.7557** | **0.5051** | +0.0034 | **+0.0021** | **0.0868** | **2.0084** | 0.0508 | 0.0547 |
| `consolidation_slow` | 0.7512 | 0.5035 | -0.0011 | +0.0005 | 0.0891 | 2.0184 | 0.0660 | 0.0616 |
| `hippocampal_full` | 0.7518 | 0.5031 | -0.0005 | +0.0001 | 0.0895 | 2.0183 | 0.0660 | 0.0553 |
| `source` | 0.7523 | 0.5030 | 0.0000 | 0.0000 | 0.0903 | 2.0224 | - | - |
| `assoc_recall_unfamiliarity` | 0.7497 | 0.5025 | -0.0026 | -0.0004 | 0.0899 | 2.0226 | 0.0327 | 0.0000 |
| `assoc_recall_familiarity` | 0.7500 | 0.5025 | -0.0023 | -0.0005 | 0.0901 | 2.0227 | 0.0247 | 0.0000 |
| `completion_T3_best_gate` | 0.7504 | 0.5023 | -0.0020 | -0.0006 | 0.0902 | 2.0229 | 0.0755 | 0.0000 |
| `assoc_recall_const` | 0.7497 | 0.5021 | -0.0026 | -0.0009 | 0.0899 | 2.0240 | 0.0573 | 0.0000 |

Base `e6_simvq_linear`, restringida por Phase 0 a `episodic_only`:

| variante | clean | corrupt avg | Delta clean | Delta corrupt | ECE corrupt | NLL corrupt | completion | epi churn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `source` | 0.7317 | 0.4807 | 0.0000 | 0.0000 | 0.0623 | 2.0309 | - | - |
| `episodic_only` | 0.7322 | 0.4806 | +0.0005 | -0.0001 | 0.0621 | 2.0267 | 0.0517 | 0.0878 |

## Insight 1 - El mecanismo se activa, pero no llega con fuerza al logit

E10 no falla por estar desconectado. Las senales internas son positivas:

- `completion_amount_corrupt_avg`: 0.0247-0.0755 en variantes semanticas,
  0.0508 en `episodic_only`, 0.0660 en `hippocampal_full`.
- `recall_sharpness_corrupt_avg`: 0.81-0.86 en la base principal, indicando
  recall concentrado y no uniforme.
- `episodic_buffer_churn_corrupt_avg`: 0.0547 en `episodic_only`, 0.0553 en
  `hippocampal_full`, 0.0616 en `consolidation_slow`.
- `traj_max_step_corrupt_avg`: finito y acotado, alrededor de 0.18-0.48 segun
  variante, sin evidencia de explosion del loop.

La lectura es clara: la memoria se activa en el espacio latente, pero la mezcla
conservadora `lambda_max=0.1` y el clasificador congelado lavan el efecto antes
de cambiar la decision de clase de manera robusta.

## Insight 2 - El codebook semantico no corrige: softear el argmin empeora

`assoc_recall_const` aisla el mecanismo minimo: hacer recall semantico suave
desde el codebook, sin gate biologico ni buffer episodico. Es la peor variante
de la base principal:

```text
source.corrupt_acc_avg             = 0.5029544099
assoc_recall_const.corrupt_acc_avg = 0.5021006126  (-0.0009)
```

Anadir familiaridad o unfamiliaridad reduce el dano, pero no lo convierte en
ganancia:

```text
assoc_recall_familiarity.corrupt_acc_avg   = 0.5024936304
assoc_recall_unfamiliarity.corrupt_acc_avg = 0.5025478398
```

Esto sugiere que el codebook aprendido ya es util cuando el VQ hace nearest
neighbor duro, pero su version Hopfield/soft no aporta una correccion limpia al
clasificador. La memoria semantica suavizada introduce una pequena deriva que
no esta alineada con las fronteras de clase.

## Insight 3 - El gate biologico discrimina shift, pero esta subutilizado

Phase 0 muestra que la familiaridad si mide degradacion: en
`e6_ema_kmeans_restart`, la mediana del gate cae de `0.497` en clean a `0.395`
en gaussian noise y `0.446` en pixel mask. Sin embargo, en el experimento final
esa senal solo escala una mezcla limitada por `lambda_max=0.1`.

El resultado practico:

- `assoc_recall_familiarity`: Delta corrupt `-0.0005`.
- `assoc_recall_unfamiliarity`: Delta corrupt `-0.0004`.
- `completion_T3_best_gate`: Delta corrupt `-0.0006`.

El gate esta vivo, pero no tiene suficiente autoridad. E10 valida la senal de
familiaridad como diagnostico; no valida aun que usarla como un multiplicador
suave de `lambda` mejore accuracy.

## Insight 4 - La via episodica es la unica pista positiva

`episodic_only` es la unica variante que mejora simultaneamente clean, corrupt,
ECE y NLL en la base principal:

```text
source:        clean=0.7523, corrupt=0.5030, ECE=0.0903, NLL=2.0224
episodic_only: clean=0.7557, corrupt=0.5051, ECE=0.0868, NLL=2.0084
```

La mejora no es suficiente para declararla como resultado positivo fuerte, pero
si cambia la direccion del roadmap. Cuando la memoria es **episodica** y se
actualiza online con EMA, no depende de que el codebook semantico sea una buena
base de recall para corrupciones. Esto encaja con la hipotesis CLS: el subsistema
rapido puede capturar regularidades locales del stream que la memoria semantica
congelada no corrige.

La misma via en SimVQ queda plana (`-0.0001` corrupt), pero alli el source ya
parte de menor robustez y el codebook esta colapsado; esa base no contradice el
resultado, mas bien limita que se puede atribuirle.

## Insight 5 - Consolidar no basta bajo esta interfaz

`hippocampal_full` y `consolidation_slow` combinan codebook semantico,
episodic buffer, completion T=3 y gate de familiaridad. En teoria deberian ser
la forma mas completa de la tesis biologica. En practica:

```text
hippocampal_full.corrupt_acc_avg   = 0.5031034857  (+0.0001)
consolidation_slow.corrupt_acc_avg = 0.5034558465  (+0.0005)
```

La consolidacion lenta ayuda un poco frente a `hippocampal_full`, pero no supera
a `episodic_only`. La combinacion con recall semantico parece arrastrar hacia
abajo lo que el buffer episodico aporta. En este regimen, "mas biologico" no es
"mas efectivo": el cuello de botella es que todo entra como una pequena mezcla
en `zq_pool`, no como una regla de decision o recuperacion que altere el logit
con suficiente margen.

## Cierre

E10 cierra una tercera familia de memoria despues de E7b y E7c:

- E7b: memoria latente como regularizador sobre LayerNorm - segura pero
  estructuralmente inalcanzable.
- E7c: codebook plastic - alcanzable y preservable, pero sin impacto en
  accuracy bajo hiperparametros conservadores.
- E10: memoria asociativa/episodica TTA-only - mecanicamente activa, pero
  numericamente inerte salvo una senal episodica marginal.

La conclusion no es que la tesis Q5 este muerta. Es mas precisa: **la memoria
esta viva como mecanismo interno, pero no cambia la prediccion cuando se la
inyecta solo como soft-mix pequeno antes de un clasificador congelado**.

## Proximo paso

No conviene agregar un cuarto mecanismo biologico con la misma interfaz. El
siguiente experimento debe atacar el cuello de botella:

1. **Retrieval no-parametrico que afecte el logit**: cache/kNN/test-time
   retrieval o re-ranking tipo Tip-Adapter, usando la via episodica que fue la
   unica net-positiva.
2. **Lambda derivado por muestra**: usar la familiaridad validada en Phase 0
   para permitir mezclas mas fuertes solo en muestras donde el trade-off de
   calibracion lo justifique.
3. **Proyeccion/denoising a manifold limpio**: usar el codebook como prior de
   correccion mas agresivo, no como mezcla `0.1` en `zq_pool`.

El criterio de exito para la siguiente etapa no debe ser solo "drift" o
"completion > 0"; debe ser demostrar que la memoria **llega al logit** con coste
de calibracion acotado.
