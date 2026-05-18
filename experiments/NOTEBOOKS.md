# Notebook lineage hacia E5

Este directorio conserva solo los notebooks activos que permiten reconstruir la ruta tecnica hasta `experiments/atracctor/e5_final_clean.ipynb`, el notebook final del modelo ganador.

## Orden recomendado

1. `VQ/baseline.ipynb`
   - Baseline justo ResNet18 vs DeMemte + VQ-VAE.
   - Sirve como referencia metodologica para comparar clean/corrupt con el mismo protocolo.

2. `VQ/dememte_variants.ipynb`
   - Notebook reducido de la variante `dememte_transformer`.
   - Define el punto de partida VQ espacial con atencion antes de pasar a memoria atractora.

3. `atracctor/attractor_memory.ipynb`
   - Notebook principal de experimentacion DeMemteAttractor.
   - Contiene el screening de ablations E0-E5 y genera los artefactos de la corrida final.

4. `atracctor/no_ood_debug.ipynb`
   - Debug aislado de la variante sin senal OOD.
   - Se conserva porque documenta el comportamiento del gate que motivo la calibracion final.

5. `atracctor/e5_final_clean.ipynb`
   - Notebook final limpio.
   - Carga el checkpoint E5, lee metricas guardadas, grafica curvas del gate y permite reevaluar o reentrenar E5 de forma opcional.

## Archivo historico

Los notebooks antiguos, duplicados o laterales se movieron a `archive/notebooks/` para no contaminar el flujo activo. No forman parte de la ruta principal hacia E5.
