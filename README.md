# Proyecto Final: Análisis de Sesgo Racial en COMPAS

Este repositorio contiene el código fuente para el proyecto final de la asignatura "Aprendizaje Avanzado". El objetivo es evaluar y mitigar el sesgo racial en la predicción de reincidencia utilizando el dataset COMPAS, simulando el impacto de un sistema de soporte a decisiones judiciales.

## Estructura del Proyecto

- `data/`: Directorio para almacenar el dataset (RAW y procesado).
- `notebooks/`: Cuadernos Jupyter o scripts para análisis exploratorio (EDA) y experimentación paso a paso.
- `src/`: Código fuente modular con la lógica de preprocesamiento, modelado y evaluación.
- `informe/`: Código fuente LaTeX del informe final.

## Instalación

Se recomienda crear un entorno virtual e instalar las dependencias exactas usando:

```bash
pip install -r requirements.txt
```

## Reproducibilidad

Para garantizar la reproducibilidad de los resultados (como se exige en el enunciado), todos los scripts y módulos establecen semillas aleatorias fijas (`SEED = 42`).