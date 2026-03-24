# Análisis de Sesgo Racial en el Dataset COMPAS

Este proyecto tiene como objetivo analizar y mitigar el sesgo racial en el sistema COMPAS (Correctional Offender Management Profiling for Alternative Sanctions), utilizado para predecir la reincidencia criminal.

## Estructura del Proyecto

El proyecto sigue una estructura profesional y escalable:

- `data/`: Datos crudos (`raw`), procesados (`processed`) y externos (`external`).
- `notebooks/`: Cuadernos Jupyter para análisis exploratorio, identificación de sesgo y experimentos.
- `src/`: Código fuente modularizado para carga de datos, ingeniería de características (features), modelado y visualización.
- `models/`: Modelos entrenados y serializados.
- `reports/`: Borrador del artículo científico y figuras generadas.
- `docs/`: Documentación adicional.
- `tests/`: Pruebas unitarias para el código de `src/`.

## Requisitos

Para instalar las dependencias necesarias, ejecuta:

```bash
pip install -r requirements.txt
```

## Dataset

El dataset principal es **COMPAS Recidivism**, obtenido originalmente por ProPublica. Se centra en la predicción de reincidencia a dos años y en la comparación de tasas de falsos positivos entre individuos afroamericanos y caucásicos.

## Metodología

1. **Definición del Problema**: Justificación de la importancia del sesgo en algoritmos judiciales.
2. **EDA**: Análisis profundo de los datos e identificación de sesgos existentes.
3. **Baseline**: Entrenamiento de un modelo estándar.
4. **Intervención**: Aplicación de técnicas de Fairness (mitigación de sesgo) y explicabilidad (XAI).
5. **Evaluación**: Comparativa de resultados antes y después de aplicar técnicas de equidad.
6. **Discusión**: Reflexión sobre las implicaciones éticas y legales.

---
*Proyecto para la asignatura: Aprendizaje Automático en Problemas del Mundo Real.*