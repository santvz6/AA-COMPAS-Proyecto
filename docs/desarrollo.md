# Guía de Desarrollo para el Proyecto COMPAS

Esta guía proporciona instrucciones y estándares para el desarrollo del análisis de sesgo racial en el sistema COMPAS.

## Comandos de Desarrollo

- **Instalación de dependencias**: `pip install -r requirements.txt`
- **Ejecución de pruebas**: `pytest`
- **Linting**: `flake8 src/` (o el linter preferido)
- **Ejecución de experimentos**: Los cuadernos se encuentran en `notebooks/`. Los scripts modulares en `src/`.

## Contexto del Proyecto y Objetivos

El objetivo principal es analizar y mitigar el sesgo racial en las predicciones de reincidencia de COMPAS, replicando el análisis de ProPublica y extendiéndolo con técnicas modernas de equidad algoritímica.

### Objetivos Específicos
1.  **Análisis Comparativo**: Evaluar cuatro enfoques de optimización:
    - Maximizar Accuracy global.
    - Igualar Accuracy entre grupos raciales.
    - Igualar Tasas de Falsos Positivos (FPR).
    - Igualar Tasas de Falsos Negativos (FNR).
2.  **Explicabilidad (XAI)**: Identificar qué variables contribuyen más a la predicción y al sesgo.
3.  **Robustez**: Evaluar la estabilidad del modelo ante variaciones en los datos.
4.  **Marco Legal**: Discutir los resultados bajo la perspectiva de la legislación europea vigente sobre IA y justicia.

## Estándares de Código y Calidad

- **Reproducibilidad**: Es MANDATORIO fijar semillas (`random_state`) en todos los procesos estocásticos (entrenamiento, splits, etc.).
- **Estructura Modular**: La lógica de procesamiento, modelado y métricas debe residir en `src/`. Los `notebooks/` solo deben importar y usar estas funciones.
- **Evitar Data Leakage**: Garantizar una separación estricta entre entrenamiento y prueba. No realizar ingeniería de características basada en estadísticas de todo el dataset. **Especial atención a la codificación de variables: el agente debe asegurarse de aplicar `train_test_split` SIEMPRE antes de realizar transformaciones categóricas como `LabelEncoder` o `pd.get_dummies`. El ajuste (`fit`) debe hacerse exclusivamente sobre el conjunto de entrenamiento para evitar que el modelo conozca la distribución global.**
- **Documentación**: Usar docstrings en funciones y comentarios explicativos en los notebooks. Seguir un tono científico y formal, evitando lenguaje coloquial.

## Criterios de Evaluación Críticos

Para cumplir con los requisitos académicos del `enunciado.pdf`:
-   **EDA Profundo**: No basta con estadísticas básicas; se debe analizar el desbalanceo y los sesgos inherentes al origen de los datos.
-   **Baseline Sólido**: Entrenar un modelo de referencia sin intervenciones de fairness para comparar.
-   **Análisis Crítico**: La discusión técnica debe incluir implicaciones éticas y sociales reales.
-   **Código Reproducible**: El repositorio debe ser funcional de principio a fin siguiendo el README.

## Stack Tecnológico Sugerido
-   **Base**: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.
-   **Fairness**: `fairlearn` o `aif360`.
-   **Explicabilidad**: `shap` o `lime`.
-   **Reporte**: Formato científico (basado en `plantilla.tex`).