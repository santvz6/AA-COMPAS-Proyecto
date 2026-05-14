# Análisis de Sesgo Racial en el Dataset COMPAS

Proyecto académico (asignatura **Aprendizaje Automático en Problemas del Mundo Real**, UA) cuyo objetivo es analizar y mitigar el sesgo racial del sistema COMPAS, replicando el estudio de ProPublica (Angwin et al., 2016) y extendiéndolo con técnicas modernas de *fairness*, explicabilidad (XAI) y robustez adversaria.

---

## 1. Sistema de ejecución (visión general)

El proyecto **no se ejecuta como un único script**: la lógica reutilizable vive en `src/` y los **notebooks de `notebooks/` actúan como orquestadores** del pipeline. Esta separación cumple los requisitos del enunciado (modularidad, reproducibilidad, sin *data leakage*).

```
 data/raw/compas-scores-two-years.csv          (descarga manual — ProPublica)
            │
            ▼
 src/data/make_dataset.py        ── clean_compas_data() ──►   data/processed/compas_cleaned.csv
            │
            ▼
 notebooks/01_EDA.ipynb           ── EDA: distribución racial, decile score, recidivismo
            │
            ▼
 notebooks/02_Modeling_and_Fairness.ipynb        (pipeline completo)
            │   ├── features/build_features.py   ── split + ColumnTransformer
            │   ├── models/train_model.py        ── baseline LR + ThresholdOptimizer (Fairlearn)
            │   ├── models/evaluate.py           ── MetricFrame, DP/EO/FPR diff, CV k=5
            │   ├── models/robustness.py         ── ataque FGM (ART, ε=0.1)
            │   └── visualization/plots.py       ── 6 figuras publication-ready
            ▼
 reports/figures/*.png            (alimentan el reporte LaTeX)
 reports/report.tex               (compilable a PDF con pdflatex/latexmk)
```

### Convenciones clave

- **Reproducibilidad**: `random_state = 80` (constante `SEED`) en splits y modelos.
- **Sin data leakage**: `train_test_split` se aplica **antes** de cualquier `fit`. El `ColumnTransformer` aprende medias, varianzas y categorías solo del *train*.
- **Atributo sensible**: `race`, filtrado a `African-American` y `Caucasian` (criterio ProPublica).
- **Target**: `two_year_recid` (reincidencia a 2 años).

---

## 2. Estructura de directorios

| Ruta | Contenido |
|---|---|
| `data/raw/` | CSV original de ProPublica (no versionado, ver §3). |
| `data/processed/` | Dataset limpio que produce `make_dataset.py`. |
| `data/external/` | Datos auxiliares externos (placeholder). |
| `notebooks/01_EDA.ipynb` | Análisis exploratorio: distribuciones por raza, *decile score*, recidivismo. |
| `notebooks/02_Modeling_and_Fairness.ipynb` | Pipeline completo: baseline → CV → 3 intervenciones de equidad → SHAP → robustez. |
| `src/data/make_dataset.py` | Limpieza con filtros ProPublica (`days_b_screening_arrest`, `is_recid`, `score_text`, etc.). |
| `src/features/build_features.py` | `get_data_splits()` y `build_preprocessor()` (numéricas: median + scaler; categóricas: most_frequent + OneHot). |
| `src/models/train_model.py` | `train_baseline()` (Regresión Logística) y `train_fair_model()` (Fairlearn `ThresholdOptimizer`, *post-processing*). |
| `src/models/evaluate.py` | `evaluate_model()`, `get_fairness_summary()`, `cross_validate_baseline()` (k=5 estratificada). |
| `src/models/robustness.py` | `evaluate_robustness()` con Fast Gradient Method (ART). |
| `src/visualization/plots.py` | CV, comparativa de equidad, *heatmap* FPR/FNR, *trade-off*, matrices de confusión, robustez. |
| `reports/figures/` | PNGs generados por los notebooks (entrada del LaTeX). |
| `reports/report.tex` | Manuscrito científico. `report.bib` para bibliografía. |
| `docs/` | `enunciado.pdf`, `propuesta.md`, `desarrollo.md`. |

---

## 3. Instalación

Recomendado: entorno virtual aislado (Python 3.10+).

```bash
# Clonar el repo y entrar
git clone <url-del-repo>
cd AA-COMPAS-Proyecto

# Entorno virtual
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# Dependencias
pip install -r requirements.txt
```

`requirements.txt` incluye:
- **Core**: numpy, pandas, scikit-learn, matplotlib, seaborn, jupyter.
- **Fairness / XAI**: fairlearn, aif360, shap, lime, dice-ml.
- **Robustez**: adversarial-robustness-toolbox, evidently.

### Descarga del dataset

El CSV crudo **no se incluye en el repositorio** (ver `.gitignore`). Hay que descargarlo manualmente desde el repositorio público de ProPublica y colocarlo en `data/raw/`:

```bash
# Desde la raíz del proyecto
curl -L -o data/raw/compas-scores-two-years.csv \
  https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
```

---

## 4. Ejecución paso a paso

### Paso 1 — Generar el dataset limpio

```bash
python src/data/make_dataset.py
```

Aplica los filtros de ProPublica y guarda `data/processed/compas_cleaned.csv` (~6 000 filas).

> Alternativa: el propio notebook `02_Modeling_and_Fairness.ipynb` invoca `clean_compas_data(...)` en su primera celda, por lo que **este paso es opcional** si se ejecuta el notebook completo.

### Paso 2 — EDA

```bash
jupyter lab notebooks/01_EDA.ipynb     # o jupyter notebook
```

Genera las visualizaciones de distribución racial, tasa de reincidencia por grupo y distribución del *decile score*. Útil para entender el sesgo presente en los datos antes de modelar.

### Paso 3 — Pipeline completo (modelado + fairness + XAI + robustez)

```bash
jupyter lab notebooks/02_Modeling_and_Fairness.ipynb
```

Recorrido del notebook:

1. **Carga + split 80/20** estratificado.
2. **Preprocesador** (`ColumnTransformer`) integrado en el `Pipeline` de sklearn.
3. **Baseline**: Regresión Logística (justificada por SHAP exacto, `predict_proba` calibrado, estabilidad con L-BFGS).
4. **Validación cruzada estratificada k=5** sobre el train.
5. **Métricas de equidad** del baseline: accuracy/precision/recall/F1/FPR/FNR por grupo, `demographic_parity_difference`, `equalized_odds_difference`.
6. **SHAP** con `LinearExplainer` (valores exactos) para detectar *proxy discrimination*.
7. **Tres intervenciones** vía `fairlearn.ThresholdOptimizer` (post-procesado):
   - `demographic_parity`
   - `false_positive_rate_parity`  ← métrica central del caso COMPAS
   - `equalized_odds`
8. **Comparativa** Accuracy vs. DP/FPR/EO + 4 visualizaciones (barras, heatmap, scatter trade-off, matrices de confusión).
9. **Robustez adversaria** con FGM (ART, ε = 0.1).

Todas las figuras se guardan automáticamente en `reports/figures/`.

### Paso 4 — Compilar el reporte

```bash
cd reports
pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex
# o, más cómodo:
latexmk -pdf report.tex
```

El PDF resultante consume las figuras de `reports/figures/`. Si se han regenerado los `.png`, el LaTeX las recoge automáticamente.

---

## 5. Cómo entender el código

Si vienes de cero, este es el orden recomendado de lectura:

1. **`docs/propuesta.md`** — qué se intenta resolver y por qué.
2. **`docs/desarrollo.md`** — estándares del equipo (semillas, anti-leakage, modularidad).
3. **`src/data/make_dataset.py`** — qué filas se descartan y por qué (criterio ProPublica).
4. **`src/features/build_features.py`** — dónde se evita el *data leakage*.
5. **`src/models/train_model.py`** — diferencia entre el baseline y el modelo mitigado.
6. **`src/models/evaluate.py`** — qué significa cada métrica de equidad.
7. **`notebooks/02_Modeling_and_Fairness.ipynb`** — pipeline ejecutándose de principio a fin con justificaciones inline.

Los notebooks contienen celdas markdown con la **justificación científica** de cada decisión (elección de modelo, métricas, valores de hiperparámetros), pensadas para ser citables en el reporte.

---

## 6. Salidas esperadas

Tras ejecutar el pipeline completo encontrarás en `reports/figures/`:

- `cv_results.png` — métricas de validación cruzada (media ± std).
- `shap_summary.png`, `shap_bar.png` — explicabilidad del baseline.
- `fairness_comparison.png` — barras Accuracy/DP/FPR/EO por enfoque.
- `group_metrics_heatmap.png` — FPR/FNR por grupo racial y estrategia.
- `tradeoff_scatter.png` — Accuracy vs. disparidad (zona óptima abajo-derecha).
- `confusion_matrices.png` — matrices normalizadas por enfoque.
- `robustness_fgm.png` (+ variantes por enfoque) — caída de accuracy ante FGM.

---

## 7. Equipo

Proyecto desarrollado por Santiago Álvarez, Taron Sargsyan y Joaquín Sigüenza Chilar para la asignatura **Aprendizaje Automático en Problemas del Mundo Real** (Universidad de Alicante).
