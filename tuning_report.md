# Tuning Report — StreamLoop Churn Model

## Resumen

Ajuste de hiperparámetros de un **Random Forest Classifier** para predecir cancelación de clientes (churn) en una plataforma de streaming. El dataset contiene 7043 clientes con ~26.5% de tasa de churn.

**Métrica principal:** Recall (sensibilidad) — el negocio prioriza detectar clientes que cancelarán, aunque ello genere algunos falsos positivos.

---

## Línea Base

Modelo: `RandomForestClassifier` con **hiperparámetros por defecto** (random_state=42), dentro de un `Pipeline` con `ColumnTransformer` para preprocesamiento (OneHotEncoder + SimpleImputer).

| Métrica | Baseline (CV, train) | Baseline (test) |
|---|---|---|
| Recall | 0.456 ± 0.012 | 0.468 |
| Precision | 0.674 ± 0.026 | 0.625 |
| F1-score | 0.544 ± 0.013 | 0.535 |
| ROC-AUC | 0.826 ± 0.011 | 0.821 |

---

## Proceso de Búsqueda

### Fase 1 — RandomizedSearchCV (12 iteraciones, 5-fold CV)

Se exploró un espacio amplio de 6 hiperparámetros:
- `n_estimators`: 100, 200, 300, 400
- `max_depth`: None, 8, 12, 16, 24, 32
- `min_samples_split`: randint(2, 21)
- `min_samples_leaf`: randint(1, 11)
- `max_features`: sqrt, log2
- `class_weight`: None, balanced, balanced_subsample

**Mejor resultado:** Recall = 0.849 ± 0.025

### Fase 2 — GridSearchCV (27 combinaciones, 5-fold CV)

Refinamiento alrededor de la mejor región identificada por RandomizedSearch:
- `n_estimators`: 250, 300, 350
- `max_depth`: 7, 8, 9
- `min_samples_leaf`: 2, 3, 4
- `min_samples_split`: 13
- `max_features`: log2
- `class_weight`: balanced_subsample

**Mejor resultado:** Recall = 0.854 ± 0.016

---

## Análisis de cv_results_ — Estabilidad entre folds

Al revisar `cv_results_` se observó que las mejores configuraciones tenían desviaciones estándar pequeñas (≈0.016), indicando comportamiento consistente entre folds. La configuración elegida (rank 1) fue la de mayor recall promedio, pero las del top 3 tenían desviaciones similares, lo que respalda la decisión sin necesidad de sacrificar rendimiento por estabilidad.

---

## Modelo Final

### Hiperparámetros óptimos

| Parámetro | Valor |
|---|---|
| `n_estimators` | 250 |
| `max_depth` | 7 |
| `min_samples_leaf` | 3 |
| `min_samples_split` | 13 |
| `max_features` | `log2` |
| `class_weight` | `balanced_subsample` |

### Justificación de la elección

Se eligió la configuración de **rank 1** (mayor recall promedio en CV). Aunque otras configuraciones del top 3 tenían rendimiento similar, la elegida ofrece el mejor recall con desviación estándar baja (0.016), sin necesidad de comprometer por estabilidad.

- **`balanced_subsample`** — clave para el desbalance 26.5%/73.5%. Re-pondera muestras en cada árbol, forzando atención a la clase minoritaria.
- **`max_depth=7`** — evita overfitting. Con ~20 features, profundidad 7 captura interacciones sin memorizar ruido.
- **`min_samples_leaf=3` y `min_samples_split=13`** — regularización adicional que evita splits espurios.
- **`n_estimators=250`** — suficiente para estabilizar el error sin costo computacional excesivo.
- **`max_features='log2'`** — reduce correlación entre árboles para mejor generalización.

### Trade-off considerado

El aumento de recall (0.468 → 0.791) viene acompañado de:
- Caída en precisión (0.625 → 0.457) — más falsos positivos
- Caída en accuracy (0.784 → 0.695)
- Ligera caída en ROC-AUC (0.821 → 0.789)

Este trade-off es aceptable porque el negocio prefiere **falsos positivos (ofrecer retención innecesaria) sobre falsos negativos (perder clientes no detectados)**.

---

## Comparativa Final

| Métrica | Baseline (test) | Modelo Ajustado (test) | Mejora |
|---|---|---|---|
| **Recall** | **0.468** | **0.791** | **+0.323** |
| Precision | 0.625 | 0.457 | -0.168 |
| F1-score | 0.535 | 0.579 | +0.044 |
| Accuracy | 0.784 | 0.695 | -0.089 |
| ROC-AUC | 0.821 | 0.789 | -0.032 |

El modelo ajustado **identifica ~79 de cada 100 clientes que cancelarán**, frente a solo ~47 del baseline — una mejora de **+32.3 puntos porcentuales** en recall.

---

## Archivos

- `notebooks/streamloop_churn_tuning.ipynb` — Notebook completo con EDA, baseline, búsqueda, y evaluación final
- `tuning_report.md` — Este reporte
