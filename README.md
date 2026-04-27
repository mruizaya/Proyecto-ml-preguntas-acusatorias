# Detección de preguntas acusatorias en contratación pública

Clasificador binario de texto en español que identifica preguntas con tono acusatorio dentro del proceso de observaciones de contrataciones públicas. Compara dos modelos lineales (Regresión Logística regularizada y SVM lineal) sobre representaciones TF-IDF más variables estilísticas, con validación cruzada repetida y prueba de hipótesis pareada.

## Contenido del repositorio

```
.
├── Copia_de_Copia_de_proyecto_ml_preguntas_acusatorias.ipynb   # Notebook principal
├── dataset.xlsx                                                # Dataset (no incluido)
└── README.md
```

El notebook está organizado en 20 secciones numeradas:

| # | Sección | Descripción |
|---|---------|-------------|
| 1 | Imports | Dependencias del proyecto |
| 2 | Carga del dataset | Lectura del archivo `dataset.xlsx` |
| 3 | Variables principales | Definición de columnas objetivo y predictoras |
| 4 | Limpieza y feature engineering | Normalización del texto y variables estilísticas |
| 5 | EDA básico | Distribuciones de clase y de longitud |
| 6 | `SafeSelectKBest` | Envoltorio robusto para selección filtrada |
| 7 | Train / Validation / Test split | Partición 60/20/20 estratificada |
| 8 | Preprocesador | `ColumnTransformer` con rama textual y rama estilística |
| 9 | Pipelines de modelos | LR y SVM lineal en `Pipeline` |
| 10 | Cross-validation | `RepeatedStratifiedKFold` 5×2 con métricas múltiples |
| 11–12 | Grid search | Búsqueda de hiperparámetros para ambos modelos |
| 13 | Optimización de λ | Gráfica de regularización |
| 14 | Curva de aprendizaje | Train vs. validación a tamaños crecientes |
| 15 | Optimización del threshold | Barrido de umbrales en validación |
| 16 | Comparación estadística | Wilcoxon y t-test pareados |
| 17 | Evaluación final en test | Métricas y matriz de confusión |
| 18 | Selección del mejor modelo | Comparación de AUC en test |
| 19 | Interpretabilidad | Coeficientes más relevantes |
| 20 | t-SNE | Visualización 2D del espacio de características |

## Dataset

El archivo `dataset.xlsx` debe ubicarse en la raíz del proyecto. Esquema esperado:

| Columna | Tipo | Descripción |
|---|---|---|
| `contract_id` | int | Identificador del proceso de contratación |
| `pregunta_id` | int | Identificador de la pregunta |
| `pregunta` | str | Texto de la observación / pregunta |
| `sum_pregunta_isAcusatoria` | int | Número de anotadores que la marcaron acusatoria |
| `final_pregunta_isAcusatoria` | int (0/1) | Etiqueta final binaria (objetivo) |

El dataset no se versiona en este repositorio.

## Requisitos

- Python ≥ 3.10
- Jupyter / JupyterLab o Google Colab

Dependencias principales:

```
numpy
pandas
matplotlib
scipy
scikit-learn
openpyxl
joblib
```

Instalación rápida:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scipy scikit-learn openpyxl joblib jupyterlab
```

## Ejecución

1. Colocar `dataset.xlsx` en la raíz del repositorio.
2. Abrir el notebook:

   ```bash
   jupyter lab Copia_de_Copia_de_proyecto_ml_preguntas_acusatorias.ipynb
   ```

3. Ejecutar las celdas en orden (`Run All`). Las semillas (`random_state=42`) están fijadas en todas las particiones, validaciones y modelos para garantizar reproducibilidad.

> **Nota sobre tiempos:** las celdas 11 y 12 (`GridSearchCV`) entrenan 540 combinaciones × 10 *folds* = 5400 ajustes por modelo. En CPU de 8 hilos pueden tardar entre 15 y 30 minutos cada una. Se recomienda usar `n_jobs=-1` (ya configurado).

## Decisiones de diseño

- **Pipelines de scikit-learn** en lugar de transformaciones manuales, para que la selección de variables y el TF-IDF se reajusten dentro de cada *fold* y no haya fuga de información hacia validación.
- **`ColumnTransformer`** que separa la rama de texto (TF-IDF + chi²) de la rama de variables estilísticas (`StandardScaler`), permitiendo que cada tipo de feature reciba el tratamiento adecuado.
- **Selección filtrada con chi²** dentro de un envoltorio `SafeSelectKBest` que recorta automáticamente el `k` cuando supera el número real de features disponibles.
- **`class_weight="balanced"`** en ambos clasificadores para compensar el desbalance del objetivo (~3% de positivos).
- **`RepeatedStratifiedKFold` 5×2** como compromiso entre estabilidad estadística (10 mediciones por configuración) y costo computacional, dado el tamaño de la grilla.
- **Calibración del umbral** sobre el conjunto de validación con barrido de F1 macro, en lugar del umbral 0.5 por defecto.
- **`TruncatedSVD` + `t-SNE`** para visualizar el espacio disperso de TF-IDF (la SVD reduce a 50 dimensiones antes del t-SNE para evitar artefactos en alta dimensión).

## Reproducibilidad

Todas las particiones, validaciones cruzadas y modelos usan `random_state=42`. Re-ejecutar el notebook con el mismo dataset debe producir los mismos resultados.

## Licencia

Proyecto académico. Sin licencia específica.
