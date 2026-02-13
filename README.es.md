# Predicción de Deserción Escolar en Argentina

## Descripción General

Este proyecto desarrolla un **Sistema de Alerta Temprana (SAT)** para identificar estudiantes en riesgo de deserción escolar en Argentina, utilizando datos de la Encuesta Permanente de Hogares (EPH).

### Contexto

La educación en Argentina está garantizada por la Constitución Nacional y regulada por la Ley N.º 26.206 (Ley de Educación Nacional). La educación obligatoria abarca 14 años consecutivos, desde educación inicial hasta la educación secundaria. Si bien la educación primaria logra una cobertura casi universal (99% de asistencia), la finalización de la educación secundaria sigue siendo un desafío—aproximadamente el 20% de los adultos jóvenes de 18 a 24 años no han completado la educación obligatoria.

La tasa de deserción en el nivel primario es baja (0,41%), pero aumenta significativamente en el nivel secundario (7,66%), afectando al 16,8% de los estudiantes en su último año de educación obligatoria.

### Objetivos del Proyecto

1. Desarrollar un modelo predictivo de deserción escolar para potenciar el SAT.
2. Construir un panel interactivo para analizar datos educativos de la EPH.
3. Integrar el SAT en el panel de análisis para obtener información accionable.

### Mejoras Respecto a Trabajos Anteriores

Este proyecto mejora la investigación anterior (tesis de Michel Torino) de varias formas clave:

- **Seguimiento Longitudinal**: Aprovecha la estructura de panel 2-2-2 de la EPH (hogares rastreados durante 18 meses) para construir trayectorias de deserción. El trabajo anterior se basaba en instantáneas puntuales.
- **Población Objetivo**: Se enfoca en edades 15+ para capturar deserciones de nivel secundario y participación en educación de adultos, en lugar de poblaciones teóricas en edad escolar.
- **Estrategia de Imputación**: Explora técnicas avanzadas de imputación para variables de ingresos y deciles de ingresos, más allá de la simple imputación de media.

---

## Estructura del Repositorio

### Estándares de Arquitectura

- **Modularidad**: La lógica se organiza en `src/` por dominio:
  - `src/data/`: Carga, limpieza y preprocesamiento de datos
  - `src/features/`: Funciones de ingeniería de características
  - `src/models/`: Entrenamiento, evaluación y gestión de modelos
  - `src/pipelines/`: Orquestación de ETL y entrenamiento
  - `src/interpret/`: Explicabilidad (SHAP, planificado)
  - `src/api/`: Puntos finales de predicción de FastAPI (planificado)
  - `src/dashboard/`: Panel analítico Plotly

- **Calidad de Código**:
  - Type hints en todas las funciones
  - Docstrings estilo Google para clases y métodos
  - Funciones de preprocesamiento puro (DataFrame entrada → DataFrame salida, sin efectos secundarios)
  - Todo el código y documentación en inglés

- **Mejores Prácticas de ML**:
  - Sin fuga de datos (información futura excluida)
  - Validación cruzada estratificada K-fold para manejo de clases desbalanceadas

### Distribución de Directorios

```
desercionAR/
├── src/
│   ├── __init__.py
│   ├── data/                    # Carga, limpieza, preprocesamiento
│   ├── features/                # Ingeniería de características
│   ├── models/                  # Trainer, Evaluator, factories de preprocesamiento
│   │   └── preprocessing/       # Factories de Imputer, Scaler, Encoder
│   ├── pipelines/               # Orquestación de ETL y entrenamiento
│   ├── interpret/               # Lógica de explicabilidad (planificado)
│   ├── api/                     # Puntos finales de FastAPI (planificado)
│   ├── dashboard/               # Aplicación Plotly
│   └── cli/                     # Puntos de entrada CLI
├── configs/
│   ├── etl_pipeline.yaml        # Configuración de ETL
│   ├── training.yaml            # Configuración de entrenamiento completa
│   └── training_light.yaml      # Configuración de entrenamiento ligero
├── data/
│   ├── raw/                     # Datos EPH originales
│   ├── processed/               # Salida de ETL (train.csv, test.csv, predict.csv)
│   └── stage/                   # Artefactos de procesamiento intermedio
├── models/                      # Puntos de control de modelos y registro
├── notebooks/                   # Cuadernos experimentales (fechados)
├── figures/                     # Visualizaciones generadas
├── tests/                       # Pruebas unitarias e integración
├── pyproject.toml               # Dependencias del proyecto
├── Dockerfile
└── README.md
```

---

## Tubería de ETL

La tubería de ETL (`src/pipelines/etl.py`) procesa datos brutos de la encuesta EPH en conjuntos de datos listos para análisis para la predicción de deserción.

### Etapas de la Tubería

1. **Extracción**: Descarga microdatos brutos de EPH individuales y de hogares usando `pyeph`; selecciona características predefinidas.
2. **Transformación**: Filtra estudiantes de 14+ años que no han completado la educación secundaria; aplica transformaciones de ingeniería de características desde `configs/etl_pipeline.yaml`.
3. **Carga**: Genera variable objetivo (indicador de deserción), divide datos cronológicamente en conjuntos train/test/predict, previene fuga de datos, aplica limpieza final (homogenización binaria, manejo de valores faltantes). Genera tres archivos CSV.

### Conjuntos de Datos de Salida

- `train.csv`: Datos de entrenamiento con variable objetivo para desarrollo del modelo
- `test.csv`: Conjunto de prueba para evaluación final del modelo
- `predict.csv`: Datos sin etiquetar para predicciones en producción

### Inicio Rápido

```bash
# Generar todos los conjuntos de datos usando la configuración predeterminada de configs/etl_pipeline.yaml
PYTHONPATH=$PWD/src poetry run python -m src.cli.etl

# Generar conjuntos de datos con un archivo de configuración personalizado
PYTHONPATH=$PWD/src poetry run python -m src.cli.etl --config-path configs/etl_pipeline.yaml
```

---

## Tubería de Entrenamiento

La tubería de entrenamiento (`src/pipelines/training.py`) orquesta el entrenamiento del modelo, evaluación y seguimiento de experimentos mediante MLflow.

### Componentes

- **Trainer** (`trainer.py`): Construye pipelines de sklearn combinando preprocesamiento (imputación, escalado, codificación) con clasificadores. Ejecuta GridSearchCV en redes de hiperparámetros configurables; registra resultados en MLflow.

- **Evaluator** (`evaluator.py`): Calcula métricas del conjunto de prueba (F1, precisión, recall, AUC); genera matrices de confusión; registra artefactos en MLflow.

- **TrainingPipeline** (`training.py`): Orquesta el flujo de trabajo end-to-end—configura experimento de MLflow, entrena múltiples modelos en ejecuciones paralelas, evalúa cada uno, y registra el mejor en el Registro de Modelos.

### Configuración

El entrenamiento se controla mediante YAML:

```yaml
mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "dropout_prediction"

data:
  train_path: "data/processed/train.csv"
  test_path: "data/processed/test.csv"

preprocessing:
  imputer:
    enabled: true
  scaler:
    enabled: true
  encoder:
    enabled: true

models:
  - type: "logistic_regression"
    hyperparams:
      C: [0.001, 0.01, 0.1, 1.0]
      penalty: ['l1', 'l2']
  - type: "random_forest"
    hyperparams:
      n_estimators: [8, 20, 38]

training:
  cv_folds: 5
  scoring: "f1"
```

### Inicio Rápido

```bash
# Entrenar con la configuración predeterminada de configs/training.yaml
PYTHONPATH=$PWD/src poetry run python -m src.cli.training

# Entrenar con un archivo de configuración personalizado
PYTHONPATH=$PWD/src poetry run python -m src.cli.training --config configs/training_light.yaml
```

### Integración de MLflow

- Todos los hiperparámetros y puntuaciones de CV se registran por modelo
- Métricas del conjunto de prueba y matrices de confusión se registran en la interfaz de MLflow
- El mejor modelo se registra en el Registro de Modelos de MLflow para servicio
- Acceder a interfaz: `mlflow ui --backend-store-uri file:./mlruns`

---

## Despliegue

Este proyecto está diseñado para ser desplegado usando Docker y Docker Compose. La pila completa de la aplicación, incluyendo el servidor de MLflow, la API de predicción FastAPI y el panel de control de Plotly Dash, se puede ejecutar en una única instancia en la nube (por ejemplo, una `t3.small` de AWS EC2 o superior).

### Servicios
- **Servidor MLflow:** Accesible en el puerto `5000`. Sirve la interfaz de usuario de MLflow y realiza el seguimiento de los experimentos. Utiliza un bucket de S3 para el almacenamiento de artefactos y un volumen persistente para los metadatos.
- **FastAPI:** Accesible en el puerto `8000`. Proporciona un endpoint `/predict` para obtener predicciones de deserción del último modelo registrado.
- **Dashboard:** Accesible en el puerto `8050`. Una aplicación interactiva de Plotly Dash para la visualización y análisis de datos.

### Inicio Rápido (en EC2)

1.  **Prerrequisitos:** Una instancia EC2 con Docker y Docker Compose instalados.
2.  **Clonar el repositorio:** `git clone https://github.com/jbeiroa/desercionAR`
3.  **Configurar el Entorno:** Configurar las credenciales de AWS (por ejemplo, a través de un rol de IAM asociado a la instancia) con acceso a los buckets de S3 necesarios.
4.  **Construir y Ejecutar:**
    ```bash
    cd desercionAR
    docker-compose build
    docker-compose up -d
    ```
5.  **Acceder a los Servicios:** Asegurarse de que el grupo de seguridad de la instancia permita el tráfico entrante en los puertos 5000, 8000 y 8050.

---

## Configuración y Dependencias

### Requisitos

- Python 3.12.3
- Poetry 1.6.1+

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/jbeiroa/desercionAR
cd desercionAR

# Instalar dependencias
poetry install

# Activar entorno
eval $(poetry env activate)
```

---

## Desarrollo

### Pruebas

Ejecutar pruebas después de cualquier refactorización:

```bash
pytest tests/
```

### Agregar Nuevos Modelos

Para agregar un nuevo tipo de modelo:

1. Agregar clase de modelo a `MODEL_REGISTRY` en `trainer.py`
2. Definir hiperparámetros en `training.yaml` bajo `models`
3. Ejecutar tubería de entrenamiento

### Extender Preprocesamiento

Para agregar preprocesamiento personalizado:

1. Crear función factory en `preprocessing/`
2. Registrar en `PREPROCESSOR_REGISTRY` en `trainer.py`
3. Referenciar en `training.yaml`

---

## Cronograma del Proyecto y Estado

- **v0.1.0**: ETL inicial y modelos de línea base
- **v0.2.0**: Ingeniería de características y refinamiento del modelo
- **Actual**: Integración de MLflow, arquitectura modular, entrenamiento multi-modelo

---

## Contacto y Colaboración

Para preguntas o contribuciones, abra un issue o contacte a los mantenedores del proyecto.

---

## Licencia

Este proyecto está licenciado bajo la Licencia Creative Commons Atribución-NoComercial 4.0 Internacional. Consulte el archivo [LICENSE](LICENSE) para más detalles.
