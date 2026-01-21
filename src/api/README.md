# School Dropout Prediction API

This directory contains the FastAPI application for serving the school dropout prediction model.

## How to run the API

1. Make sure you have all the dependencies installed. If not, run `poetry install` from the root directory of the project.

2. Run the API using `uvicorn`:
```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Endpoints

- `GET /health`: Health check endpoint.
- `POST /predict`: Prediction endpoint.

### `POST /predict`

This endpoint takes a JSON payload with a `data` key, which is a list of records. Each record is a dictionary of feature names and values.

**Example Payload:**
```json
{
  "data": [
    {
      "ANO4": 2022,
      "II1": 1,
      "II2": 1,
      "II3": 1,
      "II4_1": 1,
      "II4_2": 1,
      "II4_3": 1,
      "II8": 1,
      "II9": 1,
      "IV1": 1,
      "IV2": 1,
      "IV3": 1,
      "IV4": 1,
      "IV5": 1,
      "IV6": 1,
      "IV7": 1,
      "IV9": 1,
      "IV11": 1,
      "IV12_2": 1,
      "IX_MEN10": 1,
      "IX_TOT": 1,
      "V1": 1,
      "V11": 1,
      "V12": 1,
      "V13": 1,
      "V14": 1,
      "V2": 1,
      "V21": 1,
      "V22": 1,
      "V3": 1,
      "V5": 1,
      "V6": 1,
      "V7": 1,
      "V8": 1,
      "AGLOMERADO": 1234567.89,
      "CAT_INAC": 1,
      "CH03": 1,
      "CH04": 1,
      "CH06": 20,
      "CH07": 1,
      "CH08": 1,
      "CH09": 1,
      "CH11": 1,
      "CH15": 1,
      "CH16": 1,
      "COMPONENTE": 1,
      "DECCFR": 1,
      "ESTADO": 1,
      "NIVEL_ED": 1,
      "NRO_HOGAR": 1,
      "PP02E": 1,
      "PP02H": 1,
      "PONDERA": 1,
      "REGION": 1,
      "TRIMESTRE": 1,
      "APORTES_JUBILATORIOS_jefx": 1,
      "CAT_OCUP_jefx": 1,
      "CH06_jefx": 1,
      "CONYUGE_TRABAJA": 1,
      "ESTADO_conyuge": 1,
      "ESTADO_jefx": 1,
      "HOGAR_MONOP": 1,
      "JEFA_MUJER": 1,
      "NBI_COBERTURA_PREVISIONAL": 1,
      "NBI_DIFLABORAL": 1,
      "NBI_HACINAMIENTO": 1,
      "NBI_SANITARIA": 1,
      "NBI_TENENCIA": 1,
      "NBI_TRABAJO_PRECARIO": 1,
      "NBI_VIVIENDA": 1,
      "NBI_ZONA_VULNERABLE": 1,
      "NIVEL_ED_jefx": 1,
      "PP04B1_jefx": 1,
      "PP07I_jefx": 1,
      "PP02E_jefx": 1,
      "CODUSU": "TQRMNPRXJHABTRCCCJJXCY00",
      "PP04B1_jefx": 1,
      "servicio_domestico": 1,
      "ratio_ocupados": 0.5
    }
  ]
}
```

**Example Response:**
```json
{
  "predictions": [
    {
      "prediction": 1,
      "probability": 0.85
    }
  ]
}
```
**Note:** The example payload above is not a real example. You need to provide a valid payload with all the required features. You can get the list of features from the `train.csv` file in `data/processed`.
