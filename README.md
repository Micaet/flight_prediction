# Flight Delay Prediction

Projekt na danych **US DOT BTS On-Time Performance**.
Klasyfikacja binarna — czy lot spóźni się o więcej niż **X minut**.


## Wymagania

- [uv](https://docs.astral.sh/uv/)
- Python 3.11

## Uruchomienie

```bash
# 1. Odtworzenie środowiska z lock-a (identyczne wersje jak w repo)
uv sync

# 2. Pobranie danych BTS
uv run python -m flight_delay.ingestion.bts --year 2024
#   wybrane miesiące:  --months 1 2 3
#   ponowne pobranie:  --overwrite

# 3. Praca w notebookach
uv run jupyter lab
```

## Struktura

```
src/flight_delay/
  ingestion/       # pobieranie surowych danych 
  transform/       # feature engineering (Faza A: pandas; Faza B: PySpark)
  training/        # trening XGBoost + MLflow
  serving/         # FastAPI /predict
notebooks/         # EDA i raport 
tests/             # pytest
data/              # dane
```

## Stack

Python 3.11 · uv · pandas/pyarrow · MLflow · PySpark · Airflow · FastAPI · Docker
