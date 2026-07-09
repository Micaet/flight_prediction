# Flight Delay Prediction

Projekt portfolio (Data Science → ML Engineering) na danych **US DOT BTS On-Time Performance**.
Zadanie: klasyfikacja binarna — czy lot spóźni się o więcej niż **X minut**.

> Mapa projektu, status i decyzje: **[`PLAN.md`](PLAN.md)**. Zasady pracy: **[`CLAUDE.md`](CLAUDE.md)**.

## Wymagania

- [uv](https://docs.astral.sh/uv/) (menedżer zależności i wersji Pythona)
- Python 3.11 (uv pobierze automatycznie)

## Uruchomienie

```bash
# 1. Odtworzenie środowiska z lock-a (identyczne wersje jak w repo)
uv sync

# 2. Pobranie danych BTS (domyślnie cały rok do data/raw/)
uv run python -m flight_delay.ingestion.bts --year 2024
#   wybrane miesiące:  --months 1 2 3
#   ponowne pobranie:  --overwrite

# 3. Praca w notebookach (EDA)
uv run jupyter lab
```

Pobieranie jest **idempotentne** — ponowny run pomija pliki, które już istnieją.

## Struktura

```
src/flight_delay/
  ingestion/       # pobieranie surowych danych (BTS, później OpenSky/Open-Meteo)
  transform/       # feature engineering (Faza A: pandas; Faza B: PySpark)
  training/        # trening XGBoost + MLflow
  serving/         # FastAPI /predict
notebooks/         # EDA i raport (Faza A)
tests/             # pytest
data/              # dane (poza gitem; odtwarzalne skryptem ingestion)
```

## Stack

Python 3.11 · uv · pandas/pyarrow · XGBoost · MLflow · PySpark · Airflow · FastAPI · Docker · pytest · GitHub Actions
