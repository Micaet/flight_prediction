"""Pobieranie miesięcznych plików BTS "Reporting Carrier On-Time Performance".

Źródło: US DOT Bureau of Transportation Statistics, katalog PREZIP z gotowymi
paczkami ZIP (jeden plik = jeden miesiąc). Każdy ZIP zawiera duży CSV z lotami
danego miesiąca oraz `readme.html`.

Uruchomienie z linii poleceń (pobiera cały rok 2024 do data/raw/):

    uv run python -m flight_delay.ingestion.bts --year 2024

Idempotentność: `download_month` pomija pobieranie, jeśli docelowy ZIP już
istnieje i ma niezerowy rozmiar — ponowny run nie ściąga danych drugi raz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

# Wzorzec URL katalogu PREZIP. Miesiąc BEZ zera wiodącego (1..12), rok czterocyfrowy.
BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Domyślny katalog na surowe dane (względem korzenia repo).
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# Nagłówek User-Agent — serwer BTS bywa wrażliwy na brak przeglądarkowego UA.
_HEADERS = {"User-Agent": "Mozilla/5.0 (flight-delay-prediction; dane edukacyjne)"}


def month_url(year: int, month: int) -> str:
    """Zwraca URL paczki ZIP dla danego roku i miesiąca."""
    return BASE_URL.format(year=year, month=month)


def download_month(
    year: int,
    month: int,
    dest_dir: Path = DEFAULT_RAW_DIR,
    *,
    overwrite: bool = False,
    timeout: int = 120,
) -> Path:
    """Pobiera jeden miesiąc danych BTS do `dest_dir` i zwraca ścieżkę do ZIP-a.

    Args:
        year: rok (np. 2024).
        month: miesiąc 1..12 (bez zera wiodącego w URL).
        dest_dir: katalog docelowy; tworzony, jeśli nie istnieje.
        overwrite: gdy False (domyślnie), pomija pobieranie istniejącego pliku
            o niezerowym rozmiarze (idempotentność).
        timeout: limit czasu połączenia w sekundach.

    Returns:
        Ścieżka do pobranego pliku ZIP.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"bts_ontime_{year}_{month:02d}.zip"

    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        print(f"[skip] {dest.name} już istnieje ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    url = month_url(year, month)
    # Zapis do pliku tymczasowego i atomowe przeniesienie — przerwane pobieranie
    # nie zostawia uszkodzonego pliku wyglądającego na kompletny.
    tmp = dest.with_suffix(".zip.part")
    with requests.get(url, headers=_HEADERS, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with (
            open(tmp, "wb") as fh,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=f"{year}-{month:02d}",
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)
    return dest


def download_year(
    year: int,
    dest_dir: Path = DEFAULT_RAW_DIR,
    *,
    months: list[int] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Pobiera wskazane miesiące (domyślnie 1..12) danego roku."""
    months = months or list(range(1, 13))
    return [
        download_month(year, m, dest_dir, overwrite=overwrite) for m in months
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pobierz dane BTS On-Time Performance.")
    p.add_argument("--year", type=int, required=True, help="Rok, np. 2024")
    p.add_argument(
        "--months",
        type=int,
        nargs="*",
        default=None,
        help="Miesiące 1..12 (domyślnie cały rok)",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Katalog docelowy (domyślnie data/raw/)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Pobierz ponownie, nawet jeśli plik istnieje",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    paths = download_year(
        args.year, args.dest, months=args.months, overwrite=args.overwrite
    )
    total_mb = sum(p.stat().st_size for p in paths) / 1e6
    print(f"\nGotowe: {len(paths)} plików, razem {total_mb:.1f} MB w {args.dest}")


if __name__ == "__main__":
    main()
