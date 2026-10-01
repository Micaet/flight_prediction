"""Download monthly BTS On-Time Performance ZIPs (one ZIP per month).

    uv run python -m flight_delay.ingestion.bts --year 2024

Months that are already on disk are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

# Month without a leading zero (1..12).
BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# BTS sometimes rejects requests without a browser-like User-Agent.
_HEADERS = {"User-Agent": "Mozilla/5.0 (flight-delay-prediction; educational use)"}


def month_url(year: int, month: int) -> str:
    return BASE_URL.format(year=year, month=month)


def download_month(
    year: int,
    month: int,
    dest_dir: Path = DEFAULT_RAW_DIR,
    *,
    overwrite: bool = False,
    timeout: int = 120,
) -> Path:
    """Download one month to `dest_dir` and return the ZIP path.

    Skips the download if a non-empty file is already there, unless `overwrite`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"bts_ontime_{year}_{month:02d}.zip"

    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    url = month_url(year, month)
    # Download to .part first so an interrupted run doesn't leave a half file
    # that the skip check above would treat as complete.
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
    months = months or list(range(1, 13))
    return [
        download_month(year, m, dest_dir, overwrite=overwrite) for m in months
    ]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download BTS On-Time Performance data.")
    p.add_argument("--year", type=int, required=True, help="e.g. 2024")
    p.add_argument(
        "--months",
        type=int,
        nargs="*",
        default=None,
        help="months 1..12 (default: whole year)",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="output dir (default: data/raw/)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="download again even if the file exists",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    paths = download_year(
        args.year, args.dest, months=args.months, overwrite=args.overwrite
    )
    total_mb = sum(p.stat().st_size for p in paths) / 1e6
    print(f"\nDone: {len(paths)} files, {total_mb:.1f} MB in {args.dest}")


if __name__ == "__main__":
    main()
