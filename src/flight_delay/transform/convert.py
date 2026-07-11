from pathlib import Path
import zipfile
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

KEEP_COLS = [
    # czas
    "FlightDate", "Month", "DayOfWeek",
    # przewoźnik
    "Reporting_Airline", "Tail_Number",
    # trasa
    "Origin", "Dest", "OriginState", "DestState",
    # rozkładowe — znane PRZED lotem => bezpieczne cechy modelu
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "DistanceGroup",
    # wykonanie — PO locie => tylko EDA, NIE cechy (leakage)
    "DepDelay", "DepDelayMinutes", "DepDel15",
    "ArrDelay", "ArrDelayMinutes", "ArrDel15",
    "ActualElapsedTime", "AirTime", "TaxiOut", "TaxiIn",
    # status
    "Cancelled", "Diverted", "CancellationCode",
    # przyczyny opóźnień — PO locie => złoto do EDA "co napędza"
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]

DTYPES = {
    "Reporting_Airline": "category",
    "Tail_Number": "category",
    "Origin": "category",
    "Dest": "category",
    "OriginState": "category",
    "DestState": "category",
    "CancellationCode": "category",    
    "Month": "int8",
    "DayOfWeek": "int8",
    "DistanceGroup": "int8",
    "CRSDepTime": "int16",
    "CRSArrTime": "int16",
    "CRSElapsedTime": "float32",
    "ActualElapsedTime": "float32",
    "AirTime": "float32",
    "Distance": "float32",
    "TaxiOut": "float32",
    "TaxiIn": "float32",
    "DepDelay": "float32",
    "DepDelayMinutes": "float32",
    "DepDel15": "float32",
    "ArrDelay": "float32",
    "ArrDelayMinutes": "float32",
    "ArrDel15": "float32",
    "CarrierDelay": "float32",
    "WeatherDelay": "float32",
    "NASDelay": "float32",
    "SecurityDelay": "float32",
    "LateAircraftDelay": "float32",
    "Cancelled": "float32",
    "Diverted": "float32",
}

def read_raw_month(year: int, month: int) -> pd.DataFrame:
    '''Read a raw flight data zip file for a given year and month, returning a DataFrame with selected columns and dtypes.'''
    zpath = RAW_DIR / f"bts_ontime_{year}_{month:02d}.zip"
    with zipfile.ZipFile(zpath) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as fh:
            df = pd.read_csv(fh,usecols=KEEP_COLS,dtype=DTYPES,parse_dates=["FlightDate"], low_memory=False,
        )
        df["Cancelled"] = df["Cancelled"].astype(bool)
        df["Diverted"] = df["Diverted"].astype(bool)       
    return df

def write_month(df: pd.DataFrame, month: int, out_dir: Path = OUT_DIR) -> None:
    '''Write a DataFrame to a Parquet file for a given month.'''
    out_dir.mkdir(parents=True, exist_ok=True)
    out= out_dir / f"flights_{month:02d}.parquet"
    df.to_parquet(out, engine="pyarrow", index=False)
    return 


def convert_year(year: int, months: list[int] | None = None) -> None:
    '''Convert raw flight data for a given year and list of months to Parquet format.'''
    if months is None:
        months = list(range(1, 13))
    for month in months:
        df = read_raw_month(year, month)
        write_month(df, month)

if __name__ == "__main__":
    convert_year(2024, months=list(range(1, 13)))