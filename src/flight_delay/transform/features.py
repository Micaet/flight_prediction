"""Feature engineering (prototype in `notebooks/FE.ipynb`).

Prediction happens before departure, so only schedule columns are allowed in.
`SAFE_RAW` is a whitelist, so any new raw column stays out by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SAFE_RAW: list[str] = [
    "Month", "DayOfWeek",
    "Reporting_Airline",
    "Origin", "Dest", "OriginState", "DestState",
    "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
    "Distance", "DistanceGroup",
]

LEAKAGE: list[str] = [
    "DepDelay", "DepDelayMinutes", "DepDel15",
    "TaxiOut", "TaxiIn",
    "ActualElapsedTime", "AirTime",
    "ArrDelay", "ArrDelayMinutes", "ArrDel15",
    "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",
    "Cancelled", "Diverted", "CancellationCode",
]


SEASON_BY_MONTH: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}

DELAY_THRESHOLD_MIN: int = 15



def prepare_population(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cancelled (1.36%) and diverted (0.25%) flights - they have no arrival delay.
    """
    return df.loc[(df["Cancelled"] == False) & (df["Diverted"] == False), :].copy().reset_index(drop=True)


def make_label(df: pd.DataFrame, threshold: int = DELAY_THRESHOLD_MIN) -> pd.DataFrame:
    """y = 1 if ArrDelay >= threshold. Matches BTS `ArrDel15` exactly for 15 min."""
    df["y"] = df["ArrDelay"].ge(threshold).astype(int)
    return df


def add_free_features(flights: pd.DataFrame) -> pd.DataFrame:
    """Whitelist columns plus features computed from a single row.

    CRSDepTime is HHMM as an int, and 2400 means midnight -> DepHour 0.
    """
    df = flights[SAFE_RAW].copy()
    df["DepHour"] = (flights["CRSDepTime"] // 100).replace(24, 0)
    df["IsWeekend"] = flights["DayOfWeek"].isin([6, 7]).astype(int)
    df["Season"] = flights["Month"].map(SEASON_BY_MONTH)
    
    return df




M = 100  
PRIOR = None 

def add_history_feature(frame : pd.DataFrame, keys: str | list[str], m: int = M, prior: float | None = PRIOR) -> np.ndarray:
    """Delay rate for `keys` from days strictly before each flight, smoothed toward `prior`.

    `keys` is a column or a list, e.g. ["Origin", "Dest"] for a route.
    Returns a float32 array in `frame`'s row order.
    """
    prior = prior if prior is not None else frame["y"].mean()
    if isinstance(keys, str):
        keys = [keys]
    daily = (
        frame.groupby(keys + ["FlightDate"], observed=True)["y"]
        .agg(sum_y="sum", n="count")
        .reset_index()
        .sort_values("FlightDate")
    )
    g = daily.groupby(keys, observed=True)
    daily["hist_sum"] = g["sum_y"].cumsum() - daily["sum_y"]
    daily["hist_n"] = g["n"].cumsum() - daily["n"]
    daily["hist_rate"] = (daily["hist_sum"] + m * prior) / (daily["hist_n"] + m)
    out = frame[keys + ["FlightDate"]].merge(
        daily[keys + ["FlightDate", "hist_rate"]], on=keys + ["FlightDate"], how="left"
    )
    return out["hist_rate"].to_numpy(dtype="float32")



def add_congestion_feature(df: pd.DataFrame) -> np.ndarray:
    """Scheduled departures from the same airport, day and hour.

    Schedule only, so using the same day is fine here.
    """
    df["DepHour"] = (df["CRSDepTime"] // 100).replace(24, 0)
    df["OriginHourCongestion"] = (
    df.groupby(["Origin", "FlightDate", "DepHour"], observed=True)["DepHour"]
    .transform("size")
    .to_numpy(dtype="int32"))
    return df["OriginHourCongestion"].to_numpy(dtype="int32")


def build_features(
    df: pd.DataFrame,
    m: int = M,
    prior: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Raw BTS frame -> `(X, y, flight_date)`.

    `flight_date` is only for the time split, not a feature. NaNs are left
    for XGBoost to handle.
    """
    df = prepare_population(df)
    df = make_label(df)
    y = df["y"].to_numpy(dtype="int8")
    flight_date = df["FlightDate"].copy()
    X = add_free_features(df)
    X["CarrierHist"] = add_history_feature(df, "Reporting_Airline", m=m, prior=prior)
    X["OriginHist"] = add_history_feature(df, "Origin", m=m, prior=prior)
    X["RouteHist"] = add_history_feature(df, ["Origin", "Dest"], m=m, prior=prior)
    X["OriginHourCongestion"] = add_congestion_feature(df)
    X["OriginHourHist"] = add_history_feature(df, ["Origin", "DepHour"], m=m, prior=prior)
    return X, y, flight_date
