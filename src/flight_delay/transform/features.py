"""Feature engineering for the flight-delay model (S6).

The prototype lives in `notebooks/FE.ipynb`; this module is the reusable version.
Phase B (Spark / Airflow) wraps these functions instead of re-deriving the logic.

Horizon: we predict BEFORE departure, so no at-departure or post-flight column may
enter the feature matrix. `SAFE_RAW` is a whitelist rather than a blacklist so that
a newly appearing column drops out by default instead of sneaking in.
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
    """Restrict to completed flights: drop cancelled and diverted, reset the index.

    Cancelled (1.36%) and diverted (0.25%) flights have no arrival delay, so they
    can neither be labelled nor predicted; they are out of the model population.
    """
    return df.loc[(df["Cancelled"] == False) & (df["Diverted"] == False), :].copy().reset_index(drop=True)


def make_label(df: pd.DataFrame, threshold: int = DELAY_THRESHOLD_MIN) -> pd.DataFrame:
    """Binary label: 1 if the flight arrived `threshold`+ minutes late.

    Validated against the BTS `ArrDel15` column (agreement 1.0 row for row).
    """
    df["y"] = df["ArrDelay"].ge(threshold).astype(int)
    return df


def add_free_features(flights: pd.DataFrame) -> pd.DataFrame:
    """Whitelist columns plus the features derivable from a single row.

    `DepHour` decodes `CRSDepTime` (HHMM as an integer, so not linear in time)
    and maps 2400 to 0. `IsWeekend` / `Season` are partly redundant for a tree,
    but the linear baseline in S7 cannot find those thresholds on its own.
    """
    df = flights[SAFE_RAW].copy()
    df["DepHour"] = (flights["CRSDepTime"] // 100).replace(24, 0)
    df["IsWeekend"] = flights["DayOfWeek"].isin([6, 7]).astype(int)
    df["Season"] = flights["Month"].map(SEASON_BY_MONTH)
    
    return df




M = 100  
PRIOR = None 

def add_history_feature(frame : pd.DataFrame, keys: str | list[str], m: int = M, prior: float | None = PRIOR) -> np.ndarray:
    """Leakage-safe historical delay rate for `keys`, using only days strictly before each flight.

    Smoothed toward `prior` with strength `m`. Returns a float32 array aligned to `frame`'s rows.
    `keys`: a column name or a list (e.g. "Origin", or ["Origin", "Dest"] for a route).
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
    """Scheduled departures from this airport, in this hour, on this day.

    Built from the schedule alone, so it carries no label information and may use
    the current day. This is the hub-congestion effect found in EDA2, handed to
    the model directly instead of hoping it rediscovers it from raw columns.
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
    """Raw BTS frame to model inputs.

    Returns `(X, y, flight_date)`. `flight_date` is kept aside deliberately: it
    drives the time-based split in S8 but is not a feature.

    Missing values are left as NaN on purpose - XGBoost learns a default branch,
    which is more honest than imputing a mean.
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
