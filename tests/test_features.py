"""Tests for the feature engineering in `flight_delay.transform.features`.

Everything runs on tiny hand-built frames: a test that needs 7 million rows is a
test nobody runs. The values below are small enough to verify with a calculator,
which is the point - the leakage guard is only worth something if a wrong result
is obvious.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_delay.transform import features as F


@pytest.fixture
def history_frame() -> pd.DataFrame:
    """Three days for carrier AA, plus a second carrier to prove keys stay separate.

    AA: day 1 -> [0, 0]      (2 flights, 0 delayed)
        day 2 -> [1, 1, 1]   (3 flights, 3 delayed)
        day 3 -> [0]         (1 flight, ignored by its own history)
    BB: day 2 -> [1]         (must not touch AA's numbers)

    With m=2 and prior=0.5 the expected AA rates are:
        day 1 -> (0 + 2*0.5) / (0 + 2) = 0.5      (cold start = prior)
        day 2 -> (0 + 1) / (2 + 2)     = 0.25     (day 1 only)
        day 3 -> (3 + 1) / (5 + 2)     = 4/7      (days 1-2, never day 3)
    """
    rows = [
        ("2024-01-01", "AA", 0),
        ("2024-01-01", "AA", 0),
        ("2024-01-02", "AA", 1),
        ("2024-01-02", "AA", 1),
        ("2024-01-02", "AA", 1),
        ("2024-01-02", "BB", 1),
        ("2024-01-03", "AA", 0),
    ]
    frame = pd.DataFrame(rows, columns=["FlightDate", "Reporting_Airline", "y"])
    frame["FlightDate"] = pd.to_datetime(frame["FlightDate"])
    frame["Reporting_Airline"] = frame["Reporting_Airline"].astype("category")
    frame["row_id"] = range(len(frame))  # marker: the merge must not reorder rows
    return frame


@pytest.fixture
def raw_flights() -> pd.DataFrame:
    """A miniature version of the processed BTS frame, including rows that must be dropped."""
    n = 8
    frame = pd.DataFrame(
        {
            "FlightDate": pd.to_datetime(
                ["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-01-03"] * 2
            ),
            "Month": np.int8(1),
            "DayOfWeek": np.array([1, 1, 1, 2, 2, 2, 3, 3], dtype="int8"),
            "Reporting_Airline": pd.Categorical(["AA", "AA", "BB"] * 2 + ["AA", "BB"]),
            "Origin": pd.Categorical(["ATL", "ATL", "ORD"] * 2 + ["ATL", "ORD"]),
            "Dest": pd.Categorical(["MCO", "ORD", "ATL"] * 2 + ["MCO", "ATL"]),
            "OriginState": pd.Categorical(["GA", "GA", "IL"] * 2 + ["GA", "IL"]),
            "DestState": pd.Categorical(["FL", "IL", "GA"] * 2 + ["FL", "GA"]),
            "CRSDepTime": np.array([600, 1345, 2400, 600, 1345, 2400, 600, 900], dtype="int16"),
            "CRSArrTime": np.array([800, 1500, 130, 800, 1500, 130, 800, 1100], dtype="int16"),
            "CRSElapsedTime": np.float32(120.0),
            "Distance": np.float32(400.0),
            "DistanceGroup": np.int8(2),
            # post-flight columns: must never reach X
            "ArrDelay": np.array([-5.0, 30.0, 0.0, 20.0, -10.0, 45.0, 5.0, 60.0], dtype="float32"),
            "DepDelay": np.float32(10.0),
            "TaxiOut": np.float32(15.0),
            "Cancelled": np.array([False] * 6 + [True, False]),
            "Diverted": np.array([False] * 7 + [True]),
        }
    )
    assert len(frame) == n
    return frame


def test_population_drops_cancelled_and_diverted(raw_flights: pd.DataFrame) -> None:
    """Cancelled and diverted flights leave the population; the index is reset."""
    raise NotImplementedError  # TODO


def test_label_matches_threshold(raw_flights: pd.DataFrame) -> None:
    """y is 1 exactly when ArrDelay >= 15, boundary included."""
    raise NotImplementedError  # TODO


def test_history_cold_start_equals_prior(history_frame: pd.DataFrame) -> None:
    """A key seen for the first time gets the prior, not NaN."""
    raise NotImplementedError  # TODO


def test_history_uses_only_strictly_earlier_days(history_frame: pd.DataFrame) -> None:
    """Day 3 for AA equals 4/7: days 1-2 only, its own day excluded.

    This is the test that fails if `<` ever becomes `<=`.
    """
    raise NotImplementedError  # TODO


def test_history_keeps_keys_independent(history_frame: pd.DataFrame) -> None:
    """BB's flights must not move AA's history, and row order must survive the merge."""
    raise NotImplementedError  # TODO


def test_no_leakage_column_reaches_x(raw_flights: pd.DataFrame) -> None:
    """No column from LEAKAGE appears in X, and FlightDate is returned separately."""
    raise NotImplementedError  # TODO


def test_dep_hour_decoding(raw_flights: pd.DataFrame) -> None:
    """CRSDepTime 1345 -> 13 and 2400 -> 0 (midnight, not hour 24)."""
    raise NotImplementedError  # TODO
