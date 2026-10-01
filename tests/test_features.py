"""Tests for `flight_delay.transform.features` on tiny hand-made frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flight_delay.transform import features as F


@pytest.fixture
def history_frame() -> pd.DataFrame:
    """AA over three days plus one BB flight.

    Expected AA rates with m=2, prior=0.5:
        day 1 -> (0 + 2*0.5) / (0 + 2) = 0.5
        day 2 -> (0 + 1) / (2 + 2)     = 0.25
        day 3 -> (3 + 1) / (5 + 2)     = 4/7
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
    frame["row_id"] = range(len(frame))
    return frame


@pytest.fixture
def raw_flights() -> pd.DataFrame:
    """Small processed BTS frame with one cancelled and one diverted flight."""
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
    population = F.prepare_population(raw_flights)

    assert len(population) == 6
    assert not population["Cancelled"].any()
    assert not population["Diverted"].any()
    # Needed - with a gapped index the history features end up as NaN.
    assert population.index.equals(pd.RangeIndex(6))
    assert len(raw_flights) == 8


def test_label_matches_threshold(raw_flights: pd.DataFrame) -> None:
    frame = raw_flights.copy()
    frame["ArrDelay"] = np.array(
        [14.0, 14.9, 15.0, 15.1, 100.0, -20.0, 0.0, 15.0], dtype="float32"
    )

    labelled = F.make_label(frame)
    np.testing.assert_array_equal(
        labelled["y"].to_numpy(), np.array([0, 0, 1, 1, 1, 0, 0, 1])
    )

    relabelled = F.make_label(frame, threshold=100)
    np.testing.assert_array_equal(
        relabelled["y"].to_numpy(), np.array([0, 0, 0, 0, 1, 0, 0, 0])
    )


def test_history_cold_start_equals_prior(history_frame: pd.DataFrame) -> None:
    hist = F.add_history_feature(history_frame, "Reporting_Airline", m=2, prior=0.5)

    assert not np.isnan(hist).any()
    np.testing.assert_allclose(hist[[0, 1, 5]], 0.5, rtol=1e-6)


def test_history_uses_only_strictly_earlier_days(history_frame: pd.DataFrame) -> None:
    hist = F.add_history_feature(history_frame, "Reporting_Airline", m=2, prior=0.5)

    expected = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.5, 4 / 7], dtype="float32")
    np.testing.assert_allclose(hist, expected, rtol=1e-6)
    # 0.5 is what you'd get if day 3 counted itself.
    assert hist[6] != pytest.approx(0.5)


def test_history_keeps_keys_independent(history_frame: pd.DataFrame) -> None:
    full = F.add_history_feature(history_frame, "Reporting_Airline", m=2, prior=0.5)

    # "BB" stays as an unused category here, so this also checks observed=True.
    aa_only = history_frame[history_frame["Reporting_Airline"] == "AA"]
    without_bb = F.add_history_feature(aa_only, "Reporting_Airline", m=2, prior=0.5)
    np.testing.assert_allclose(full[[0, 1, 2, 3, 4, 6]], without_bb, rtol=1e-6)

    # Row order must survive the merge.
    shuffled = history_frame.sample(frac=1, random_state=0)
    hist_shuffled = F.add_history_feature(shuffled, "Reporting_Airline", m=2, prior=0.5)
    by_row_id = dict(zip(shuffled["row_id"], hist_shuffled))
    np.testing.assert_allclose(
        [by_row_id[row_id] for row_id in history_frame["row_id"]], full, rtol=1e-6
    )


def test_no_leakage_column_reaches_x(raw_flights: pd.DataFrame) -> None:
    X, y, flight_date = F.build_features(raw_flights, m=2, prior=0.5)

    assert not set(F.LEAKAGE) & set(X.columns)
    assert "FlightDate" not in X.columns
    assert flight_date.name == "FlightDate"
    assert len(flight_date) == len(y) == len(X) == 6

    # 12 raw + 3 free + 4 history + 1 congestion
    assert X.shape == (6, 20)
    np.testing.assert_array_equal(y, np.array([0, 1, 0, 1, 0, 1], dtype="int8"))

    histories = ["CarrierHist", "OriginHist", "RouteHist", "OriginHourHist"]
    assert X[histories].notna().all().all()


def test_dep_hour_decoding(raw_flights: pd.DataFrame) -> None:
    X = F.add_free_features(F.prepare_population(raw_flights))

    np.testing.assert_array_equal(
        X["DepHour"].to_numpy(), np.array([6, 13, 0, 6, 13, 0])
    )
    assert X["DepHour"].max() <= 23
