"""Time-based train/test split."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_CUTOFF: str = "2024-11-01"


def time_split(
    flight_date: pd.Series,
    cutoff: str | pd.Timestamp = DEFAULT_CUTOFF,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks `(train, test)`: days before `cutoff` train, the rest test.
    """
    return flight_date < cutoff, flight_date >= cutoff


def describe_split(
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    flight_date: pd.Series,
) -> pd.DataFrame:
    """Sanity table: row count, date range and positive rate for each side.
    """
    train_count = np.sum(train)
    test_count = np.sum(test)
    train_date_range = (flight_date[train].min(), flight_date[train].max())
    test_date_range = (flight_date[test].min(), flight_date[test].max())
    train_positive_rate = np.mean(y[train])
    test_positive_rate = np.mean(y[test])

    return pd.DataFrame({
        "side": ["train", "test"],
        "count": [train_count, test_count],
        "date_range": [train_date_range, test_date_range],
        "positive_rate": [train_positive_rate, test_positive_rate]
    })


def train_prior(y: np.ndarray, train: np.ndarray) -> float:
    return np.mean(y[train])
