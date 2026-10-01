"""Logistic regression baseline that XGBoost has to beat.

The three column lists below split all columns of `X` between the linear model
and "dropped", so a new feature has to be assigned somewhere on purpose.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LINEAR_NUMERIC: list[str] = [
    "IsWeekend",
    "CRSElapsedTime",
    "Distance",
    "CarrierHist",
    "OriginHist",
    "RouteHist",
    "OriginHourHist",
    "OriginHourCongestion",
]

LINEAR_ONEHOT: list[str] = [
    "Month",
    "DayOfWeek",
    "DepHour",
    "Reporting_Airline",
]

LINEAR_DROPPED: list[str] = [
    "Origin",
    "Dest",
    "OriginState",
    "DestState",
    "CRSDepTime",
    "CRSArrTime",
    "DistanceGroup",
    "Season",
]


def check_columns_covered(X: pd.DataFrame) -> None:
    """Fail if a column of `X` is missing from the lists or vice versa.

    Otherwise a new feature would only reach XGBoost and make it look better
    than it is.
    """
    columns = set(X.columns)
    linear_columns = set(LINEAR_NUMERIC + LINEAR_ONEHOT + LINEAR_DROPPED)
    assert columns == linear_columns, (
        f"in X but unassigned: {sorted(columns - linear_columns)}; "
        f"assigned but missing from X: {sorted(linear_columns - columns)}"
    )


def make_baseline_pipeline(C: float = 1.0, class_weight: str | None = None) -> Pipeline:
    """Scale numeric, one-hot categorical, then logistic regression."""
    num = StandardScaler()
    cat = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num, LINEAR_NUMERIC),
            ("cat", cat, LINEAR_ONEHOT),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)),
        ]
    )