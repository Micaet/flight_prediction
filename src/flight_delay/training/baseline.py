"""Baselines for the delay model (S7): the numbers XGBoost has to beat.

Three references, each answering a different question:

* `most_frequent` - predicts "on time" for every flight. Accuracy equals the share
  of on-time flights (~0.79), recall equals 0. This is the entire argument for
  dropping accuracy: a model that never finds a single delay looks 79% correct.
* `stratified` - guesses at the right base rate. Worth one row for a single reason:
  recall 0.27 at precision 0.15 shows that recall on its own means nothing. It does
  NOT pin the no-skill lines - `predict_proba` returns a random one-hot per row, so
  it lands near them with sampling noise (measured 0.4964 and 0.1503 against a floor
  of 0.1511, i.e. below its own floor). The exact references come from
  `most_frequent`: a constant score gives ROC-AUC 0.5 and PR-AUC equal to the
  positive rate on the nose, because a flat ranking has nothing to exploit.
* logistic regression - the honest baseline. If gradient boosting in S8 cannot beat
  a linear model on the same features, the extra complexity is not paying for itself.

The linear model cannot eat the feature matrix as it stands, and how it is cut down
is the interesting part - see the three column lists below. They partition all 20
columns of `X`, so a feature added later has to be placed deliberately instead of
quietly vanishing from the baseline.
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
    """Assert the three lists partition `X.columns` exactly - no overlap, nothing missing.

    Cheap insurance against the failure mode where S8 adds a feature, nobody updates
    the lists, and the baseline silently keeps competing without it - which would
    flatter XGBoost by exactly the value of the new column.
    """
    columns = set(X.columns)
    linear_columns = set(LINEAR_NUMERIC + LINEAR_ONEHOT + LINEAR_DROPPED)
    assert columns == linear_columns, (
        f"in X but unassigned: {sorted(columns - linear_columns)}; "
        f"assigned but missing from X: {sorted(linear_columns - columns)}"
    )


def make_baseline_pipeline(C: float = 1.0, class_weight: str | None = None) -> Pipeline:
    """`ColumnTransformer` (scale numeric, one-hot the rest) feeding a logistic regression.
    """
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