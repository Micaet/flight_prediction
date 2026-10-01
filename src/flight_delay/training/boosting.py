
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


VALID_CUTOFF: str = "2024-10-01"

CATEGORICAL: list[str] = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "OriginState",
    "DestState",
    "Season",
]


DEFAULT_PARAMS: dict[str, object] = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 10, 
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist", 
    "enable_categorical": True,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
}


def prepare_for_xgboost(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `X` with every column in `CATEGORICAL` cast to `category`.
    """
    X = X.copy()
    for col in CATEGORICAL:
        assert col in X.columns, f"Missing column {col} from {list(X.columns)}"
        X[col] = X[col].astype("category")
    # Catch it here - XGBoost's own error doesn't say which column is wrong.
    leftover = [
        col
        for col in X.columns
        if not (
            pd.api.types.is_numeric_dtype(X[col])
            or isinstance(X[col].dtype, pd.CategoricalDtype)
        )
    ]
    assert not leftover, f"neither numeric nor categorical: {leftover}"
    return X


def carve_validation(
    flight_date: pd.Series,
    train: np.ndarray,
    valid_cutoff: str | pd.Timestamp = VALID_CUTOFF,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the training mask into `(fit, valid)` by date at `valid_cutoff`."""
    valid_cutoff = pd.Timestamp(valid_cutoff)
    fit = train & (flight_date < valid_cutoff)
    valid = train & (flight_date >= valid_cutoff)
    assert np.all((fit | valid) == train), "fit | valid != train"
    assert fit.any() and valid.any(), (
        f"empty side: fit={int(fit.sum())}, valid={int(valid.sum())}, cutoff={valid_cutoff.date()}"
    )
    return fit, valid


def make_model(
    params: dict[str, object] | None = None,
    scale_pos_weight: float | None = None,
) -> XGBClassifier:
    """`XGBClassifier` built from `DEFAULT_PARAMS`, overridden by `params`.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params)


def fit_with_early_stopping(
    model: XGBClassifier,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> XGBClassifier:
    """Fit with `eval_set=[(X_valid, y_valid)]` and return the fitted model.
    """
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, y_valid)],
        verbose=100,
    )
    return model


def feature_gain(model: XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Gain importance per feature, sorted descending (unused features get 0)."""
    gain = model.get_booster().get_score(importance_type="gain")
    gain = pd.DataFrame.from_dict(gain, orient="index", columns=["gain"])
    gain.index.name = "feature"
    gain = gain.reindex(feature_names).fillna(0.0)
    gain = gain.sort_values("gain", ascending=False)
    return gain
