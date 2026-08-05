from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_ORDER: list[str] = [
    "roc_auc",
    "pr_auc",
    "no_skill_pr",
    "precision",
    "recall",
    "f1",
    "accuracy",
]


def evaluate_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Metric bundle for one model on one set.
    """
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "no_skill_pr": float(np.mean(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def compare(results: dict[str, dict[str, float]], decimals: int = 4) -> pd.DataFrame:
    """Stack per-model metric bundles into one table, models as rows.
    """
    table = pd.DataFrame.from_dict(results, orient="index")
    known = [m for m in METRIC_ORDER if m in table.columns]
    extra = [c for c in table.columns if c not in METRIC_ORDER]
    return table[known + extra].round(decimals)
