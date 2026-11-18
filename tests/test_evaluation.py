"""Test evaluation harness logic."""

import numpy as np
import pandas as pd
from models.evaluation import (
    compute_binary_metrics,
    compute_observation_top1,
    evaluate_group_cross_validation,
)
from sklearn.linear_model import LogisticRegression


def test_compute_binary_metrics():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["log_loss"] < 0.3


def test_compute_observation_top1():
    df_meta = pd.DataFrame({
        "trip_id": ["001", "001", "001", "001", "002", "002"],
        "obs_idx": [0, 0, 1, 1, 0, 0],
        "label": [1, 0, 0, 1, 1, 0],
    })
    # observation 0 of trip 001: candidate 0 gets score 0.8 (correct, label=1)
    # observation 1 of trip 001: candidate 1 gets score 0.7 (correct, label=1)
    # observation 0 of trip 002: candidate 1 gets score 0.9 (incorrect, label=0)
    scores = np.array([0.8, 0.2, 0.1, 0.7, 0.3, 0.9])
    
    top1, top_df = compute_observation_top1(df_meta, scores)
    # 2 out of 3 observations matched correctly -> 2/3
    assert np.isclose(top1, 2.0 / 3.0)
    assert len(top_df) == 3


def test_group_cv():
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 4)
    y = np.random.randint(0, 2, size=n)
    groups = np.repeat(np.arange(10), 20)
    df_meta = pd.DataFrame({
        "trip_id": groups,
        "obs_idx": np.tile(np.arange(20), 10),
        "label": y,
    })

    result = evaluate_group_cross_validation(
        lambda: LogisticRegression(),
        X, y, groups, df_meta, n_splits=3
    )
    assert "macro_f1_mean" in result
    assert "top1_mean" in result
    assert "log_loss_mean" in result
