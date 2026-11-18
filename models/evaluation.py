"""Evaluation harness for candidate classification and observation-level route matching.

Provides group-aware cross-validation across trip_id, computing:
- Accuracy, Precision, Recall, Macro F1
- Log Loss (binary cross-entropy on calibrated probabilities)
- Observation-level Top-1 Accuracy (ranking candidates per GPS point)
- Confusion matrices
- Training and inference latency metrics
"""

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


@dataclass
class EvaluationResult:
    """Consolidated metrics container conforming to university guidelines."""

    model_name: str
    algorithm_family: str
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    cv_top1_accuracy_mean: float
    cv_top1_accuracy_std: float
    cv_log_loss_mean: float
    cv_log_loss_std: float
    test_macro_f1: Optional[float] = None
    test_top1_accuracy: Optional[float] = None
    test_log_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_roc_auc: Optional[float] = None
    confusion_mat: Optional[np.ndarray] = None
    training_time_s: float = 0.0
    inference_time_ms_per_1k: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a flattened dictionary for reporting tables."""
        return {
            "Model Name": self.model_name,
            "Algorithm Family": self.algorithm_family,
            "CV Macro F1 (Mean)": round(self.cv_macro_f1_mean, 4),
            "CV Macro F1 (Std)": round(self.cv_macro_f1_std, 4),
            "CV Top-1 Acc (Mean)": round(self.cv_top1_accuracy_mean, 4),
            "CV Top-1 Acc (Std)": round(self.cv_top1_accuracy_std, 4),
            "CV Log Loss (Mean)": round(self.cv_log_loss_mean, 4),
            "CV Log Loss (Std)": round(self.cv_log_loss_std, 4),
            "Student-Test Macro F1": round(self.test_macro_f1, 4) if self.test_macro_f1 is not None else np.nan,
            "Student-Test Top-1 Acc": round(self.test_top1_accuracy, 4) if self.test_top1_accuracy is not None else np.nan,
            "Student-Test Log Loss": round(self.test_log_loss, 4) if self.test_log_loss is not None else np.nan,
            "Student-Test Accuracy": round(self.test_accuracy, 4) if self.test_accuracy is not None else np.nan,
            "Student-Test Precision": round(self.test_precision, 4) if self.test_precision is not None else np.nan,
            "Student-Test Recall": round(self.test_recall, 4) if self.test_recall is not None else np.nan,
            "Training Time (s)": round(self.training_time_s, 3),
            "Inference Latency (ms/1k)": round(self.inference_time_ms_per_1k, 3),
        }


def compute_observation_top1(
    df_meta: pd.DataFrame,
    scores: np.ndarray,
    label_col: str = "label",
    trip_col: str = "trip_id",
    obs_col: str = "obs_idx",
) -> Tuple[float, pd.DataFrame]:
    """Compute Observation-level Top-1 Accuracy.
    
    For each GPS observation (trip_id, obs_idx), choose the candidate with
    the highest predicted score and check if label == 1.
    """
    sub_df = pd.DataFrame({
        trip_col: df_meta[trip_col].values,
        obs_col: df_meta[obs_col].values,
        label_col: df_meta[label_col].values,
        "score": scores,
    })
    # Fast vectorized top-1: sort descending by score and keep first occurrence per (trip_id, obs_idx)
    top_candidates = sub_df.sort_values("score", ascending=False).drop_duplicates(subset=[trip_col, obs_col])
    top1_accuracy = float(top_candidates[label_col].mean())
    return top1_accuracy, top_candidates


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute standard classification metrics on candidate arc predictions."""
    y_pred = (y_prob >= threshold).astype(int)
    clipped_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    try:
        loss = float(log_loss(y_true, clipped_prob, labels=[0, 1]))
    except Exception:
        loss = np.nan

    try:
        auc = float(roc_auc_score(y_true, clipped_prob))
    except Exception:
        auc = np.nan

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "log_loss": loss,
        "roc_auc": auc,
    }


def evaluate_group_cross_validation(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    df_meta: pd.DataFrame,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Perform group-aware cross-validation across trips."""
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_macro_f1s = []
    fold_top1s = []
    fold_log_losses = []
    
    oof_probs = np.zeros(len(y), dtype=float)

    start_time = time.perf_counter()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        meta_val = df_meta.iloc[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
        else:
            scores = model.decision_function(X_val)
            probs = 1.0 / (1.0 + np.exp(-scores))

        oof_probs[val_idx] = probs
        
        metrics = compute_binary_metrics(y_val, probs)
        top1, _ = compute_observation_top1(meta_val, probs)

        fold_macro_f1s.append(metrics["macro_f1"])
        fold_top1s.append(top1)
        fold_log_losses.append(metrics["log_loss"])

    total_cv_time = time.perf_counter() - start_time

    return {
        "macro_f1_mean": float(np.mean(fold_macro_f1s)),
        "macro_f1_std": float(np.std(fold_macro_f1s)),
        "top1_mean": float(np.mean(fold_top1s)),
        "top1_std": float(np.std(fold_top1s)),
        "log_loss_mean": float(np.mean(fold_log_losses)),
        "log_loss_std": float(np.std(fold_log_losses)),
        "oof_probs": oof_probs,
        "cv_runtime_s": total_cv_time,
    }
