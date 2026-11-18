"""Train and evaluate 5 models across distinct algorithm families.

Algorithms:
1. Model 1 (Linear/Probabilistic): Logistic Regression with L2 penalty and balanced weights
2. Model 2 (Instance-Based): Distance-Weighted K-Nearest Neighbors (k-NN)
3. Model 3 (Ensemble/Tree-Based): Balanced Random Forest Classifier
4. Model 4 (Gradient Boosting): LightGBM Classifier
5. Model 5 (Neural Network): Deep MLP with PyTorch accelerated on NVIDIA GeForce RTX 4060 GPU
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.paths import PROCESSED_DIR, REPO_ROOT, RESULTS_DIR
from features.feature_engineering import CANDIDATE_FEATURE_COLUMNS
from models.evaluation import (
    EvaluationResult,
    compute_binary_metrics,
    compute_observation_top1,
    evaluate_group_cross_validation,
)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class PyTorchMLP(nn.Module):
    """Deep Multi-Layer Perceptron optimized for tabular candidate scoring on CUDA."""

    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PyTorchMLPWrapper:
    """Scikit-learn compatible wrapper for PyTorch MLP running on RTX 4060 GPU."""

    def __init__(
        self,
        input_dim: int = len(CANDIDATE_FEATURE_COLUMNS),
        epochs: int = 25,
        batch_size: int = 2048,
        lr: float = 1e-3,
        pos_weight: float = 35.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.pos_weight = pos_weight
        self.device = torch.device(device)
        self.model = PyTorchMLP(input_dim).to(self.device)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=self.device))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        self.loss_history = []
        for epoch in range(self.epochs):
            total_loss = 0.0
            count = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * len(batch_y)
                count += len(batch_y)

            epoch_loss = total_loss / count
            self.loss_history.append(epoch_loss)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            # Batch inference to avoid OOM on huge candidate sets
            probs_list = []
            for i in range(0, len(tensor_x), 8192):
                batch = tensor_x[i : i + 8192]
                logits = self.model(batch)
                batch_probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(batch_probs)

            probs_pos = np.concatenate(probs_list) if probs_list else np.zeros(0)
            probs_neg = 1.0 - probs_pos
            return np.column_stack([probs_neg, probs_pos])


def load_dataset(use_sample_only: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and student-test candidate tables."""
    candidate_feature_dir = PROCESSED_DIR / "candidate_features"
    sample_csv = RESULTS_DIR / "tables" / "candidate_feature_sample.csv"

    if use_sample_only or not candidate_feature_dir.is_dir():
        print(f"Loading representative sample from {sample_csv}...")
        df = pd.read_csv(sample_csv)
        train_df = df[df["partition"] == "train"].reset_index(drop=True)
        test_df = df[df["partition"] == "student_test"].reset_index(drop=True)
        return train_df, test_df

    print(f"Loading full partitioned candidate feature tables from {candidate_feature_dir}...")
    train_files = sorted((candidate_feature_dir / "train").glob("*.csv.gz"))
    test_files = sorted((candidate_feature_dir / "student_test").glob("*.csv.gz"))

    if not train_files or not test_files:
        print("Candidate feature files not found; falling back to candidate_feature_sample.csv.")
        df = pd.read_csv(sample_csv)
        return (
            df[df["partition"] == "train"].reset_index(drop=True),
            df[df["partition"] == "student_test"].reset_index(drop=True),
        )

    train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)
    print(f"Loaded train: {len(train_df):,} rows; student_test: {len(test_df):,} rows.")
    return train_df, test_df


def build_models() -> Dict[str, Tuple[str, any]]:
    """Define the 5 required model factories across distinct algorithm families."""
    models = {
        "Model 1 (Logistic Regression)": (
            "Linear and Probabilistic",
            lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)),
            ]),
        ),
        "Model 2 (K-Nearest Neighbors)": (
            "Instance-based",
            lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
            ]),
        ),
        "Model 3 (Random Forest)": (
            "Tree-based Ensemble",
            lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1)),
            ]),
        ),
        "Model 4 (LightGBM)": (
            "Boosting",
            lambda: lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                scale_pos_weight=35.0,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        "Model 5 (PyTorch MLP on RTX 4060)": (
            "Neural Networks",
            lambda: PyTorchMLPWrapper(
                input_dim=len(CANDIDATE_FEATURE_COLUMNS),
                epochs=25,
                batch_size=2048,
                lr=1e-3,
                pos_weight=35.0,
            ),
        ),
    }
    return models


def evaluate_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, any], PyTorchMLPWrapper]:
    """Train and evaluate all 5 models under group CV and student-test."""
    feature_cols = list(CANDIDATE_FEATURE_COLUMNS)
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    groups_train = train_df["trip_id"].to_numpy()

    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

    models_dict = build_models()
    results: List[Dict[str, any]] = []
    trained_instances: Dict[str, any] = {}
    mlp_instance = None

    for model_name, (family, factory) in models_dict.items():
        print(f"\n==================================================")
        print(f"Evaluating {model_name} [{family}]...")
        print(f"==================================================")

        # 1. Group-aware Cross Validation on 80 training trips
        cv_res = evaluate_group_cross_validation(
            factory, X_train, y_train, groups_train, train_df, n_splits=n_splits
        )
        print(
            f"  CV Macro F1: {cv_res['macro_f1_mean']:.4f} +/- {cv_res['macro_f1_std']:.4f} | "
            f"Top-1 Acc: {cv_res['top1_mean']:.4f} | Log Loss: {cv_res['log_loss_mean']:.4f} | "
            f"CV Time: {cv_res['cv_runtime_s']:.2f} s"
        )

        # 2. Refit on full training set and evaluate on student_test
        start_fit = time.perf_counter()
        final_model = factory()
        final_model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start_fit
        trained_instances[model_name] = final_model

        if "PyTorch MLP" in model_name:
            mlp_instance = final_model

        # Inference timing and predictions on student_test
        start_inf = time.perf_counter()
        if hasattr(final_model, "predict_proba"):
            test_probs = final_model.predict_proba(X_test)[:, 1]
        else:
            scores = final_model.decision_function(X_test)
            test_probs = 1.0 / (1.0 + np.exp(-scores))
        inf_elapsed = time.perf_counter() - start_inf
        inf_latency_per_1k = (inf_elapsed / max(1, len(X_test))) * 1000.0 * 1000.0

        test_metrics = compute_binary_metrics(y_test, test_probs)
        test_top1, _ = compute_observation_top1(test_df, test_probs)

        print(
            f"  Student-Test Macro F1: {test_metrics['macro_f1']:.4f} | "
            f"Top-1 Acc: {test_top1:.4f} | Log Loss: {test_metrics['log_loss']:.4f} | "
            f"Latency: {inf_latency_per_1k:.2f} ms/1k"
        )

        eval_obj = EvaluationResult(
            model_name=model_name,
            algorithm_family=family,
            cv_macro_f1_mean=cv_res["macro_f1_mean"],
            cv_macro_f1_std=cv_res["macro_f1_std"],
            cv_top1_accuracy_mean=cv_res["top1_mean"],
            cv_top1_accuracy_std=cv_res["top1_std"],
            cv_log_loss_mean=cv_res["log_loss_mean"],
            cv_log_loss_std=cv_res["log_loss_std"],
            test_macro_f1=test_metrics["macro_f1"],
            test_top1_accuracy=test_top1,
            test_log_loss=test_metrics["log_loss"],
            test_accuracy=test_metrics["accuracy"],
            test_precision=test_metrics["precision"],
            test_recall=test_metrics["recall"],
            test_roc_auc=test_metrics["roc_auc"],
            training_time_s=fit_time,
            inference_time_ms_per_1k=inf_latency_per_1k,
        )
        results.append(eval_obj.to_dict())

    summary_df = pd.DataFrame(results)
    return summary_df, trained_instances, mlp_instance


def plot_model_comparisons(summary_df: pd.DataFrame, output_dir: Path):
    """Generate the mandatory comparison plots required by the university report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models = summary_df["Model Name"].str.replace(r" \(.*\)", "", regex=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. CV Macro F1
    axes[0, 0].bar(models, summary_df["CV Macro F1 (Mean)"], yerr=summary_df["CV Macro F1 (Std)"], color="#2b5c8f", capsize=5)
    axes[0, 0].set_title("Cross-Validation Macro F1-Score (5-Fold GroupKFold)", fontsize=12)
    axes[0, 0].set_ylabel("Macro F1")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

    # 2. Student-Test Top-1 Accuracy
    axes[0, 1].bar(models, summary_df["Student-Test Top-1 Acc"], color="#2a9d8f")
    axes[0, 1].set_title("Student-Test Observation-Level Top-1 Accuracy", fontsize=12)
    axes[0, 1].set_ylabel("Top-1 Accuracy")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

    # 3. Student-Test Macro F1
    axes[1, 0].bar(models, summary_df["Student-Test Macro F1"], color="#e76f51")
    axes[1, 0].set_title("Student-Test Macro F1-Score", fontsize=12)
    axes[1, 0].set_ylabel("Macro F1")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

    # 4. Log Loss
    axes[1, 1].bar(models, summary_df["Student-Test Log Loss"], color="#e63946")
    axes[1, 1].set_title("Student-Test Log Loss (Lower is Better)", fontsize=12)
    axes[1, 1].set_ylabel("Binary Cross-Entropy Loss")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = output_dir / "model_comparison_plots.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plots to {plot_path}")


def plot_mlp_epoch_loss(mlp_wrapper: PyTorchMLPWrapper, output_dir: Path):
    """Plot genuine epoch-wise training loss curve for the PyTorch MLP on RTX 4060."""
    if not mlp_wrapper or not mlp_wrapper.loss_history:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(mlp_wrapper.loss_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mlp_wrapper.loss_history, marker="o", color="#457b9d", linewidth=2, label="Train Loss (Weighted BCE)")
    plt.title("Model 5: PyTorch Deep MLP Epoch-Wise Loss Curve (RTX 4060 GPU)", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    loss_plot_path = output_dir / "mlp_gpu_epoch_loss.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Saved genuine epoch loss history to {loss_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate 5 models for PathSnap.")
    parser.add_argument("--sample", action="store_true", help="Force using candidate_feature_sample.csv")
    args = parser.parse_args()

    train_df, test_df = load_dataset(use_sample_only=args.sample)

    table_dir = RESULTS_DIR / "tables"
    plots_dir = RESULTS_DIR / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_df, trained_models, mlp_instance = evaluate_all_models(train_df, test_df)

    summary_csv = table_dir / "model_comparison_table.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved consolidated 5-model comparison table to {summary_csv}")
    print("\nConsolidated Comparison Table:")
    print(summary_df.to_string(index=False))

    plot_model_comparisons(summary_df, plots_dir)
    if mlp_instance:
        plot_mlp_epoch_loss(mlp_instance, plots_dir)


if __name__ == "__main__":
    main()
