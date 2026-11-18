"""Role B: Model training and evaluation for Models 1, 2, and 3 (and 5-model baseline).

Algorithm Families:
- Model 1: Linear and Probabilistic (Logistic Regression with balanced weights)
- Model 1-GPU: Linear and Probabilistic benchmark on NVIDIA GeForce RTX 4060
- Model 2: Instance-based (Distance-Weighted K-Nearest Neighbors)
- Model 3: Tree-based Ensemble (Balanced Random Forest Classifier)
- Model 4: Gradient Boosting (Histogram-based Gradient Boosting, Person C family)
- Model 5: Neural Network / MLP on NVIDIA GeForce RTX 4060 (Person C family)

Produces:
- 5-fold group-aware cross-validation on trip_id
- Student-testing evaluation on the 10 reserved development trips
- Comprehensive metrics: Accuracy, Precision, Recall, Macro F1, Log Loss, Top-1 Accuracy, Latency
- Consolidated comparison tables and plots for Role B report sections 3.1, 3.2, and 3.3.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from common.paths import PROCESSED_DIR, REPO_ROOT, RESULTS_DIR
from features.feature_engineering import CANDIDATE_FEATURE_COLUMNS
from models.evaluation import (
    EvaluationResult,
    compute_binary_metrics,
    compute_observation_top1,
    evaluate_group_cross_validation,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GPULogisticRegression(nn.Module):
    """Logistic Regression implemented in PyTorch running directly on NVIDIA RTX 4060 GPU."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class GPULogisticRegressionWrapper:
    """Scikit-learn compatible wrapper for PyTorch Logistic Regression on RTX 4060 GPU."""

    def __init__(self, epochs: int = 15, lr: float = 0.05, pos_weight: float = 35.0, batch_size: int = 16384):
        self.epochs = epochs
        self.lr = lr
        self.pos_weight = pos_weight
        self.batch_size = batch_size
        self.device = DEVICE
        self.model = GPULogisticRegression(len(CANDIDATE_FEATURE_COLUMNS)).to(self.device)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_imp = self.imputer.fit_transform(X)
        X_scl = self.scaler.fit_transform(X_imp)

        # Move full tensors directly to RTX 4060 GPU VRAM for instant slicing
        tx = torch.tensor(X_scl, dtype=torch.float32, device=self.device)
        ty = torch.tensor(y, dtype=torch.float32, device=self.device)
        n_samples = len(tx)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=self.device))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for i in range(0, n_samples, self.batch_size):
                idx = perm[i : i + self.batch_size]
                bx, by = tx[idx], ty[idx]
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.imputer.transform(X)
        X_scl = self.scaler.transform(X_imp)
        self.model.eval()
        with torch.no_grad():
            tx = torch.tensor(X_scl, dtype=torch.float32, device=self.device)
            logits_list = []
            for i in range(0, len(tx), self.batch_size * 4):
                batch = tx[i : i + self.batch_size * 4]
                logits_list.append(self.model(batch))
            logits = torch.cat(logits_list)
            probs_pos = torch.sigmoid(logits).cpu().numpy()
            probs_neg = 1.0 - probs_pos
            return np.column_stack([probs_neg, probs_pos])


class GPUMLP(nn.Module):
    """3-layer Multi-Layer Perceptron (MLP) running on NVIDIA GeForce RTX 4060 GPU."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GPUMLPWrapper:
    """Scikit-learn compatible wrapper for GPU MLP on RTX 4060."""

    def __init__(self, epochs: int = 15, lr: float = 0.005, pos_weight: float = 35.0, batch_size: int = 16384):
        self.epochs = epochs
        self.lr = lr
        self.pos_weight = pos_weight
        self.batch_size = batch_size
        self.device = DEVICE
        self.model = GPUMLP(len(CANDIDATE_FEATURE_COLUMNS), hidden_dim=64).to(self.device)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_imp = self.imputer.fit_transform(X)
        X_scl = self.scaler.fit_transform(X_imp)

        tx = torch.tensor(X_scl, dtype=torch.float32, device=self.device)
        ty = torch.tensor(y, dtype=torch.float32, device=self.device)
        n_samples = len(tx)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=self.device))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for i in range(0, n_samples, self.batch_size):
                idx = perm[i : i + self.batch_size]
                bx, by = tx[idx], ty[idx]
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_imp = self.imputer.transform(X)
        X_scl = self.scaler.transform(X_imp)
        self.model.eval()
        with torch.no_grad():
            tx = torch.tensor(X_scl, dtype=torch.float32, device=self.device)
            logits_list = []
            for i in range(0, len(tx), self.batch_size * 4):
                batch = tx[i : i + self.batch_size * 4]
                logits_list.append(self.model(batch))
            logits = torch.cat(logits_list)
            probs_pos = torch.sigmoid(logits).cpu().numpy()
            probs_neg = 1.0 - probs_pos
            return np.column_stack([probs_neg, probs_pos])


def load_candidate_data(subsample_train_trips: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and student-test candidate tables.
    
    Subsamples training trips if requested to ensure fast, memory-safe execution across all algorithms.
    """
    candidate_feature_dir = PROCESSED_DIR / "candidate_features"
    train_dir = candidate_feature_dir / "train"
    test_dir = candidate_feature_dir / "student_test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        print("Candidate feature directory not found; using candidate_feature_sample.csv.", flush=True)
        sample_csv = RESULTS_DIR / "tables" / "candidate_feature_sample.csv"
        df = pd.read_csv(sample_csv)
        return (
            df[df["partition"] == "train"].reset_index(drop=True),
            df[df["partition"] == "student_test"].reset_index(drop=True),
        )

    train_files = sorted(train_dir.glob("*.csv.gz"))
    test_files = sorted(test_dir.glob("*.csv.gz"))

    # Load test files (all 10 trips)
    print(f"Loading {len(test_files)} student_test trip candidate tables...", flush=True)
    test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

    # Load train files (deterministic selection of trips)
    selected_train = train_files[:subsample_train_trips] if subsample_train_trips > 0 else train_files
    print(f"Loading {len(selected_train)}/{len(train_files)} training trip candidate tables...", flush=True)
    train_df = pd.concat([pd.read_csv(f) for f in selected_train], ignore_index=True)

    print(
        f"Loaded train: {len(train_df):,} rows ({train_df['trip_id'].nunique()} trips); "
        f"student_test: {len(test_df):,} rows ({test_df['trip_id'].nunique()} trips).",
        flush=True,
    )
    return train_df, test_df


def train_and_evaluate_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, any]]:
    """Train Models 1-3 (Role B) and Models 4-5 (Role C baselines) under GroupKFold CV."""
    feature_cols = list(CANDIDATE_FEATURE_COLUMNS)
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    groups_train = train_df["trip_id"].to_numpy()

    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

    print(f"\nHardware check: Using device '{DEVICE}' for neural/tensor models.", flush=True)

    models_config = {
        "Model 1 (Logistic Regression)": (
            "Linear & Probabilistic",
            lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=300, random_state=RANDOM_SEED)),
            ]),
            False,
        ),
        "Model 1-GPU (Logistic Regression on RTX 4060)": (
            "Linear & Probabilistic (GPU)",
            lambda: GPULogisticRegressionWrapper(epochs=12, lr=0.05, pos_weight=35.0),
            False,
        ),
        "Model 2 (K-Nearest Neighbors)": (
            "Instance-based",
            lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
            ]),
            True,  # requires subsampling for fast, tractable distance evaluation
        ),
        "Model 3 (Random Forest)": (
            "Tree-based Ensemble",
            lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )),
            ]),
            False,
        ),
        "Model 4 (HistGradientBoosting)": (
            "Gradient Boosting",
            lambda: Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", HistGradientBoostingClassifier(
                    max_iter=100,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                )),
            ]),
            False,
        ),
        "Model 5 (PyTorch MLP on RTX 4060)": (
            "Neural Network (GPU)",
            lambda: GPUMLPWrapper(epochs=12, lr=0.005, pos_weight=35.0),
            False,
        ),
    }

    results: List[Dict[str, any]] = []
    fitted_models: Dict[str, any] = {}

    for name, (family, factory, subsample_required) in models_config.items():
        print(f"\n========================================================", flush=True)
        print(f"Training & Evaluating {name} [{family}]", flush=True)
        print(f"========================================================", flush=True)

        if subsample_required and len(X_train) > 25000:
            print("  [Note: Subsampling 25,000 rows for k-NN to prevent massive O(N^2) latency]", flush=True)
            idx_sub = np.random.choice(len(X_train), size=25000, replace=False)
            X_tr_eval, y_tr_eval, grp_eval, df_eval = (
                X_train[idx_sub],
                y_train[idx_sub],
                groups_train[idx_sub],
                train_df.iloc[idx_sub],
            )
            # For test evaluation of k-NN, evaluate on 25,000 test rows to keep inference time ~5 seconds
            test_sub_idx = np.random.choice(len(X_test), size=min(25000, len(X_test)), replace=False)
            X_te_eval, y_te_eval, df_te_eval = (
                X_test[test_sub_idx],
                y_test[test_sub_idx],
                test_df.iloc[test_sub_idx],
            )
        else:
            X_tr_eval, y_tr_eval, grp_eval, df_eval = X_train, y_train, groups_train, train_df
            X_te_eval, y_te_eval, df_te_eval = X_test, y_test, test_df

        # 1. Group-aware Cross Validation
        print("  Running 5-Fold GroupKFold Cross-Validation on trip_id...", flush=True)
        cv_res = evaluate_group_cross_validation(
            factory, X_tr_eval, y_tr_eval, grp_eval, df_eval, n_splits=n_splits
        )
        print(
            f"  CV Macro F1: {cv_res['macro_f1_mean']:.4f} +/- {cv_res['macro_f1_std']:.4f} | "
            f"Top-1 Acc: {cv_res['top1_mean']:.4f} +/- {cv_res['top1_std']:.4f} | "
            f"Log Loss: {cv_res['log_loss_mean']:.4f}",
            flush=True,
        )

        # 2. Refit on train and evaluate on student_test
        print("  Refitting model on full training partition...", flush=True)
        start_fit = time.perf_counter()
        final_model = factory()
        final_model.fit(X_tr_eval, y_tr_eval)
        fit_time = time.perf_counter() - start_fit
        fitted_models[name] = final_model

        # 3. Student-test inference
        print(f"  Evaluating student_test inference ({len(X_te_eval):,} candidate rows)...", flush=True)
        start_inf = time.perf_counter()
        if hasattr(final_model, "predict_proba"):
            probs = final_model.predict_proba(X_te_eval)[:, 1]
        else:
            scores = final_model.decision_function(X_te_eval)
            probs = 1.0 / (1.0 + np.exp(-scores))
        inf_elapsed = time.perf_counter() - start_inf
        latency_per_1k = (inf_elapsed / max(1, len(X_te_eval))) * 1000.0 * 1000.0

        test_metrics = compute_binary_metrics(y_te_eval, probs)
        test_top1, _ = compute_observation_top1(df_te_eval, probs)

        print(
            f"  Student-Test Macro F1: {test_metrics['macro_f1']:.4f} | "
            f"Top-1 Acc: {test_top1:.4f} | Log Loss: {test_metrics['log_loss']:.4f} | "
            f"Accuracy: {test_metrics['accuracy']:.4f} | Recall: {test_metrics['recall']:.4f} | "
            f"Fit Time: {fit_time:.2f}s | Latency: {latency_per_1k:.2f} ms/1k",
            flush=True,
        )

        res_obj = EvaluationResult(
            model_name=name,
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
            inference_time_ms_per_1k=latency_per_1k,
        )
        results.append(res_obj.to_dict())

    full_df = pd.DataFrame(results)
    # Role B models only (Models 1, 1-GPU, 2, 3)
    role_b_df = full_df[full_df["Model Name"].str.contains("Model 1|Model 2|Model 3")].reset_index(drop=True)
    return role_b_df, full_df, fitted_models


def plot_model_comparisons(df: pd.DataFrame, output_path: Path, title_prefix: str = "Role B Models"):
    """Plot performance comparison bar charts."""
    models = df["Model Name"].str.replace(r" \(.*\)", "", regex=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. CV Macro F1
    axes[0, 0].bar(models, df["CV Macro F1 (Mean)"], yerr=df["CV Macro F1 (Std)"], color="#1d3557", capsize=5)
    axes[0, 0].set_title(f"{title_prefix}: 5-Fold Group CV Macro F1 (trip_id)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Macro F1-Score")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

    # 2. Student-Test Top-1 Accuracy
    axes[0, 1].bar(models, df["Student-Test Top-1 Acc"], color="#2a9d8f")
    axes[0, 1].set_title(f"{title_prefix}: Student-Test Top-1 Observation Accuracy", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Top-1 Accuracy")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

    # 3. Student-Test Macro F1
    axes[1, 0].bar(models, df["Student-Test Macro F1"], color="#e76f51")
    axes[1, 0].set_title(f"{title_prefix}: Student-Test Macro F1-Score", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Macro F1")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.7)

    # 4. Student-Test Log Loss
    axes[1, 1].bar(models, df["Student-Test Log Loss"], color="#e63946")
    axes[1, 1].set_title(f"{title_prefix}: Student-Test Log Loss (Lower is Better)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Log Loss (BCE)")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Execute Role B Model Training & Evaluation.")
    parser.add_argument("--trips", type=int, default=15, help="Number of training trips to load (default: 15)")
    args = parser.parse_args()

    train_df, test_df = load_candidate_data(subsample_train_trips=args.trips)

    tables_dir = RESULTS_DIR / "tables"
    plots_dir = RESULTS_DIR / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    role_b_df, full_5models_df, fitted_models = train_and_evaluate_all_models(train_df, test_df)

    # Save Role B summary
    b_csv = tables_dir / "role_b_models_summary.csv"
    role_b_df.to_csv(b_csv, index=False)
    print(f"\nSaved Role B model summary table to {b_csv}", flush=True)

    # Save full 5-model summary for Section 3.1
    five_csv = tables_dir / "model_comparison_5models.csv"
    full_5models_df.to_csv(five_csv, index=False)
    print(f"Saved consolidated 5-model summary table to {five_csv}", flush=True)

    print("\n================== ROLE B MODEL SUMMARY ==================", flush=True)
    print(role_b_df.to_string(index=False), flush=True)

    print("\n================== CONSOLIDATED 5-MODEL SUMMARY ==================", flush=True)
    print(full_5models_df.to_string(index=False), flush=True)

    # Generate plots
    plot_model_comparisons(role_b_df, plots_dir / "role_b_model_comparison_plots.png", "Role B Models")
    plot_model_comparisons(full_5models_df, plots_dir / "model_comparison_5models_plots.png", "Consolidated 5 Models")


if __name__ == "__main__":
    main()
