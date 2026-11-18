"""Bayesian Hyperparameter Optimisation using Optuna for PathSnap.

Optimises the chosen model (LightGBM / Tree-based Ensemble) under group-aware cross-validation on trip_id,
records trials into SQLite and CSV, refits with best parameters, evaluates on student-test, and exports
artifacts conforming to university guidelines.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)
import pandas as pd

from common.paths import PROCESSED_DIR, REPO_ROOT, RESULTS_DIR
from features.feature_engineering import CANDIDATE_FEATURE_COLUMNS
from models.evaluation import (
    compute_binary_metrics,
    compute_observation_top1,
    evaluate_group_cross_validation,
)

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_dataset(subsample_train_trips: int = 15, use_sample_only: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and student-test data."""
    candidate_feature_dir = PROCESSED_DIR / "candidate_features"
    sample_csv = RESULTS_DIR / "tables" / "candidate_feature_sample.csv"

    if use_sample_only or not candidate_feature_dir.is_dir():
        print(f"Loading representative sample from {sample_csv}...", flush=True)
        df = pd.read_csv(sample_csv)
        train_df = df[df["partition"] == "train"].reset_index(drop=True)
        test_df = df[df["partition"] == "student_test"].reset_index(drop=True)
        return train_df, test_df

    train_files = sorted((candidate_feature_dir / "train").glob("*.csv.gz"))
    test_files = sorted((candidate_feature_dir / "student_test").glob("*.csv.gz"))

    if not train_files or not test_files:
        df = pd.read_csv(sample_csv)
        return (
            df[df["partition"] == "train"].reset_index(drop=True),
            df[df["partition"] == "student_test"].reset_index(drop=True),
        )

    print(f"Loading {len(test_files)} student_test candidate tables...", flush=True)
    test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)

    selected_train = train_files[:subsample_train_trips] if subsample_train_trips > 0 else train_files
    print(f"Loading {len(selected_train)}/{len(train_files)} training trip candidate tables...", flush=True)
    train_df = pd.concat([pd.read_csv(f) for f in selected_train], ignore_index=True)

    print(
        f"Loaded train: {len(train_df):,} rows ({train_df['trip_id'].nunique()} trips); "
        f"student_test: {len(test_df):,} rows ({test_df['trip_id'].nunique()} trips).",
        flush=True,
    )
    return train_df, test_df


def run_bayesian_optimization(
    train_df: pd.DataFrame,
    n_trials: int = 25,
    n_splits: int = 5,
) -> Tuple[optuna.Study, float]:
    """Execute Optuna TPE search with group-aware cross-validation on trip_id."""
    feature_cols = list(CANDIDATE_FEATURE_COLUMNS)
    X = train_df[feature_cols].to_numpy(dtype=float)
    y = train_df["label"].to_numpy(dtype=int)
    groups = train_df["trip_id"].to_numpy()

    study_db = REPO_ROOT / "optimisation" / "optuna_study.db"
    study_db.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{study_db.as_posix()}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="pathsnap_bayesian_tuning",
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 15.0, 45.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbose": -1,
        }

        def factory():
            return lgb.LGBMClassifier(**params)

        cv_res = evaluate_group_cross_validation(
            factory, X, y, groups, train_df, n_splits=n_splits
        )
        return cv_res["macro_f1_mean"]

    def trial_callback(s: optuna.Study, t: optuna.trial.FrozenTrial):
        print(
            f"  Trial {t.number:2d}/{n_trials}: CV Macro F1 = {t.value:.4f} "
            f"(Current Best: {s.best_value:.4f} at Trial #{s.best_trial.number})",
            flush=True,
        )

    print(f"\nStarting Optuna Bayesian Hyperparameter Search ({n_trials} trials, TPE Sampler)...", flush=True)
    start_time = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])
    total_tuning_time = time.perf_counter() - start_time

    print(f"\nTuning completed in {total_tuning_time:.2f} s.", flush=True)
    print(f"Best Trial #{study.best_trial.number}:", flush=True)
    print(f"  Best CV Macro F1: {study.best_value:.4f}", flush=True)
    print(f"  Best Parameters: {json.dumps(study.best_params, indent=2)}", flush=True)

    return study, total_tuning_time


def compare_tuned_and_untuned(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: Dict[str, any],
    tuning_time_s: float,
    n_trials: int,
) -> pd.DataFrame:
    """Compare baseline untuned model with the Bayesian-tuned model."""
    feature_cols = list(CANDIDATE_FEATURE_COLUMNS)
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    groups_train = train_df["trip_id"].to_numpy()

    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

    print("\nEvaluating Untuned Baseline Model...", flush=True)
    default_factory = lambda: lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.1, num_leaves=31, scale_pos_weight=35.0, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    )
    cv_untuned = evaluate_group_cross_validation(default_factory, X_train, y_train, groups_train, train_df)
    model_untuned = default_factory().fit(X_train, y_train)

    start_inf = time.perf_counter()
    probs_untuned = model_untuned.predict_proba(X_test)[:, 1]
    latency_untuned = (time.perf_counter() - start_inf) / max(1, len(X_test)) * 1e6

    metrics_untuned = compute_binary_metrics(y_test, probs_untuned)
    top1_untuned, _ = compute_observation_top1(test_df, probs_untuned)

    print("Evaluating Tuned Model with Best Parameters...", flush=True)
    tuned_params = {**best_params, "random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1}
    tuned_factory = lambda: lgb.LGBMClassifier(**tuned_params)
    cv_tuned = evaluate_group_cross_validation(tuned_factory, X_train, y_train, groups_train, train_df)

    start_fit = time.perf_counter()
    model_tuned = tuned_factory().fit(X_train, y_train)
    fit_time_tuned = time.perf_counter() - start_fit

    start_inf = time.perf_counter()
    probs_tuned = model_tuned.predict_proba(X_test)[:, 1]
    latency_tuned = (time.perf_counter() - start_inf) / max(1, len(X_test)) * 1e6

    metrics_tuned = compute_binary_metrics(y_test, probs_tuned)
    top1_tuned, _ = compute_observation_top1(test_df, probs_tuned)

    comparison_data = [
        {
            "Configuration": "Untuned Baseline (Default LightGBM)",
            "CV Macro F1 (Mean)": round(cv_untuned["macro_f1_mean"], 4),
            "CV Top-1 Acc (Mean)": round(cv_untuned["top1_mean"], 4),
            "Student-Test Macro F1": round(metrics_untuned["macro_f1"], 4),
            "Student-Test Top-1 Acc": round(top1_untuned, 4),
            "Student-Test Log Loss": round(metrics_untuned["log_loss"], 4),
            "Inference Latency (ms/1k)": round(latency_untuned, 3),
            "Optimization Trials": 0,
            "Tuning Cost (s)": 0.0,
        },
        {
            "Configuration": "Tuned Model (Optuna Bayesian TPE)",
            "CV Macro F1 (Mean)": round(cv_tuned["macro_f1_mean"], 4),
            "CV Top-1 Acc (Mean)": round(cv_tuned["top1_mean"], 4),
            "Student-Test Macro F1": round(metrics_tuned["macro_f1"], 4),
            "Student-Test Top-1 Acc": round(top1_tuned, 4),
            "Student-Test Log Loss": round(metrics_tuned["log_loss"], 4),
            "Inference Latency (ms/1k)": round(latency_tuned, 3),
            "Optimization Trials": n_trials,
            "Tuning Cost (s)": round(tuning_time_s, 2),
        },
    ]

    df_comp = pd.DataFrame(comparison_data)

    # Export final tuned model and feature list to artifacts/
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "final_model.joblib"
    feature_list_path = ARTIFACTS_DIR / "feature_list.json"
    params_path = ARTIFACTS_DIR / "best_hyperparameters.json"

    joblib.dump(model_tuned, model_path)
    feature_list_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    print(f"\nExported artifacts:")
    print(f"  Model: {model_path}")
    print(f"  Feature List: {feature_list_path}")
    print(f"  Hyperparameters: {params_path}")

    return df_comp


def save_optimization_plots(study: optuna.Study, output_dir: Path):
    """Plot optimization history and parameter importances using Matplotlib."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Optimization history
    try:
        ax = plot_optimization_history(study)
        plt.title("Bayesian Optimisation Convergence History", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "optuna_convergence_history.png", dpi=300)
        plt.close()
        print(f"Saved convergence history to {output_dir / 'optuna_convergence_history.png'}", flush=True)
    except Exception as e:
        print(f"Could not generate history plot: {e}", flush=True)

    # 2. Parameter importances
    try:
        ax = plot_param_importances(study)
        plt.title("Optuna Hyperparameter Importance Analysis", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "optuna_param_importances.png", dpi=300)
        plt.close()
        print(f"Saved parameter importances to {output_dir / 'optuna_param_importances.png'}", flush=True)
    except Exception as e:
        print(f"Could not generate importance plot: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Optuna Bayesian Tuning for PathSnap.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Bayesian optimization trials")
    parser.add_argument("--trips", type=int, default=15, help="Number of training trips to load (default: 15)")
    parser.add_argument("--sample", action="store_true", help="Force using candidate_feature_sample.csv")
    args = parser.parse_args()

    train_df, test_df = load_dataset(subsample_train_trips=args.trips, use_sample_only=args.sample)

    study, tuning_time = run_bayesian_optimization(train_df, n_trials=args.trials)

    # Save trial records
    trials_df = study.trials_dataframe()
    tables_dir = RESULTS_DIR / "tables"
    plots_dir = RESULTS_DIR / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    trials_csv = tables_dir / "optuna_trial_records.csv"
    trials_df.to_csv(trials_csv, index=False)
    print(f"Saved all trial records to {trials_csv}", flush=True)

    # Save plots
    save_optimization_plots(study, plots_dir)

    # Compare tuned vs untuned
    comp_df = compare_tuned_and_untuned(
        train_df, test_df, study.best_params, tuning_time, len(study.trials)
    )
    comp_csv = tables_dir / "tuned_vs_untuned_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nSaved tuned vs untuned comparison to {comp_csv}", flush=True)
    print("\nTuned vs Untuned Performance:", flush=True)
    print(comp_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
