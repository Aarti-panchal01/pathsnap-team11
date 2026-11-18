"""Run Role C's two models and their pointwise-versus-Viterbi comparison."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from common.loader import list_trips, load_trip
from decoder import ViterbiConfig, decode
from models.role_c import FEATURE_COLUMNS, model_factories, score_candidates


SEED = 11
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def read_candidates(data, partition, trip_id, *, labels=False):
    columns = ["obs_idx", "cand_arc_id", *FEATURE_COLUMNS]
    if labels:
        columns.append("label")
    dtypes = dict.fromkeys(FEATURE_COLUMNS, "float32")
    dtypes.update(obs_idx="int32", cand_arc_id="int64", label="int8")
    del dtypes["is_gap_point"]
    path = data / "processed" / "candidate_features" / partition / f"{trip_id}.csv.gz"
    frame = pd.read_csv(path, usecols=columns, dtype=dtypes)
    frame["is_gap_point"] = frame["is_gap_point"].astype("float32")
    return frame


def sample_training_trip(data, trip_id):
    rows = read_candidates(data, "train", trip_id, labels=True)
    positive = rows.loc[rows["label"].eq(1)]
    negative = rows.loc[rows["label"].eq(0)]
    nearest = negative.sort_values("candidate_rank").drop_duplicates("obs_idx")
    remaining = negative.drop(index=nearest.index)
    rng = np.random.default_rng([SEED, int(trip_id)])
    additional = remaining.sample(n=min(len(remaining), 2 * len(positive)), random_state=rng)
    return pd.concat([positive, nearest, additional], ignore_index=True).loc[:, [*FEATURE_COLUMNS, "label"]]


def reference_labels(data, partition, trip_id):
    path = data / "processed" / "labels" / partition / f"{trip_id}.csv"
    labels = pd.read_csv(path, usecols=["obs_idx", "label_arc_id"]).sort_values("obs_idx")
    if not np.array_equal(labels["obs_idx"].to_numpy(), np.arange(len(labels))):
        raise ValueError(f"Expected a dense zero-based observation index in {path}.")
    return labels["label_arc_id"].fillna(-1).to_numpy(dtype=np.int64)


def fit_classifier(factory, samples, trip_ids):
    model = factory()
    rows = pd.concat([samples[trip_id] for trip_id in trip_ids], ignore_index=True)
    start = perf_counter()
    model.fit(rows.loc[:, list(FEATURE_COLUMNS)], rows["label"])
    return model, perf_counter() - start


def evaluate_trip(model, rows, truth, trip=None, config=None):
    started = perf_counter()
    scores = score_candidates(model, rows)
    ranked = scores.sort_values(["obs_idx", "score", "cand_arc_id"], ascending=[True, False, True])
    best = ranked.drop_duplicates("obs_idx")
    pointwise = np.full(len(truth), -1, dtype=np.int64)
    pointwise[best["obs_idx"].to_numpy()] = best["cand_arc_id"].to_numpy()
    elapsed = perf_counter() - started

    labeled = truth >= 0
    if not labeled.any():
        raise ValueError("Trip has no reference labels for evaluation.")
    candidate_truth = truth[scores["obs_idx"].to_numpy()]
    valid_rows = candidate_truth >= 0
    arc_ids = scores["cand_arc_id"].to_numpy()
    target = (arc_ids == candidate_truth)[valid_rows]
    probabilities = scores["score"].to_numpy(dtype=np.float64)[valid_rows]
    binary = probabilities >= 0.5
    observed_hits = scores["obs_idx"].to_numpy()[arc_ids == candidate_truth]
    metrics = {
        "top1_accuracy": float(np.mean(pointwise[labeled] == truth[labeled])),
        "candidate_recall": len(np.unique(observed_hits)) / int(labeled.sum()),
        "candidate_accuracy": accuracy_score(target, binary) if len(target) else np.nan,
        "macro_f1": f1_score(target, binary, labels=[False, True], average="macro", zero_division=0)
        if len(target) else np.nan,
        "precision": precision_score(target, binary, zero_division=0) if len(target) else np.nan,
        "recall": recall_score(target, binary, zero_division=0) if len(target) else np.nan,
        "log_loss": log_loss(target, probabilities, labels=[False, True]) if len(target) else np.nan,
        "inference_seconds": elapsed,
    }
    if trip is not None:
        endpoints = trip.arcs.loc[:, ["from_node", "to_node"]].to_numpy(dtype=np.int64)
        times = trip.track["t"].to_numpy(dtype=float)
        if len(times) != len(truth):
            raise ValueError(f"Reference labels and GPS length disagree for {trip.trip_id}.")
        started = perf_counter()
        decoded = decode(scores, endpoints, times, config)
        metrics["decoder_seconds"] = perf_counter() - started
        metrics["decoded_accuracy"] = float(np.mean(decoded[labeled] == truth[labeled]))
        delta = np.diff(times)
        pairs = (delta > 0) & (delta <= config.max_gap_seconds)
        metrics["direct_continuity"] = continuity(pointwise, endpoints, pairs)
        metrics["decoded_direct_continuity"] = continuity(decoded, endpoints, pairs)
        metrics["unmatched_observations"] = int((decoded < 0).sum())
    return metrics


def continuity(predictions, endpoints, pairs):
    if not pairs.any():
        return np.nan
    left, right = predictions[:-1][pairs], predictions[1:][pairs]
    matched = (left >= 0) & (right >= 0)
    left, right = left[matched], right[matched]
    return np.count_nonzero((left == right) | (endpoints[left, 1] == endpoints[right, 0])) / pairs.sum()


def load_folds(train_ids, path):
    if path is None:
        ids = np.asarray(train_ids)
        splitter = KFold(n_splits=3, shuffle=True, random_state=SEED)
        return [(ids[fit].tolist(), ids[validation].tolist()) for fit, validation in splitter.split(ids)]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    folds = [(fold["fit"], fold["validation"]) for fold in manifest["folds"]]
    valid_ids = set(train_ids)
    if not folds:
        raise ValueError("The split manifest contains no folds.")
    for fit, validation in folds:
        if (not fit or not validation or len(set(fit)) != len(fit) or len(set(validation)) != len(validation)
                or set(fit) & set(validation) or not (set(fit) | set(validation)) <= valid_ids):
            raise ValueError("Each fold needs disjoint, nonempty sets of unique training trip IDs.")
    return folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "role_c")
    parser.add_argument("--folds-json", type=Path, help="Optional Role B manifest with folds[].fit and folds[].validation.")
    parser.add_argument("--jobs", type=int, choices=range(1, 17), default=4, metavar="1..16")
    args = parser.parse_args()

    data = args.data_dir.resolve()
    output = args.output_dir.resolve()
    train_ids = [path.name for path in list_trips(data / "train")]
    test_dirs = list_trips(data / "student_test")
    if len(train_ids) < 3 or not test_dirs:
        parser.error("At least three training trips and one student-test trip are required.")
    folds = load_folds(train_ids, args.folds_json)
    config = ViterbiConfig.from_json(ARTIFACTS / "hmm_config.json")
    factories = model_factories(SEED)
    provenance = "provided_role_b_manifest" if args.folds_json else "provisional_role_c_splits"
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(json.dumps({
        "seed": SEED,
        "split_source": provenance,
        "folds": [{"fit": fit, "validation": valid} for fit, valid in folds],
        "student_test_trips": [path.name for path in test_dirs],
        "feature_columns": list(FEATURE_COLUMNS),
        "candidate_settings": "Existing Role A tables, calibrated on the full training partition.",
        "training_sample": "All positives, nearest negative per observation, up to two extra random negatives per positive per trip.",
    }, indent=2), encoding="utf-8")

    with threadpool_limits(limits=args.jobs):
        print(f"Reading {len(train_ids)} training trips ({provenance}).", flush=True)
        samples = {trip_id: sample_training_trip(data, trip_id) for trip_id in train_ids}
        test = [(load_trip(path, load_route=False),
                 read_candidates(data, "student_test", path.name),
                 reference_labels(data, "student_test", path.name)) for path in test_dirs]
        test_metrics = []
        for name, factory in factories.items():
            model, training_seconds = fit_classifier(factory, samples, train_ids)
            metrics = [dict(model=name, trip_id=trip.trip_id,
                            **evaluate_trip(model, rows, truth, trip, config)) for trip, rows, truth in test]
            test_metrics.extend(metrics)
            pd.DataFrame(test_metrics).to_csv(output / "student_test_per_trip.csv", index=False)
            joblib.dump({"model": model, "model_name": name, "feature_columns": FEATURE_COLUMNS,
                         "training_seconds": training_seconds}, output / f"{name}.joblib")
            print(f"{name}: finished student-test predictions and saved scorer.", flush=True)
        del test

        validation_metrics = []
        for number, (fit, validation) in enumerate(folds, 1):
            fitted = {name: fit_classifier(factory, samples, fit) for name, factory in factories.items()}
            for trip_id in validation:
                rows = read_candidates(data, "train", trip_id)
                truth = reference_labels(data, "train", trip_id)
                for name, (model, training_seconds) in fitted.items():
                    validation_metrics.append(dict(model=name, fold=number, trip_id=trip_id,
                                                   fold_training_seconds=training_seconds,
                                                   **evaluate_trip(model, rows, truth)))
            pd.DataFrame(validation_metrics).to_csv(output / "cv_per_trip.csv", index=False)
            print(f"Finished validation fold {number}/{len(folds)}.", flush=True)

    cv = pd.DataFrame(validation_metrics)
    student_test = pd.DataFrame(test_metrics)
    cv.drop(columns="fold").groupby("model").mean(numeric_only=True).to_csv(output / "cv_summary.csv")
    student_test.groupby("model").mean(numeric_only=True).to_csv(output / "student_test_summary.csv")
    print(f"Role C outputs saved to {output}", flush=True)


if __name__ == "__main__":
    main()
