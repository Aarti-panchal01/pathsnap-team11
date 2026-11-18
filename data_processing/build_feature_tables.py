"""Create compressed per-trip candidate-feature tables outside the repository."""

import json
import time

import numpy as np
import pandas as pd

from common.loader import list_trips, load_trip
from common.paths import PROCESSED_DIR, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from features.candidates import _generate_candidate_table
from features.feature_engineering import CANDIDATE_FEATURE_COLUMNS, engineer_candidate_features

SAMPLE_ROWS_PER_TRIP = 50


def _read_train_fitted_settings() -> tuple[float, int]:
    """Reject candidate settings that were not fitted exclusively on train."""
    config_path = PROCESSED_DIR / "candidate_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run python -m data_processing.candidate_analysis first: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("partition_used") != "train":
        raise ValueError("Candidate settings must be fitted on train only.")
    return float(config["radius_m"]), int(config["max_candidates"])


def _read_label_arc_ids(partition: str, trip_id: str, expected_rows: int) -> np.ndarray:
    """Reuse the audited route labels instead of recomputing the dynamic program."""
    labels_path = PROCESSED_DIR / "labels" / partition / f"{trip_id}.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Run python -m features.labeling first: {labels_path}")
    labels = pd.read_csv(labels_path, usecols=["label_arc_id"])["label_arc_id"]
    if len(labels) != expected_rows:
        raise ValueError(f"Expected {expected_rows} labels for trip {trip_id}, found {len(labels)}.")
    return pd.to_numeric(labels, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)


def _build_trip(
    trip_dir,
    partition: str,
    output_dir,
    radius_m: float,
    max_candidates: int,
) -> tuple[dict[str, int | float | str], pd.DataFrame, dict[str, int]]:
    """Build and persist one trip at a time to keep peak memory bounded."""
    trip = load_trip(trip_dir, load_route=False)
    labels = _read_label_arc_ids(partition, trip.trip_id, len(trip.track))
    candidates, radius_counts, kept_counts, label_ranks = _generate_candidate_table(
        trip,
        radius_m,
        max_candidates,
        labels,
        collect_rows=True,
    )
    featured = engineer_candidate_features(trip, candidates)
    featured.insert(0, "partition", partition)
    output_path = output_dir / f"{trip.trip_id}.csv.gz"
    featured.to_csv(output_path, index=False, compression="gzip")

    labelled = labels >= 0
    radius_misses = int((labelled & (label_ranks == 0)).sum())
    cap_misses = int((labelled & (label_ranks > max_candidates)).sum())
    summary: dict[str, int | float | str] = {
        "partition": partition,
        "trip_id": trip.trip_id,
        "gps_rows": len(trip.track),
        "labelled_observations": int(labelled.sum()),
        "candidate_rows": len(featured),
        "average_candidates_per_point": float(kept_counts.mean()),
        "candidate_misses": radius_misses + cap_misses,
        "candidate_miss_rate": float((radius_misses + cap_misses) / labelled.sum())
        if labelled.any()
        else np.nan,
        "radius_misses": radius_misses,
        "cap_misses": cap_misses,
        "output_bytes": output_path.stat().st_size,
    }
    missing_counts = {
        column: int(featured[column].isna().sum())
        for column in CANDIDATE_FEATURE_COLUMNS
    }
    sample = featured.sample(
        n=min(SAMPLE_ROWS_PER_TRIP, len(featured)),
        random_state=11,
    )
    return summary, sample, missing_counts


def main() -> None:
    """Build both available partitions with train-fitted settings and no dropped rows."""
    radius_m, max_candidates = _read_train_fitted_settings()
    trip_summaries = []
    sample_frames = []
    missing_by_partition: dict[str, dict[str, int]] = {}
    rows_by_partition: dict[str, int] = {}
    for partition, root in (("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)):
        output_dir = PROCESSED_DIR / "candidate_features" / partition
        output_dir.mkdir(parents=True, exist_ok=True)
        missing_by_partition[partition] = {name: 0 for name in CANDIDATE_FEATURE_COLUMNS}
        rows_by_partition[partition] = 0
        for trip_dir in list_trips(root):
            started = time.perf_counter()
            summary, sample, missing_counts = _build_trip(
                trip_dir,
                partition,
                output_dir,
                radius_m,
                max_candidates,
            )
            summary["generation_runtime_s"] = time.perf_counter() - started
            trip_summaries.append(summary)
            sample_frames.append(sample)
            rows_by_partition[partition] += int(summary["candidate_rows"])
            for name, count in missing_counts.items():
                missing_by_partition[partition][name] += count
            print(
                f"{partition} {summary['trip_id']}: {summary['candidate_rows']} candidates "
                f"({summary['generation_runtime_s']:.2f} s)"
            )

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    per_trip = pd.DataFrame(trip_summaries)
    per_trip.to_csv(tables_dir / "candidate_feature_per_trip.csv", index=False)
    partition_summary = per_trip.groupby("partition", as_index=False).agg(
        trip_count=("trip_id", "count"),
        gps_rows=("gps_rows", "sum"),
        labelled_observations=("labelled_observations", "sum"),
        candidate_rows=("candidate_rows", "sum"),
        candidate_misses=("candidate_misses", "sum"),
        radius_misses=("radius_misses", "sum"),
        cap_misses=("cap_misses", "sum"),
        output_bytes=("output_bytes", "sum"),
        generation_runtime_s=("generation_runtime_s", "sum"),
    )
    partition_summary["average_candidates_per_point"] = (
        partition_summary["candidate_rows"] / partition_summary["gps_rows"]
    )
    partition_summary["candidate_miss_rate"] = (
        partition_summary["candidate_misses"] / partition_summary["labelled_observations"]
    )
    partition_summary.to_csv(tables_dir / "candidate_feature_partition_summary.csv", index=False)

    missing_rows = []
    for partition, counts in missing_by_partition.items():
        total = rows_by_partition[partition]
        for feature, missing_count in counts.items():
            missing_rows.append(
                {
                    "partition": partition,
                    "feature": feature,
                    "candidate_rows": total,
                    "missing_values": missing_count,
                    "missing_share": missing_count / total if total else np.nan,
                }
            )
    pd.DataFrame(missing_rows).to_csv(
        tables_dir / "candidate_feature_missing_summary.csv", index=False
    )
    pd.concat(sample_frames, ignore_index=True).to_csv(
        tables_dir / "candidate_feature_sample.csv", index=False
    )
    print("\nPartition summary:")
    print(partition_summary.to_string(index=False))
    print(f"\nCompressed tables: {PROCESSED_DIR / 'candidate_features'}")
    print(f"Small deterministic sample: {tables_dir / 'candidate_feature_sample.csv'}")


if __name__ == "__main__":
    main()
