"""Fit train-only candidate settings and measure candidate recall and density."""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.loader import list_trips, load_trip
from common.paths import PROCESSED_DIR, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from features.candidates import _generate_candidate_table, build_candidates

LABEL_RADIUS_QUANTILE = 0.99
LABEL_RANK_QUANTILE = 0.99


def _read_label_arc_ids(partition: str, trip_id: str) -> np.ndarray:
    """Load route labels produced by labeling.py for offline recall measurement."""
    path = PROCESSED_DIR / "labels" / partition / f"{trip_id}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run python -m features.labeling first: {path}")
    labels = pd.read_csv(path, usecols=["label_arc_id"])["label_arc_id"]
    return pd.to_numeric(labels, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)


def fit_candidate_settings() -> tuple[float, int, dict[str, object]]:
    """Choose radius and cap quantiles from train labels and candidate ranks only."""
    assigned_distances = []
    for trip_dir in list_trips(TRAIN_DIR):
        labels = pd.read_csv(
            PROCESSED_DIR / "labels" / "train" / f"{trip_dir.name}.csv",
            usecols=["assigned_distance_m"],
        )
        assigned_distances.append(labels["assigned_distance_m"].dropna().to_numpy(dtype=float))
    all_distances = np.concatenate(assigned_distances)
    radius_m = float(np.quantile(all_distances, LABEL_RADIUS_QUANTILE))

    label_ranks = []
    radius_hits = 0
    for trip_dir in list_trips(TRAIN_DIR):
        trip = load_trip(trip_dir, load_route=False)
        arc_labels = _read_label_arc_ids("train", trip.trip_id)
        _, _, _, ranks = _generate_candidate_table(
            trip,
            radius_m,
            max_candidates=max(1, len(trip.arcs)),
            label_arc_ids=arc_labels,
            collect_rows=False,
        )
        route_ranks = ranks[arc_labels >= 0]
        label_ranks.extend(route_ranks[route_ranks > 0].tolist())
        radius_hits += int((route_ranks > 0).sum())

    if not label_ranks:
        raise ValueError("No training reference labels fall within the selected candidate radius.")
    cap = max(1, int(math.ceil(np.quantile(label_ranks, LABEL_RANK_QUANTILE))))
    config: dict[str, object] = {
        "partition_used": "train",
        "radius_m": radius_m,
        "label_distance_quantile": LABEL_RADIUS_QUANTILE,
        "max_candidates": cap,
        "label_candidate_rank_quantile": LABEL_RANK_QUANTILE,
        "training_label_observations": int(len(all_distances)),
        "training_labels_within_radius": radius_hits,
        "training_label_rank_p95": float(np.quantile(label_ranks, 0.95)),
        "training_label_rank_p99": float(np.quantile(label_ranks, LABEL_RANK_QUANTILE)),
        "training_label_rank_max": int(max(label_ranks)),
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "candidate_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return radius_m, cap, config


def _evaluate_trip(
    partition: str, trip_dir: Path, radius_m: float, cap: int
) -> dict[str, int | float | str]:
    """Measure capped candidates against the saved route label for every observation."""
    trip = load_trip(trip_dir, load_route=False)
    label_arc_ids = _read_label_arc_ids(partition, trip.trip_id)
    start = time.perf_counter()
    _, radius_counts, kept_counts, ranks = _generate_candidate_table(
        trip,
        radius_m,
        cap,
        label_arc_ids,
        collect_rows=False,
    )
    elapsed = time.perf_counter() - start
    labelled = label_arc_ids >= 0
    radius_misses = int((labelled & (ranks == 0)).sum())
    cap_misses = int((labelled & (ranks > cap)).sum())
    total_misses = radius_misses + cap_misses
    denominator = int(labelled.sum())
    return {
        "partition": partition,
        "trip_id": trip.trip_id,
        "gps_rows": len(trip.track),
        "candidate_rows": int(kept_counts.sum()),
        "average_candidates_per_point": float(kept_counts.mean()),
        "average_candidates_within_radius": float(radius_counts.mean()),
        "labelled_observations": denominator,
        "radius_misses": radius_misses,
        "cap_misses": cap_misses,
        "candidate_misses": total_misses,
        "candidate_miss_rate": float(total_misses / denominator) if denominator else np.nan,
        "generation_runtime_s": float(elapsed),
    }


def main() -> None:
    """Fit on train, evaluate both available partitions, and print Checkpoint 2 evidence."""
    radius_m, cap, config = fit_candidate_settings()
    per_trip = []
    for partition, root in (("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)):
        per_trip.extend(
            _evaluate_trip(partition, trip_dir, radius_m, cap)
            for trip_dir in list_trips(root)
        )
    frame = pd.DataFrame(per_trip)
    output_dir = RESULTS_DIR / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "candidate_precheck_summary.csv", index=False)
    partition_summary = frame.groupby("partition").apply(
        lambda part: pd.Series(
            {
                "trip_count": len(part),
                "gps_rows": int(part["gps_rows"].sum()),
                "candidate_rows": int(part["candidate_rows"].sum()),
                "average_candidates_per_point": float(
                    part["candidate_rows"].sum() / part["gps_rows"].sum()
                ),
                "labelled_observations": int(part["labelled_observations"].sum()),
                "candidate_misses": int(part["candidate_misses"].sum()),
                "candidate_miss_rate": float(
                    part["candidate_misses"].sum() / part["labelled_observations"].sum()
                ),
                "radius_misses": int(part["radius_misses"].sum()),
                "cap_misses": int(part["cap_misses"].sum()),
            }
        )
    ).reset_index()
    partition_summary.to_csv(
        output_dir / "candidate_precheck_partition_summary.csv", index=False
    )

    sample_dir = TRAIN_DIR / "00000001"
    sample_start = time.perf_counter()
    sample_candidates = build_candidates(
        sample_dir,
        load_route=False,
        radius_m=radius_m,
        max_candidates=cap,
    )
    sample_runtime = time.perf_counter() - sample_start
    sample_trip = load_trip(sample_dir, load_route=False)
    sample_average = len(sample_candidates) / len(sample_trip.track)

    print(f"Train-only radius: {radius_m:.6f} m (p{LABEL_RADIUS_QUANTILE * 100:g} assigned-label distance)")
    print(f"Train-only candidate cap: {cap} (p{LABEL_RANK_QUANTILE * 100:g} label rank within radius)")
    print("Partition results:")
    print(partition_summary.to_string(index=False))
    print(
        "Unlabeled inference-style runtime on train trip "
        f"{sample_trip.trip_id}: {sample_runtime:.3f} s; "
        f"{len(sample_candidates)} rows; {sample_average:.3f} candidates/GPS point."
    )
    print(f"Saved train-only configuration: {config}")


if __name__ == "__main__":
    main()
