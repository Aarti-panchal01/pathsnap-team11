"""Add auditable time, coordinate, and network-proximity flags to each GPS row."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.geometry import nearest_network_distances_m
from common.loader import Trip, list_trips, load_trip, validate_trip
from common.paths import PROCESSED_DIR, STUDENT_TEST_DIR, TRAIN_DIR

FAR_DISTANCE_QUANTILE = 0.99
GAP_THRESHOLD_SECONDS = 5.0
_TRAIN_THRESHOLD: float | None = None


def fit_far_threshold() -> tuple[float, dict[str, np.ndarray]]:
    """Fit the far-from-network cutoff using training GPS rows only."""
    per_trip: dict[str, np.ndarray] = {}
    values = []
    for trip_dir in list_trips(TRAIN_DIR):
        trip = load_trip(trip_dir, load_route=False)
        distances = nearest_network_distances_m(trip)
        per_trip[trip.trip_id] = distances
        values.append(distances[np.isfinite(distances)])
    observed = np.concatenate(values) if values else np.empty(0)
    if not len(observed):
        raise ValueError("No valid training GPS-to-network distances are available.")
    threshold = float(np.quantile(observed, FAR_DISTANCE_QUANTILE))
    calibration = {
        "threshold_m": threshold,
        "quantile": FAR_DISTANCE_QUANTILE,
        "training_point_count": int(len(observed)),
        "training_distance_median": float(np.median(observed)),
        "training_distance_p95_m": float(np.quantile(observed, 0.95)),
        "training_distance_p99_m": threshold,
        "training_distance_max_m": float(observed.max()),
        "partition_used": "train",
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "preprocess_threshold.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    return threshold, per_trip


def _get_far_threshold() -> float:
    """Reuse the saved train-only calibration or create it once when absent."""
    global _TRAIN_THRESHOLD
    if _TRAIN_THRESHOLD is not None:
        return _TRAIN_THRESHOLD
    calibration_path = PROCESSED_DIR / "preprocess_threshold.json"
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("partition_used") != "train":
            raise ValueError("Network-distance threshold must be calibrated on train only.")
        _TRAIN_THRESHOLD = float(calibration["threshold_m"])
    else:
        _TRAIN_THRESHOLD, _ = fit_far_threshold()
    return _TRAIN_THRESHOLD


def preprocess_trip(
    trip: Trip,
    far_threshold_m: float | None = None,
    network_distances_m: np.ndarray | None = None,
) -> Trip:
    """Preserve every source row while adding explicit flags for later decisions."""
    threshold = far_threshold_m if far_threshold_m is not None else _get_far_threshold()
    distances = (
        nearest_network_distances_m(trip)
        if network_distances_m is None
        else np.asarray(network_distances_m, dtype=float)
    )
    if len(distances) != len(trip.track):
        raise ValueError("Network-distance array must have one value per GPS row.")

    track = trip.track.copy()
    lon = track["lon"].to_numpy(dtype=float)
    lat = track["lat"].to_numpy(dtype=float)
    time = track["t"].to_numpy(dtype=float)
    valid_coord = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (np.abs(lon) <= 180)
        & (np.abs(lat) <= 90)
    )
    time_step = np.full(len(track), np.nan, dtype=float)
    if len(track) > 1:
        time_step[1:] = np.diff(time)
    track["is_valid_coord"] = valid_coord
    track["is_valid_timestamp"] = np.isfinite(time)
    track["time_step_s"] = time_step
    track["is_non_increasing_time"] = np.isfinite(time_step) & (time_step <= 0)
    track["gap_before_s"] = time_step
    track["is_gap_point"] = np.isfinite(time_step) & (time_step > GAP_THRESHOLD_SECONDS)
    track["distance_to_network_m"] = distances
    far = pd.Series(pd.NA, index=track.index, dtype="boolean")
    measurable = np.isfinite(distances)
    far.loc[measurable] = distances[measurable] > threshold
    track["is_far_from_network"] = far
    return Trip(
        trip_id=trip.trip_id,
        track=track,
        nodes=trip.nodes,
        arcs=trip.arcs,
        route=trip.route,
    )


def _process_partition(
    partition: str, root: Path, threshold: float, cached_distances: dict[str, np.ndarray] | None = None
) -> dict[str, int | str]:
    """Save annotated tracks outside the repository and report preserved flags."""
    output_dir = PROCESSED_DIR / "preprocess" / partition
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = 0
    far_points = 0
    gap_points = 0
    non_increasing_steps = 0
    invalid_coordinate_rows = 0
    broken_routes = 0
    for trip_dir in list_trips(root):
        trip = load_trip(trip_dir)
        distances = cached_distances.get(trip.trip_id) if cached_distances is not None else None
        processed = preprocess_trip(trip, threshold, distances)
        processed.track.to_csv(output_dir / f"{trip.trip_id}.csv", index=False)
        rows += len(processed.track)
        far_points += int(processed.track["is_far_from_network"].fillna(False).sum())
        gap_points += int(processed.track["is_gap_point"].sum())
        non_increasing_steps += int(processed.track["is_non_increasing_time"].sum())
        invalid_coordinate_rows += int((~processed.track["is_valid_coord"]).sum())
        route_quality = validate_trip(trip)
        broken_routes += int(
            route_quality["route_discontinuities"] > 0
            or route_quality["route_arc_ids_out_of_range"] > 0
        )
    return {
        "partition": partition,
        "trip_count": len(list_trips(root)),
        "gps_rows_preserved": rows,
        "invalid_coordinate_rows_flagged": invalid_coordinate_rows,
        "non_increasing_steps_flagged": non_increasing_steps,
        "gap_points_flagged": gap_points,
        "far_network_points_flagged": far_points,
        "trips_with_broken_reference_routes_flagged": broken_routes,
        "far_network_threshold_m": threshold,
    }


def main() -> None:
    """Calibrate on train, apply the same non-dropping transform to both partitions."""
    threshold, train_distances = fit_far_threshold()
    summaries = [
        _process_partition("train", TRAIN_DIR, threshold, train_distances),
        _process_partition("student_test", STUDENT_TEST_DIR, threshold),
    ]
    summary = pd.DataFrame(summaries)
    summary.to_csv(PROCESSED_DIR / "preprocess_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
