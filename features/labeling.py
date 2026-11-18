"""Assign route-reference labels to GPS observations for supervised experiments."""

import json

import numpy as np
import pandas as pd
from shapely import distance, points
from shapely.geometry import LineString

from common.graph import build_arc_geometries
from common.loader import Trip, list_trips, load_trip, validate_trip
from common.paths import PROCESSED_DIR, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from common.projection import lonlat_to_metres

FAR_ROUTE_QUANTILE = 0.95


def _assign_monotone_positions(
    gps_points: np.ndarray,
    route_lines: np.ndarray,
    route_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Minimize total squared GPS-to-route distance while forbidding route-order reversal."""
    point_count = len(gps_points)
    state_count = len(route_positions)
    pointer_dtype = np.uint16 if state_count <= np.iinfo(np.uint16).max else np.uint32
    back_pointers = np.empty((point_count, state_count), dtype=pointer_dtype)
    nearest_local = np.empty(point_count, dtype=np.int64)
    nearest_distances = np.empty(point_count, dtype=float)
    ambiguous = np.empty(point_count, dtype=bool)
    previous_cost = np.empty(state_count, dtype=float)

    for point_index, gps_point in enumerate(gps_points):
        distances = np.asarray(distance(gps_point, route_lines), dtype=float)
        nearest_local[point_index] = int(np.argmin(distances))
        nearest_distances[point_index] = float(distances[nearest_local[point_index]])
        ambiguous[point_index] = int(
            np.isclose(distances, nearest_distances[point_index], rtol=0.0, atol=1e-6).sum()
        ) > 1
        point_cost = distances * distances
        if point_index == 0:
            previous_cost[:] = point_cost
            continue

        prefix_cost = np.minimum.accumulate(previous_cost)
        new_prefix_minimum = np.zeros(state_count, dtype=bool)
        new_prefix_minimum[0] = True
        new_prefix_minimum[1:] = previous_cost[1:] < prefix_cost[:-1]
        best_previous = np.maximum.accumulate(
            np.where(new_prefix_minimum, np.arange(state_count), 0)
        )
        back_pointers[point_index] = best_previous.astype(pointer_dtype)
        previous_cost = point_cost + prefix_cost

    assigned_local = np.empty(point_count, dtype=np.int64)
    state = int(np.argmin(previous_cost))
    for point_index in range(point_count - 1, -1, -1):
        assigned_local[point_index] = state
        if point_index:
            state = int(back_pointers[point_index, state])
    return nearest_local, nearest_distances, ambiguous, assigned_local


def label_trip(trip: Trip, far_threshold_m: float | None = None) -> pd.DataFrame:
    """Label GPS rows by the closest globally consistent, forward-only route sequence."""
    if trip.route is None:
        raise ValueError(f"Trip {trip.trip_id} has no reference route; labels are unavailable.")

    arc_geometry = build_arc_geometries(trip)
    geometry_index = {int(arc_id): index for index, arc_id in enumerate(arc_geometry.arc_ids)}
    route_positions = np.array(
        [position for position, arc_id in enumerate(trip.route) if int(arc_id) in geometry_index],
        dtype=np.int64,
    )
    route_arc_ids = trip.route[route_positions].astype(np.int64)
    if not len(route_positions):
        raise ValueError(f"Trip {trip.trip_id} has no usable reference-route arcs.")
    route_lines = np.asarray(
        [
            LineString(
                [
                    (
                        arc_geometry.start_x_m[geometry_index[int(arc_id)]],
                        arc_geometry.start_y_m[geometry_index[int(arc_id)]],
                    ),
                    (
                        arc_geometry.end_x_m[geometry_index[int(arc_id)]],
                        arc_geometry.end_y_m[geometry_index[int(arc_id)]],
                    ),
                ]
            )
            for arc_id in route_arc_ids
        ],
        dtype=object,
    )

    track = trip.track
    lon = track["lon"].to_numpy(dtype=float)
    lat = track["lat"].to_numpy(dtype=float)
    valid_gps = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (np.abs(lon) <= 180)
        & (np.abs(lat) <= 90)
    )
    valid_obs = np.flatnonzero(valid_gps)
    if not len(valid_obs):
        raise ValueError(f"Trip {trip.trip_id} has no valid GPS observations to label.")
    x, y = lonlat_to_metres(lon[valid_gps], lat[valid_gps], arc_geometry.crs)
    gps_points = points(x, y)
    (
        nearest_local,
        nearest_distances,
        ambiguous,
        assigned_local,
    ) = _assign_monotone_positions(
        gps_points,
        route_lines,
        route_positions,
    )
    nearest_positions = route_positions[nearest_local]
    assigned_positions = route_positions[assigned_local]
    assigned_distances = np.asarray(
        distance(gps_points, route_lines[assigned_local]), dtype=float
    )
    assigned_arc_ids = trip.route[assigned_positions].astype(np.int64)

    output = pd.DataFrame(
        {
            "trip_id": trip.trip_id,
            "obs_idx": np.arange(len(track), dtype=np.int64),
            "label_arc_id": pd.array([pd.NA] * len(track), dtype="Int64"),
            "route_pos": pd.array([pd.NA] * len(track), dtype="Int64"),
            "nearest_route_pos": pd.array([pd.NA] * len(track), dtype="Int64"),
            "nearest_distance_m": np.nan,
            "assigned_distance_m": np.nan,
            "is_ambiguous": pd.array([pd.NA] * len(track), dtype="boolean"),
            "is_far_from_route": pd.array([pd.NA] * len(track), dtype="boolean"),
        }
    )
    output.loc[valid_obs, "label_arc_id"] = assigned_arc_ids
    output.loc[valid_obs, "route_pos"] = assigned_positions
    output.loc[valid_obs, "nearest_route_pos"] = nearest_positions
    output.loc[valid_obs, "nearest_distance_m"] = nearest_distances
    output.loc[valid_obs, "assigned_distance_m"] = assigned_distances
    output.loc[valid_obs, "is_ambiguous"] = ambiguous
    if far_threshold_m is not None:
        output.loc[valid_obs, "is_far_from_route"] = nearest_distances > far_threshold_m
    return output


def _trip_summary(
    trip: Trip, labels: pd.DataFrame, partition: str, far_threshold_m: float | None
) -> dict[str, int | float | str]:
    """Summarize route coverage and point-to-route error without trusting labels as truth."""
    route = trip.route
    assert route is not None
    geometry = build_arc_geometries(trip)
    usable_arc_ids = set(geometry.arc_ids.tolist())
    valid_route_positions = np.array(
        [position for position, arc_id in enumerate(route) if int(arc_id) in usable_arc_ids],
        dtype=np.int64,
    )
    positions = labels["route_pos"].dropna().to_numpy(dtype=np.int64)
    counts = np.bincount(positions, minlength=len(route))
    valid_counts = counts[valid_route_positions]
    nearest = labels["nearest_distance_m"].dropna().to_numpy(dtype=float)
    assigned = labels["assigned_distance_m"].dropna().to_numpy(dtype=float)
    result: dict[str, int | float | str] = {
        "partition": partition,
        "trip_id": trip.trip_id,
        "gps_rows": len(trip.track),
        "route_arc_occurrences": len(route),
        "usable_route_occurrences": len(valid_route_positions),
        "labelled_observations": len(positions),
        "route_occurrences_without_observations": int((valid_counts == 0).sum()),
        "unique_route_arc_ids_with_observations": int(
            labels["label_arc_id"].dropna().nunique()
        ),
        "observations_per_route_occurrence_median": float(np.median(valid_counts))
        if len(valid_counts)
        else np.nan,
        "observations_per_route_occurrence_p95": float(np.percentile(valid_counts, 95))
        if len(valid_counts)
        else np.nan,
        "observations_per_route_occurrence_max": int(valid_counts.max())
        if len(valid_counts)
        else 0,
        "exact_nearest_tie_observations": int(labels["is_ambiguous"].fillna(False).sum()),
        "monotone_adjusted_observations": int(
            (labels["route_pos"].astype("Int64") != labels["nearest_route_pos"].astype("Int64"))
            .fillna(False)
            .sum()
        ),
        "nearest_route_distance_median": float(np.median(nearest)) if len(nearest) else np.nan,
        "nearest_route_distance_p95": float(np.percentile(nearest, 95)) if len(nearest) else np.nan,
        "nearest_route_distance_max": float(nearest.max()) if len(nearest) else np.nan,
        "assigned_route_distance_median": float(np.median(assigned)) if len(assigned) else np.nan,
        "assigned_route_distance_p95": float(np.percentile(assigned, 95)) if len(assigned) else np.nan,
        "assigned_route_distance_max": float(assigned.max()) if len(assigned) else np.nan,
        "route_discontinuities": int(validate_trip(trip)["route_discontinuities"]),
        "far_nearest_points": int((nearest > far_threshold_m).sum())
        if far_threshold_m is not None
        else 0,
    }
    return result


def _route_occurrence_counts(
    trip: Trip, labels: pd.DataFrame, partition: str
) -> pd.DataFrame:
    """Keep per-occurrence counts so repeated arcs remain distinct in route order."""
    route = trip.route
    assert route is not None
    positions = labels["route_pos"].dropna().to_numpy(dtype=np.int64)
    counts = np.bincount(positions, minlength=len(route))
    usable_arc_ids = set(build_arc_geometries(trip).arc_ids.tolist())
    return pd.DataFrame(
        {
            "partition": partition,
            "trip_id": trip.trip_id,
            "route_pos": np.arange(len(route), dtype=np.int64),
            "arc_id": route.astype(np.int64),
            "arc_geometry_valid": [int(arc_id) in usable_arc_ids for arc_id in route],
            "gps_observation_count": counts,
        }
    )


def main() -> None:
    """Fit the far-route diagnostic on train, then label both available partitions."""
    train_results = []
    train_distances = []
    for trip_dir in list_trips(TRAIN_DIR):
        trip = load_trip(trip_dir)
        labels = label_trip(trip)
        train_results.append((trip, labels))
        train_distances.append(labels["nearest_distance_m"].dropna().to_numpy(dtype=float))

    all_train_distances = np.concatenate(train_distances)
    far_threshold = float(np.quantile(all_train_distances, FAR_ROUTE_QUANTILE))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "labeling_threshold.json").write_text(
        json.dumps(
            {
                "partition_used": "train",
                "quantile": FAR_ROUTE_QUANTILE,
                "far_route_threshold_m": far_threshold,
                "training_observation_count": len(all_train_distances),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_root = PROCESSED_DIR / "labels"
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    occurrence_frames = []
    for partition, root, cached in (
        ("train", TRAIN_DIR, train_results),
        ("student_test", STUDENT_TEST_DIR, None),
    ):
        output_dir = output_root / partition
        output_dir.mkdir(parents=True, exist_ok=True)
        results = cached
        if results is None:
            results = []
            for trip_dir in list_trips(root):
                trip = load_trip(trip_dir)
                results.append((trip, label_trip(trip, far_threshold)))
        for trip, labels in results:
            if partition == "train":
                valid_labels = labels["nearest_distance_m"].notna()
                labels.loc[valid_labels, "is_far_from_route"] = (
                    labels.loc[valid_labels, "nearest_distance_m"] > far_threshold
                )
            labels.to_csv(output_dir / f"{trip.trip_id}.csv", index=False)
            summaries.append(_trip_summary(trip, labels, partition, far_threshold))
            occurrence_frames.append(_route_occurrence_counts(trip, labels, partition))

    frame = pd.DataFrame(summaries)
    (RESULTS_DIR / "eda").mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "eda" / "labeling_per_trip.csv", index=False)
    pd.concat(occurrence_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "eda" / "route_observations_per_occurrence.csv", index=False
    )
    partition_summary = frame.groupby("partition").agg(
        trip_count=("trip_id", "count"),
        gps_rows=("gps_rows", "sum"),
        route_arc_occurrences=("route_arc_occurrences", "sum"),
        labelled_observations=("labelled_observations", "sum"),
        ambiguous_observations=("exact_nearest_tie_observations", "sum"),
        monotone_adjusted_observations=("monotone_adjusted_observations", "sum"),
        far_nearest_points=("far_nearest_points", "sum"),
        trips_with_route_discontinuities=("route_discontinuities", lambda values: int((values > 0).sum())),
        median_observations_per_route_occurrence=("observations_per_route_occurrence_median", "median"),
        median_assigned_route_distance_p95_m=("assigned_route_distance_p95", "median"),
    ).reset_index()
    partition_summary.to_csv(RESULTS_DIR / "eda" / "labeling_summary.csv", index=False)
    print(f"Train-only far-route threshold (p95): {far_threshold:.6f} m")
    print(partition_summary.to_string(index=False))


if __name__ == "__main__":
    main()
