"""Generate radius-limited arc candidates with per-trip metric spatial indexes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import distance, points
from shapely.strtree import STRtree

from common.graph import build_arc_geometries
from common.loader import Trip, load_trip
from common.paths import PROCESSED_DIR
from common.projection import lonlat_to_metres
from features.labeling import label_trip

QUERY_BATCH_SIZE = 256


def _read_candidate_config() -> dict[str, float | int]:
    """Load train-fitted search settings instead of inventing inference defaults."""
    config_path = PROCESSED_DIR / "candidate_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Fit candidate settings on train first: {config_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("partition_used") != "train":
        raise ValueError("Candidate radius and cap must be selected using train only.")
    return config


def _generate_candidate_table(
    trip: Trip,
    radius_m: float,
    max_candidates: int,
    label_arc_ids: np.ndarray | None,
    collect_rows: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Query nearby arcs in batches so dense intersections do not exhaust memory."""
    if radius_m <= 0 or max_candidates < 1:
        raise ValueError("radius_m and max_candidates must be positive.")
    if label_arc_ids is not None and len(label_arc_ids) != len(trip.track):
        raise ValueError("There must be one label arc ID per GPS observation.")

    arc_geometry = build_arc_geometries(trip)
    if not len(arc_geometry.arc_ids):
        empty_columns = ["trip_id", "obs_idx", "cand_arc_id", "dist_m"]
        if label_arc_ids is not None:
            empty_columns.append("label")
        empty = pd.DataFrame(columns=empty_columns)
        size = len(trip.track)
        return empty, np.zeros(size, dtype=int), np.zeros(size, dtype=int), np.zeros(size, dtype=int)

    lines = arc_geometry.as_lines()
    tree = STRtree(lines)
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
    projected_x, projected_y = lonlat_to_metres(
        lon[valid_gps], lat[valid_gps], arc_geometry.crs
    )
    gps_points = points(projected_x, projected_y)
    candidate_counts = np.zeros(len(track), dtype=np.int64)
    kept_counts = np.zeros(len(track), dtype=np.int64)
    label_ranks = np.zeros(len(track), dtype=np.int64)
    output_frames = []

    for batch_start in range(0, len(valid_obs), QUERY_BATCH_SIZE):
        batch_end = min(batch_start + QUERY_BATCH_SIZE, len(valid_obs))
        batch_points = gps_points[batch_start:batch_end]
        pairs = tree.query(batch_points, predicate="dwithin", distance=radius_m)
        if pairs.shape[1] == 0:
            continue
        point_indices, tree_indices = pairs
        distances = np.asarray(
            distance(batch_points[point_indices], lines[tree_indices]), dtype=float
        )
        arc_ids = arc_geometry.arc_ids[tree_indices]
        order = np.lexsort((arc_ids, distances, point_indices))
        sorted_points = point_indices[order]
        sorted_arcs = arc_ids[order]
        sorted_distances = distances[order]
        group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_points)) + 1]
        group_ends = np.r_[group_starts[1:], len(order)]
        selected_order = []

        for group_start, group_end in zip(group_starts, group_ends):
            local_obs = int(sorted_points[group_start])
            obs_idx = int(valid_obs[batch_start + local_obs])
            group_size = int(group_end - group_start)
            candidate_counts[obs_idx] = group_size
            if label_arc_ids is not None:
                matching = np.flatnonzero(sorted_arcs[group_start:group_end] == label_arc_ids[obs_idx])
                if len(matching):
                    label_ranks[obs_idx] = int(matching[0] + 1)
            selected_end = min(int(group_end), int(group_start + max_candidates))
            if collect_rows:
                selected_order.extend(range(int(group_start), selected_end))
            kept_counts[obs_idx] = selected_end - int(group_start)

        if collect_rows and selected_order:
            selected = np.asarray(selected_order, dtype=np.int64)
            batch_obs = valid_obs[batch_start + sorted_points[selected]]
            data: dict[str, object] = {
                "trip_id": trip.trip_id,
                "obs_idx": batch_obs.astype(np.int64),
                "cand_arc_id": sorted_arcs[selected].astype(np.int64),
                "dist_m": sorted_distances[selected],
            }
            if label_arc_ids is not None:
                data["label"] = (
                    sorted_arcs[selected] == label_arc_ids[batch_obs]
                ).astype(np.int8)
            output_frames.append(pd.DataFrame(data))

    columns = ["trip_id", "obs_idx", "cand_arc_id", "dist_m"]
    if label_arc_ids is not None:
        columns.append("label")
    candidates = (
        pd.concat(output_frames, ignore_index=True)
        if output_frames
        else pd.DataFrame(columns=columns)
    )
    return candidates, candidate_counts, kept_counts, label_ranks


def build_candidates(
    trip_dir: Path | str,
    load_route: bool = False,
    radius_m: float | None = None,
    max_candidates: int | None = None,
) -> pd.DataFrame:
    """Return each observation's closest nearby arcs, with labels only when opted in."""
    config = _read_candidate_config() if radius_m is None or max_candidates is None else {}
    radius = float(radius_m if radius_m is not None else config["radius_m"])
    cap = int(max_candidates if max_candidates is not None else config["max_candidates"])
    trip = load_trip(trip_dir, load_route=load_route)
    label_arc_ids = None
    if load_route:
        if trip.route is None:
            raise ValueError(f"Trip {trip.trip_id} has no route to generate labels.")
        labels = label_trip(trip)
        label_arc_ids = labels["label_arc_id"].fillna(-1).to_numpy(dtype=np.int64)
    candidates, _, _, _ = _generate_candidate_table(
        trip, radius, cap, label_arc_ids, collect_rows=True
    )
    return candidates
