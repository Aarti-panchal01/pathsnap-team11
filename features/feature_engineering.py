"""Build route-independent geometric and temporal features for arc candidates."""

import numpy as np
import pandas as pd

from common.graph import build_arc_geometries, build_node_arc_adjacency
from common.loader import Trip
from common.projection import lonlat_to_metres

CANDIDATE_FEATURE_COLUMNS = (
    "dist_m",
    "candidate_rank",
    "arc_length_m",
    "projection_fraction",
    "signed_cross_track_m",
    "point_to_start_m",
    "point_to_end_m",
    "arc_bearing_rad",
    "gps_heading_rad",
    "heading_difference_rad",
    "heading_alignment",
    "time_step_s",
    "gps_speed_mps",
    "is_gap_point",
    "from_node_in_degree",
    "from_node_out_degree",
    "to_node_in_degree",
    "to_node_out_degree",
)

GAP_THRESHOLD_SECONDS = 5.0


def engineer_candidate_features(trip: Trip, candidates: pd.DataFrame) -> pd.DataFrame:
    """Combine candidate geometry with same-trip motion context without route leakage."""
    required = {"trip_id", "obs_idx", "cand_arc_id", "dist_m"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table is missing required columns: {sorted(missing)}")
    if candidates.empty:
        output = candidates.copy()
        for column in CANDIDATE_FEATURE_COLUMNS:
            if column not in output:
                output[column] = pd.Series(dtype="float64")
        return output

    output = candidates.sort_values(
        ["obs_idx", "dist_m", "cand_arc_id"], kind="mergesort"
    ).reset_index(drop=True).copy()
    obs_idx = output["obs_idx"].to_numpy(dtype=np.int64)
    arc_ids = output["cand_arc_id"].to_numpy(dtype=np.int64)
    if (obs_idx < 0).any() or (obs_idx >= len(trip.track)).any():
        raise ValueError("Candidate observation indices must refer to rows in this trip.")

    geometry = build_arc_geometries(trip)
    geometry_index = {int(arc_id): index for index, arc_id in enumerate(geometry.arc_ids)}
    if any(int(arc_id) not in geometry_index for arc_id in np.unique(arc_ids)):
        raise ValueError("Candidate arc IDs must refer to valid projected road arcs.")
    arc_index = np.fromiter(
        (geometry_index[int(arc_id)] for arc_id in arc_ids),
        dtype=np.int64,
        count=len(arc_ids),
    )

    track = trip.track
    lon = track["lon"].to_numpy(dtype=float)
    lat = track["lat"].to_numpy(dtype=float)
    valid_coord = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (np.abs(lon) <= 180)
        & (np.abs(lat) <= 90)
    )
    if not valid_coord[obs_idx].all():
        raise ValueError("Candidates cannot refer to invalid GPS coordinates.")
    point_x = np.full(len(track), np.nan, dtype=float)
    point_y = np.full(len(track), np.nan, dtype=float)
    point_x[valid_coord], point_y[valid_coord] = lonlat_to_metres(
        lon[valid_coord], lat[valid_coord], geometry.crs
    )

    start_x = geometry.start_x_m[arc_index]
    start_y = geometry.start_y_m[arc_index]
    end_x = geometry.end_x_m[arc_index]
    end_y = geometry.end_y_m[arc_index]
    vector_x = end_x - start_x
    vector_y = end_y - start_y
    arc_length = geometry.length_m[arc_index]
    length_squared = arc_length * arc_length
    point_delta_x = point_x[obs_idx] - start_x
    point_delta_y = point_y[obs_idx] - start_y
    raw_fraction = np.divide(
        point_delta_x * vector_x + point_delta_y * vector_y,
        length_squared,
        out=np.zeros(len(output), dtype=float),
        where=length_squared > 0,
    )
    fraction = np.clip(raw_fraction, 0.0, 1.0)
    point_to_start = np.hypot(point_delta_x, point_delta_y)
    point_to_end = np.hypot(point_x[obs_idx] - end_x, point_y[obs_idx] - end_y)
    signed_cross_track = np.divide(
        vector_x * point_delta_y - vector_y * point_delta_x,
        arc_length,
        out=np.full(len(output), np.nan, dtype=float),
        where=arc_length > 0,
    )

    times = track["t"].to_numpy(dtype=float)
    time_step = np.full(len(track), np.nan, dtype=float)
    delta_x = np.full(len(track), np.nan, dtype=float)
    delta_y = np.full(len(track), np.nan, dtype=float)
    if len(track) > 1:
        time_step[1:] = np.diff(times)
        delta_x[1:] = np.diff(point_x)
        delta_y[1:] = np.diff(point_y)
    valid_motion = (
        np.isfinite(time_step)
        & (time_step > 0)
        & (time_step <= GAP_THRESHOLD_SECONDS)
        & np.isfinite(delta_x)
        & np.isfinite(delta_y)
    )
    movement = np.hypot(delta_x, delta_y)
    valid_heading = valid_motion & (movement > 0)
    gps_heading = np.full(len(track), np.nan, dtype=float)
    gps_heading[valid_heading] = np.arctan2(
        delta_y[valid_heading], delta_x[valid_heading]
    )
    gps_speed = np.full(len(track), np.nan, dtype=float)
    gps_speed[valid_motion] = movement[valid_motion] / time_step[valid_motion]

    arc_bearing = geometry.bearing_rad[arc_index].copy()
    arc_bearing[arc_length == 0] = np.nan
    candidate_heading = gps_heading[obs_idx]
    heading_delta = (arc_bearing - candidate_heading + np.pi) % (2 * np.pi) - np.pi

    outgoing, incoming = build_node_arc_adjacency(trip)
    from_nodes = geometry.from_node[arc_index]
    to_nodes = geometry.to_node[arc_index]
    ranked_groups = output.groupby("obs_idx", sort=False).cumcount().to_numpy() + 1
    output["candidate_rank"] = ranked_groups.astype(np.int64)
    output["arc_length_m"] = arc_length
    output["projection_fraction"] = fraction
    output["signed_cross_track_m"] = signed_cross_track
    output["point_to_start_m"] = point_to_start
    output["point_to_end_m"] = point_to_end
    output["arc_bearing_rad"] = arc_bearing
    output["gps_heading_rad"] = candidate_heading
    output["heading_difference_rad"] = heading_delta
    output["heading_alignment"] = np.cos(heading_delta)
    output["time_step_s"] = time_step[obs_idx]
    output["gps_speed_mps"] = gps_speed[obs_idx]
    output["is_gap_point"] = (
        np.isfinite(time_step[obs_idx])
        & (time_step[obs_idx] > GAP_THRESHOLD_SECONDS)
    )
    output["from_node_in_degree"] = np.fromiter(
        (len(incoming[int(node)]) for node in from_nodes), dtype=np.int64, count=len(output)
    )
    output["from_node_out_degree"] = np.fromiter(
        (len(outgoing[int(node)]) for node in from_nodes), dtype=np.int64, count=len(output)
    )
    output["to_node_in_degree"] = np.fromiter(
        (len(incoming[int(node)]) for node in to_nodes), dtype=np.int64, count=len(output)
    )
    output["to_node_out_degree"] = np.fromiter(
        (len(outgoing[int(node)]) for node in to_nodes), dtype=np.int64, count=len(output)
    )
    if not set(CANDIDATE_FEATURE_COLUMNS).issubset(output.columns):
        raise RuntimeError("Feature engineering did not produce the declared feature schema.")
    return output
