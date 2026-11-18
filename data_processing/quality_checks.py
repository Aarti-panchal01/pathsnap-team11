"""Measure GPS integrity, timing, extents, and trajectory lengths per trip."""

from pathlib import Path

import numpy as np
import pandas as pd

from common.loader import list_trips, load_trip, validate_trip
from common.paths import RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from common.projection import lonlat_to_metres, pick_metric_crs


def inspect_trip(partition: str, trip_dir: Path) -> dict[str, object]:
    """Summarize one trip without removing or repairing any source row."""
    trip = load_trip(trip_dir)
    track = trip.track
    lon = track["lon"].to_numpy(dtype=float)
    lat = track["lat"].to_numpy(dtype=float)
    time = track["t"].to_numpy(dtype=float)
    finite_coords = np.isfinite(lon) & np.isfinite(lat)
    in_range = finite_coords & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
    invalid_rows = ~in_range
    dt = np.diff(time)
    finite_dt = dt[np.isfinite(dt)]
    positive_dt = finite_dt[finite_dt > 0]
    gap_dt = positive_dt[positive_dt > 5]
    valid_lon, valid_lat = lon[in_range], lat[in_range]

    result: dict[str, object] = {
        "partition": partition,
        "trip_id": trip.trip_id,
        "gps_rows": len(track),
        "missing_track_values": int(track.isna().sum().sum()),
        "missing_gps_rows": int((~finite_coords).sum()),
        "missing_timestamp_rows": int((~np.isfinite(time)).sum()),
        "invalid_gps_rows": int(invalid_rows.sum()),
        "invalid_lon_rows": int((~np.isfinite(lon) | (np.abs(lon) > 180)).sum()),
        "invalid_lat_rows": int((~np.isfinite(lat) | (np.abs(lat) > 90)).sum()),
        "swapped_lon_lat_suspect_rows": int(
            ((np.abs(lat) > 90) & (np.abs(lat) <= 180) & (np.abs(lon) <= 90)).sum()
        ),
        "timestamp_non_increasing_steps": int((np.isfinite(dt) & (dt <= 0)).sum()),
        "positive_interval_count": len(positive_dt),
        "exact_1s_steps": int((positive_dt == 1).sum()),
        "share_exact_1s_steps": float((positive_dt == 1).mean()) if len(positive_dt) else np.nan,
        "gap_over_5s_count": len(gap_dt),
        "largest_gap_s": float(positive_dt.max()) if len(positive_dt) else np.nan,
        "gps_bbox_min_lon": float(valid_lon.min()) if len(valid_lon) else np.nan,
        "gps_bbox_min_lat": float(valid_lat.min()) if len(valid_lat) else np.nan,
        "gps_bbox_max_lon": float(valid_lon.max()) if len(valid_lon) else np.nan,
        "gps_bbox_max_lat": float(valid_lat.max()) if len(valid_lat) else np.nan,
    }

    if len(valid_lon):
        crs = pick_metric_crs(valid_lon, valid_lat)
        x, y = lonlat_to_metres(lon, lat, crs)
        result.update(
            {
                "metric_crs": crs.to_string(),
                "bbox_min_x_m": float(x[in_range].min()),
                "bbox_min_y_m": float(y[in_range].min()),
                "bbox_max_x_m": float(x[in_range].max()),
                "bbox_max_y_m": float(y[in_range].max()),
            }
        )
        adjacent_valid = in_range[:-1] & in_range[1:]
        dx, dy = np.diff(x), np.diff(y)
        result["trajectory_length_m"] = float(np.hypot(dx[adjacent_valid], dy[adjacent_valid]).sum())
    else:
        result.update(
            {
                "metric_crs": "",
                "bbox_min_x_m": np.nan,
                "bbox_min_y_m": np.nan,
                "bbox_max_x_m": np.nan,
                "bbox_max_y_m": np.nan,
                "trajectory_length_m": np.nan,
            }
        )

    checks = validate_trip(trip)
    result["route_discontinuities"] = checks["route_discontinuities"]
    result["route_arc_ids_out_of_range"] = checks["route_arc_ids_out_of_range"]
    result["arc_node_ids_out_of_range"] = checks["arc_node_ids_out_of_range"]
    reasons = []
    if result["missing_gps_rows"]:
        reasons.append("missing GPS coordinates")
    if result["invalid_gps_rows"] > result["missing_gps_rows"]:
        reasons.append("out-of-range GPS coordinates")
    if result["timestamp_non_increasing_steps"]:
        reasons.append("non-increasing timestamps")
    if result["missing_timestamp_rows"]:
        reasons.append("missing or non-finite timestamps")
    if result["gap_over_5s_count"]:
        reasons.append("GPS gap over 5 seconds")
    if result["swapped_lon_lat_suspect_rows"]:
        reasons.append("possible swapped coordinates")
    if result["route_discontinuities"] or result["route_arc_ids_out_of_range"]:
        reasons.append("broken reference route")
    result["anomaly_reason"] = "; ".join(reasons)
    return result


def _partition_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Aggregate row-level quality findings while keeping local metric extents separate."""
    intervals = frame["positive_interval_count"].sum()
    exact = frame["exact_1s_steps"].sum()
    return {
        "partition": frame["partition"].iloc[0],
        "trip_count": len(frame),
        "gps_rows": int(frame["gps_rows"].sum()),
        "missing_track_values": int(frame["missing_track_values"].sum()),
        "missing_gps_rows": int(frame["missing_gps_rows"].sum()),
        "missing_timestamp_rows": int(frame["missing_timestamp_rows"].sum()),
        "invalid_gps_rows": int(frame["invalid_gps_rows"].sum()),
        "swapped_lon_lat_suspect_rows": int(frame["swapped_lon_lat_suspect_rows"].sum()),
        "arc_node_ids_out_of_range": int(frame["arc_node_ids_out_of_range"].sum()),
        "route_arc_ids_out_of_range": int(frame["route_arc_ids_out_of_range"].sum()),
        "timestamp_non_increasing_steps": int(frame["timestamp_non_increasing_steps"].sum()),
        "positive_interval_count": int(intervals),
        "exact_1s_steps": int(exact),
        "share_exact_1s_steps": float(exact / intervals) if intervals else np.nan,
        "gap_over_5s_count": int(frame["gap_over_5s_count"].sum()),
        "largest_gap_s": float(frame["largest_gap_s"].max()),
        "anomalous_trip_count": int(frame["anomaly_reason"].ne("").sum()),
        "extent_min_lon": float(frame["gps_bbox_min_lon"].min()),
        "extent_min_lat": float(frame["gps_bbox_min_lat"].min()),
        "extent_max_lon": float(frame["gps_bbox_max_lon"].max()),
        "extent_max_lat": float(frame["gps_bbox_max_lat"].max()),
        "metric_crs_count": int(frame["metric_crs"].nunique()),
    }


def main() -> None:
    """Write per-trip quality, partition summaries, and an auditable anomaly list."""
    output_dir = RESULTS_DIR / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_trip = []
    for partition, root in (("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)):
        per_trip.extend(inspect_trip(partition, directory) for directory in list_trips(root))
    frame = pd.DataFrame(per_trip)
    summary = pd.DataFrame(
        [_partition_summary(frame[frame["partition"] == part]) for part in ("train", "student_test")]
    )
    frame.to_csv(output_dir / "quality_per_trip.csv", index=False)
    summary.to_csv(output_dir / "quality_summary.csv", index=False)
    frame.loc[frame["anomaly_reason"].ne(""), ["partition", "trip_id", "anomaly_reason"]].to_csv(
        output_dir / "anomalous_trips.csv", index=False
    )
    broken_routes = frame.loc[
        (frame["route_discontinuities"] > 0) | (frame["route_arc_ids_out_of_range"] > 0),
        ["partition", "trip_id", "route_arc_ids_out_of_range", "route_discontinuities"],
    ]
    broken_routes.to_csv(output_dir / "broken_reference_routes.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Anomalous trips: {int(frame['anomaly_reason'].ne('').sum())}")


if __name__ == "__main__":
    main()
