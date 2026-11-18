"""Summarize each trip's road graph and GPS distance to its supplied route."""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely import distance, points
from shapely.geometry import LineString, MultiLineString

from common.loader import Trip, list_trips, load_trip, validate_trip
from common.paths import PROCESSED_DIR, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from common.projection import lonlat_to_metres, pick_metric_crs


def _route_geometry(trip: Trip, crs: CRS) -> tuple[list[LineString], np.ndarray]:
    """Build route segments from valid arc references without joining broken hops."""
    if trip.route is None:
        return [], np.empty(0, dtype=np.int64)
    route = trip.route
    valid_arc = (route >= 0) & (route < len(trip.arcs))
    route_positions = np.flatnonzero(valid_arc)
    arc_pairs = trip.arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
    nodes = trip.nodes[["lon", "lat"]].to_numpy(dtype=float)
    route_arc_ids = route[route_positions]
    endpoints = arc_pairs[route_arc_ids]
    valid_endpoints = (
        (endpoints[:, 0] >= 0)
        & (endpoints[:, 0] < len(nodes))
        & (endpoints[:, 1] >= 0)
        & (endpoints[:, 1] < len(nodes))
    )
    route_positions = route_positions[valid_endpoints]
    route_arc_ids = route[route_positions]
    endpoints = arc_pairs[route_arc_ids]
    finite_endpoints = np.isfinite(nodes[endpoints[:, 0]]).all(axis=1) & np.isfinite(
        nodes[endpoints[:, 1]]
    ).all(axis=1)
    route_arc_ids = route_arc_ids[finite_endpoints]
    if not len(route_arc_ids):
        return [], route_arc_ids
    node_lon = nodes[:, 0]
    node_lat = nodes[:, 1]
    x, y = lonlat_to_metres(node_lon, node_lat, crs)
    lines = [
        LineString(
            [
                (x[arc_pairs[arc_id, 0]], y[arc_pairs[arc_id, 0]]),
                (x[arc_pairs[arc_id, 1]], y[arc_pairs[arc_id, 1]]),
            ]
        )
        for arc_id in route_arc_ids
    ]
    return lines, route_arc_ids


def inspect_network(partition: str, trip_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    """Measure directed connectivity, degree, route integrity, and route distances."""
    trip = load_trip(trip_dir)
    arc_pairs = trip.arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
    node_count = len(trip.nodes)
    valid_arcs = (
        (arc_pairs[:, 0] >= 0)
        & (arc_pairs[:, 0] < node_count)
        & (arc_pairs[:, 1] >= 0)
        & (arc_pairs[:, 1] < node_count)
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(arc_pairs[valid_arcs])
    in_degrees = np.fromiter((degree for _, degree in graph.in_degree()), dtype=np.int64, count=node_count)
    out_degrees = np.fromiter((degree for _, degree in graph.out_degree()), dtype=np.int64, count=node_count)
    pairs = set(map(tuple, arc_pairs[valid_arcs]))
    reverse_count = sum((to_node, from_node) in pairs for from_node, to_node in arc_pairs[valid_arcs])

    track = trip.track
    lon = track["lon"].to_numpy(dtype=float)
    lat = track["lat"].to_numpy(dtype=float)
    valid_gps = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
    valid_lon, valid_lat = lon[valid_gps], lat[valid_gps]
    crs = pick_metric_crs(valid_lon, valid_lat) if len(valid_lon) else None
    route_lines, route_arc_ids = _route_geometry(trip, crs) if crs is not None else ([], np.empty(0, dtype=int))
    route_distances = np.empty(0, dtype=float)
    if route_lines and len(valid_lon):
        gps_x, gps_y = lonlat_to_metres(valid_lon, valid_lat, crs)
        route_geom = MultiLineString(route_lines)
        route_distances = np.asarray(distance(points(gps_x, gps_y), route_geom), dtype=float)

    result: dict[str, object] = {
        "partition": partition,
        "trip_id": trip.trip_id,
        "node_count": node_count,
        "arc_count": len(trip.arcs),
        "invalid_arc_count": int((~valid_arcs).sum()),
        "weak_component_count": nx.number_weakly_connected_components(graph),
        "strong_component_count": nx.number_strongly_connected_components(graph),
        "isolated_node_count": int(((in_degrees + out_degrees) == 0).sum()),
        "in_degree_mean": float(in_degrees.mean()) if node_count else np.nan,
        "in_degree_median": float(np.median(in_degrees)) if node_count else np.nan,
        "in_degree_p95": float(np.percentile(in_degrees, 95)) if node_count else np.nan,
        "in_degree_min": int(in_degrees.min()) if node_count else 0,
        "in_degree_max": int(in_degrees.max()) if node_count else 0,
        "out_degree_mean": float(out_degrees.mean()) if node_count else np.nan,
        "out_degree_median": float(np.median(out_degrees)) if node_count else np.nan,
        "out_degree_p95": float(np.percentile(out_degrees, 95)) if node_count else np.nan,
        "out_degree_min": int(out_degrees.min()) if node_count else 0,
        "out_degree_max": int(out_degrees.max()) if node_count else 0,
        "reverse_twin_arc_share": float(reverse_count / valid_arcs.sum()) if valid_arcs.any() else np.nan,
        "route_arc_count": len(trip.route) if trip.route is not None else 0,
        "route_valid_arc_count": len(route_arc_ids),
        "route_length_m": float(sum(line.length for line in route_lines)),
        "route_discontinuities": validate_trip(trip)["route_discontinuities"],
        "gps_to_route_distance_count": len(route_distances),
        "gps_to_route_distance_mean_m": float(route_distances.mean()) if len(route_distances) else np.nan,
        "gps_to_route_distance_median_m": float(np.median(route_distances)) if len(route_distances) else np.nan,
        "gps_to_route_distance_p90_m": float(np.percentile(route_distances, 90)) if len(route_distances) else np.nan,
        "gps_to_route_distance_p95_m": float(np.percentile(route_distances, 95)) if len(route_distances) else np.nan,
        "gps_to_route_distance_p99_m": float(np.percentile(route_distances, 99)) if len(route_distances) else np.nan,
        "gps_to_route_distance_max_m": float(route_distances.max()) if len(route_distances) else np.nan,
    }
    distance_rows = pd.DataFrame(
        {
            "partition": partition,
            "trip_id": trip.trip_id,
            "gps_row_index": np.flatnonzero(valid_gps)[: len(route_distances)],
            "distance_to_route_m": route_distances,
        }
    )
    return result, distance_rows


def _partition_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Aggregate graph and route measures without merging different trips' CRS units."""
    return {
        "partition": frame["partition"].iloc[0],
        "trip_count": len(frame),
        "total_nodes": int(frame["node_count"].sum()),
        "total_arcs": int(frame["arc_count"].sum()),
        "total_reference_route_arcs": int(frame["route_arc_count"].sum()),
        "mean_reference_route_arcs": float(frame["route_arc_count"].mean()),
        "median_reference_route_length_m": float(frame["route_length_m"].median()),
        "mean_nodes_per_trip": float(frame["node_count"].mean()),
        "mean_arcs_per_trip": float(frame["arc_count"].mean()),
        "mean_weak_components_per_trip": float(frame["weak_component_count"].mean()),
        "median_weak_components_per_trip": float(frame["weak_component_count"].median()),
        "mean_strong_components_per_trip": float(frame["strong_component_count"].mean()),
        "median_strong_components_per_trip": float(frame["strong_component_count"].median()),
        "isolated_nodes": int(frame["isolated_node_count"].sum()),
        "trips_with_discontinuous_route": int((frame["route_discontinuities"] > 0).sum()),
        "total_route_discontinuities": int(frame["route_discontinuities"].sum()),
        "mean_reverse_twin_arc_share": float(frame["reverse_twin_arc_share"].mean()),
        "gps_to_route_distance_count": int(frame["gps_to_route_distance_count"].sum()),
        "gps_to_route_distance_median_m": float(frame["gps_to_route_distance_median_m"].median()),
        "gps_to_route_distance_p95_m": float(frame["gps_to_route_distance_p95_m"].median()),
    }


def main() -> None:
    """Write graph statistics and a partition summary to results/eda."""
    output_dir = RESULTS_DIR / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    distance_frames = []
    for partition, root in (("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)):
        for directory in list_trips(root):
            row, distances = inspect_network(partition, directory)
            rows.append(row)
            distance_frames.append(distances)
    frame = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [_partition_summary(frame[frame["partition"] == part]) for part in ("train", "student_test")]
    )
    frame.to_csv(output_dir / "network_per_trip.csv", index=False)
    summary.to_csv(output_dir / "network_summary.csv", index=False)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(distance_frames, ignore_index=True).to_csv(
        PROCESSED_DIR / "gps_to_route_distances.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
