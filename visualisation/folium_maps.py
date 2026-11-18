"""Create small, layered HTML maps for selected GPS/network examples."""

from pathlib import Path

import folium
import numpy as np

from common.loader import Trip, load_trip
from common.paths import RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from common.projection import lonlat_to_metres, metres_to_lonlat, pick_metric_crs


def _valid_arc_pairs(trip: Trip) -> np.ndarray:
    """Keep only arcs whose endpoint IDs and coordinates can be drawn."""
    pairs = trip.arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
    nodes = trip.nodes[["lon", "lat"]].to_numpy(dtype=float)
    valid = (
        (pairs[:, 0] >= 0)
        & (pairs[:, 0] < len(nodes))
        & (pairs[:, 1] >= 0)
        & (pairs[:, 1] < len(nodes))
    )
    pairs = pairs[valid]
    valid_coords = np.isfinite(nodes[pairs[:, 0]]).all(axis=1) & np.isfinite(nodes[pairs[:, 1]]).all(axis=1)
    valid_coords &= (np.abs(nodes[pairs[:, 0], 0]) <= 180) & (np.abs(nodes[pairs[:, 0], 1]) <= 90)
    valid_coords &= (np.abs(nodes[pairs[:, 1], 0]) <= 180) & (np.abs(nodes[pairs[:, 1], 1]) <= 90)
    return pairs[valid_coords]


def make_trip_map(trip_id: str, partition: str) -> Path:
    """Draw GPS samples, local road arcs, and the optional reference route."""
    roots = {"train": TRAIN_DIR, "student_test": STUDENT_TEST_DIR}
    if partition not in roots:
        raise ValueError("partition must be 'train' or 'student_test'.")
    if len(trip_id) != 8 or not trip_id.isdigit():
        raise ValueError("trip_id must be an 8-digit string.")
    trip_dir = roots[partition] / trip_id
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip does not exist in {partition}: {trip_id}")
    trip = load_trip(trip_dir)

    track = trip.track[["lon", "lat"]].to_numpy(dtype=float)
    valid_track = (
        np.isfinite(track).all(axis=1)
        & (np.abs(track[:, 0]) <= 180)
        & (np.abs(track[:, 1]) <= 90)
    )
    if not valid_track.any():
        raise ValueError(f"Trip {trip_id} has no valid GPS coordinates to map.")
    lon, lat = track[valid_track, 0], track[valid_track, 1]
    crs = pick_metric_crs(lon, lat)
    gps_x, gps_y = lonlat_to_metres(lon, lat, crs)
    padding = max(300.0, 0.05 * max(float(np.ptp(gps_x)), float(np.ptp(gps_y))))
    bounds = (
        float(gps_x.min() - padding),
        float(gps_y.min() - padding),
        float(gps_x.max() + padding),
        float(gps_y.max() + padding),
    )

    map_view = folium.Map(location=[float(lat.mean()), float(lon.mean())], control_scale=True, zoom_start=14)
    network_layer = folium.FeatureGroup(name="Nearby road-network arcs", show=True)
    route_layer = folium.FeatureGroup(name="Reference route", show=True)
    gps_layer = folium.FeatureGroup(name="GPS trajectory and points", show=True)

    nodes = trip.nodes[["lon", "lat"]].to_numpy(dtype=float)
    node_x, node_y = lonlat_to_metres(nodes[:, 0], nodes[:, 1], crs)
    arc_pairs = _valid_arc_pairs(trip)
    min_x, min_y, max_x, max_y = bounds
    for start, end in arc_pairs:
        x1, y1 = node_x[start], node_y[start]
        x2, y2 = node_x[end], node_y[end]
        if max(x1, x2) < min_x or min(x1, x2) > max_x or max(y1, y2) < min_y or min(y1, y2) > max_y:
            continue
        arc_lon, arc_lat = metres_to_lonlat([x1, x2], [y1, y2], crs)
        folium.PolyLine(
            list(zip(arc_lat.tolist(), arc_lon.tolist())),
            color="#707d8a",
            weight=1,
            opacity=0.45,
        ).add_to(network_layer)

    if trip.route is not None:
        route_points = []
        for arc_id in trip.route:
            if arc_id < 0 or arc_id >= len(trip.arcs):
                if len(route_points) > 1:
                    folium.PolyLine(route_points, color="#c026d3", weight=4, opacity=0.9).add_to(route_layer)
                route_points = []
                continue
            start, end = trip.arcs.iloc[int(arc_id)].to_numpy(dtype=int)
            if start < 0 or end < 0 or start >= len(nodes) or end >= len(nodes):
                if len(route_points) > 1:
                    folium.PolyLine(route_points, color="#c026d3", weight=4, opacity=0.9).add_to(route_layer)
                route_points = []
                continue
            if not np.isfinite(nodes[[start, end]]).all():
                if len(route_points) > 1:
                    folium.PolyLine(route_points, color="#c026d3", weight=4, opacity=0.9).add_to(route_layer)
                route_points = []
                continue
            route_lon, route_lat = metres_to_lonlat([node_x[start], node_x[end]], [node_y[start], node_y[end]], crs)
            start_point = (float(route_lat[0]), float(route_lon[0]))
            end_point = (float(route_lat[1]), float(route_lon[1]))
            if route_points and route_points[-1] != start_point:
                if len(route_points) > 1:
                    folium.PolyLine(route_points, color="#c026d3", weight=4, opacity=0.9).add_to(route_layer)
                route_points = []
            if not route_points:
                route_points.append(start_point)
            route_points.append(end_point)
        if len(route_points) > 1:
            folium.PolyLine(route_points, color="#c026d3", weight=4, opacity=0.9).add_to(route_layer)

    track_times = trip.track["t"].to_numpy(dtype=float)
    run = []
    for row_index in range(len(track)):
        if not valid_track[row_index]:
            if len(run) > 1:
                folium.PolyLine(run, color="#2563eb", weight=3, opacity=0.8).add_to(gps_layer)
            run = []
            continue
        point = (float(track[row_index, 1]), float(track[row_index, 0]))
        if run and (not np.isfinite(track_times[row_index - 1 : row_index + 1]).all()
                    or track_times[row_index] - track_times[row_index - 1] <= 0
                    or track_times[row_index] - track_times[row_index - 1] > 5):
            if len(run) > 1:
                folium.PolyLine(run, color="#2563eb", weight=3, opacity=0.8).add_to(gps_layer)
            run = []
        run.append(point)
    if len(run) > 1:
        folium.PolyLine(run, color="#2563eb", weight=3, opacity=0.8).add_to(gps_layer)
    for point_number in np.flatnonzero(valid_track):
        point_lat, point_lon = track[point_number, 1], track[point_number, 0]
        folium.CircleMarker(
            location=[float(point_lat), float(point_lon)],
            radius=2,
            color="#16a34a",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"GPS row {point_number}",
        ).add_to(gps_layer)

    network_layer.add_to(map_view)
    route_layer.add_to(map_view)
    gps_layer.add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    legend = """
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;
                padding:8px 12px;border:1px solid #666;border-radius:4px;font-size:13px">
      <b>Layers</b><br>
      <span style="color:#16a34a">●</span> GPS points /
      <span style="color:#2563eb">━</span> GPS trajectory<br>
      <span style="color:#707d8a">━</span> Road arcs /
      <span style="color:#c026d3">━</span> Reference route
    </div>
    """
    map_view.get_root().html.add_child(folium.Element(legend))
    view_lon, view_lat = metres_to_lonlat(
        [min_x, max_x, max_x, min_x], [min_y, min_y, max_y, max_y], crs
    )
    map_view.fit_bounds(
        [
            [float(view_lat.min()), float(view_lon.min())],
            [float(view_lat.max()), float(view_lon.max())],
        ]
    )
    output_dir = RESULTS_DIR / "maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{partition}_{trip_id}.html"
    map_view.save(output_path)
    return output_path


def main() -> None:
    """Generate four real-data maps selected to represent typical and flagged trips."""
    selected = (
        ("00000095", "train", "typical train trip"),
        ("00000090", "train", "train trip with very large GPS gaps"),
        ("00000069", "train", "train trip with discontinuous reference route"),
        ("00000007", "student_test", "student-test trip"),
    )
    for trip_id, partition, label in selected:
        print(f"{label}: {make_trip_map(trip_id, partition)}")


if __name__ == "__main__":
    main()
