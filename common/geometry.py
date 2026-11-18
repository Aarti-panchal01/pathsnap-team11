"""Reusable local-metric geometry helpers for network proximity checks."""

import numpy as np
from shapely import distance, points
from shapely.geometry import LineString
from shapely.strtree import STRtree

from common.loader import Trip
from common.projection import lonlat_to_metres, pick_metric_crs


def nearest_network_distances_m(trip: Trip) -> np.ndarray:
    """Measure each GPS row to its nearest valid directed road-arc geometry."""
    track = trip.track
    gps_lon = track["lon"].to_numpy(dtype=float)
    gps_lat = track["lat"].to_numpy(dtype=float)
    valid_gps = (
        np.isfinite(gps_lon)
        & np.isfinite(gps_lat)
        & (np.abs(gps_lon) <= 180)
        & (np.abs(gps_lat) <= 90)
    )
    distances = np.full(len(track), np.nan, dtype=float)
    nodes = trip.nodes[["lon", "lat"]].to_numpy(dtype=float)
    valid_nodes = (
        np.isfinite(nodes).all(axis=1)
        & (np.abs(nodes[:, 0]) <= 180)
        & (np.abs(nodes[:, 1]) <= 90)
    )
    if not valid_gps.any() or not valid_nodes.any() or trip.arcs.empty:
        return distances

    crs = pick_metric_crs(
        np.concatenate([gps_lon[valid_gps], nodes[valid_nodes, 0]]),
        np.concatenate([gps_lat[valid_gps], nodes[valid_nodes, 1]]),
    )
    node_x, node_y = lonlat_to_metres(nodes[:, 0], nodes[:, 1], crs)
    arc_pairs = trip.arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
    valid_arcs = (
        (arc_pairs[:, 0] >= 0)
        & (arc_pairs[:, 0] < len(nodes))
        & (arc_pairs[:, 1] >= 0)
        & (arc_pairs[:, 1] < len(nodes))
    )
    arc_ids = np.flatnonzero(valid_arcs)
    endpoints = arc_pairs[arc_ids]
    valid_endpoints = valid_nodes[endpoints[:, 0]] & valid_nodes[endpoints[:, 1]]
    endpoints = endpoints[valid_endpoints]
    if not len(endpoints):
        return distances

    lines = [
        LineString([(node_x[start], node_y[start]), (node_x[end], node_y[end])])
        for start, end in endpoints
    ]
    tree = STRtree(lines)
    gps_x, gps_y = lonlat_to_metres(gps_lon[valid_gps], gps_lat[valid_gps], crs)
    gps_points = points(gps_x, gps_y)
    nearest = tree.nearest(gps_points)
    distances[valid_gps] = np.asarray(distance(gps_points, tree.geometries.take(nearest)), dtype=float)
    return distances
