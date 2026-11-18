"""Build reusable metric arc geometry and directed-network lookups per trip."""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from pyproj import CRS
from shapely import linestrings

from common.loader import Trip
from common.projection import lonlat_to_metres, pick_metric_crs


@dataclass(frozen=True)
class ArcGeometry:
    """Keep projected arc properties aligned with their source arc IDs."""

    crs: CRS
    arc_ids: np.ndarray
    from_node: np.ndarray
    to_node: np.ndarray
    start_x_m: np.ndarray
    start_y_m: np.ndarray
    end_x_m: np.ndarray
    end_y_m: np.ndarray
    length_m: np.ndarray
    bearing_rad: np.ndarray

    def as_lines(self) -> np.ndarray:
        """Create geometries in the same order as arc_ids for spatial indexing."""
        coordinates = np.stack(
            [
                np.column_stack([self.start_x_m, self.start_y_m]),
                np.column_stack([self.end_x_m, self.end_y_m]),
            ],
            axis=1,
        )
        return linestrings(coordinates)


def build_arc_geometries(trip: Trip) -> ArcGeometry:
    """Project valid directed arcs so every downstream distance uses metres."""
    nodes = trip.nodes[["lon", "lat"]].to_numpy(dtype=float)
    valid_nodes = (
        np.isfinite(nodes).all(axis=1)
        & (np.abs(nodes[:, 0]) <= 180)
        & (np.abs(nodes[:, 1]) <= 90)
    )
    if not valid_nodes.any():
        raise ValueError(f"Trip {trip.trip_id} has no valid network nodes.")

    crs = pick_metric_crs(nodes[valid_nodes, 0], nodes[valid_nodes, 1])
    node_x = np.full(len(nodes), np.nan, dtype=float)
    node_y = np.full(len(nodes), np.nan, dtype=float)
    node_x[valid_nodes], node_y[valid_nodes] = lonlat_to_metres(
        nodes[valid_nodes, 0], nodes[valid_nodes, 1], crs
    )

    pairs = trip.arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
    valid_arcs = (
        (pairs[:, 0] >= 0)
        & (pairs[:, 0] < len(nodes))
        & (pairs[:, 1] >= 0)
        & (pairs[:, 1] < len(nodes))
    )
    arc_ids = np.flatnonzero(valid_arcs)
    pairs = pairs[valid_arcs]
    endpoints_valid = valid_nodes[pairs[:, 0]] & valid_nodes[pairs[:, 1]]
    arc_ids = arc_ids[endpoints_valid]
    pairs = pairs[endpoints_valid]

    start_x, start_y = node_x[pairs[:, 0]], node_y[pairs[:, 0]]
    end_x, end_y = node_x[pairs[:, 1]], node_y[pairs[:, 1]]
    delta_x, delta_y = end_x - start_x, end_y - start_y
    return ArcGeometry(
        crs=crs,
        arc_ids=arc_ids.astype(np.int64),
        from_node=pairs[:, 0].astype(np.int64),
        to_node=pairs[:, 1].astype(np.int64),
        start_x_m=start_x,
        start_y_m=start_y,
        end_x_m=end_x,
        end_y_m=end_y,
        length_m=np.hypot(delta_x, delta_y),
        bearing_rad=np.arctan2(delta_y, delta_x),
    )


def build_digraph(trip: Trip, arc_geometry: ArcGeometry | None = None) -> nx.DiGraph:
    """Build a directed graph while retaining parallel source arc IDs per edge."""
    geometry = arc_geometry or build_arc_geometries(trip)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(trip.nodes)))
    for index, arc_id in enumerate(geometry.arc_ids):
        source = int(geometry.from_node[index])
        target = int(geometry.to_node[index])
        if graph.has_edge(source, target):
            graph[source][target]["arc_ids"].append(int(arc_id))
        else:
            graph.add_edge(
                source,
                target,
                arc_ids=[int(arc_id)],
                length_m=float(geometry.length_m[index]),
                bearing_rad=float(geometry.bearing_rad[index]),
            )
    return graph


def build_node_arc_adjacency(
    trip: Trip,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Index outgoing and incoming arcs so candidate features can inspect endpoints."""
    outgoing = {node_id: [] for node_id in range(len(trip.nodes))}
    incoming = {node_id: [] for node_id in range(len(trip.nodes))}
    pairs = trip.arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
    for arc_id, (source, target) in enumerate(pairs):
        if 0 <= source < len(trip.nodes) and 0 <= target < len(trip.nodes):
            outgoing[int(source)].append(arc_id)
            incoming[int(target)].append(arc_id)
    return outgoing, incoming
