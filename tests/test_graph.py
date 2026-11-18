"""Check projected arc properties and directed adjacency helpers."""

import numpy as np
import pandas as pd

from common.graph import build_arc_geometries, build_digraph, build_node_arc_adjacency
from common.loader import Trip


def _small_trip() -> Trip:
    """Provide a compact directed network for deterministic helper tests."""
    return Trip(
        trip_id="00000000",
        track=pd.DataFrame({"lon": [8.58], "lat": [49.84], "t": [1.0]}),
        nodes=pd.DataFrame(
            {"lon": [8.58, 8.59, 8.60], "lat": [49.84, 49.84, 49.85]},
            index=pd.Index([0, 1, 2], name="node_id"),
        ),
        arcs=pd.DataFrame(
            {"from_node": [0, 1, 1], "to_node": [1, 2, 0]},
            index=pd.Index([0, 1, 2], name="arc_id"),
        ),
        route=None,
    )


def test_metric_arc_geometry_and_directional_adjacency() -> None:
    """Confirm source IDs, metre lengths, graph direction, and node lookups align."""
    trip = _small_trip()
    geometry = build_arc_geometries(trip)
    assert geometry.crs.to_epsg() == 32632
    assert geometry.arc_ids.tolist() == [0, 1, 2]
    assert np.all(geometry.length_m > 0)

    graph = build_digraph(trip, geometry)
    outgoing, incoming = build_node_arc_adjacency(trip)
    assert graph.has_edge(0, 1)
    assert not graph.has_edge(0, 2)
    assert graph[0][1]["arc_ids"] == [0]
    assert outgoing[1] == [1, 2]
    assert incoming[0] == [2]
