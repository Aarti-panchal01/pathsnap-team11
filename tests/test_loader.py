"""Regression checks for the real-data loader and local metric projection."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common.loader import Trip, load_trip, list_trips, validate_trip
from common.paths import STUDENT_TEST_DIR, TRAIN_DIR
from common.projection import lonlat_to_metres, metres_to_lonlat, pick_metric_crs


def test_real_partition_sizes_and_columns() -> None:
    """Confirm real partitions and loaded tables match the documented schema."""
    train_trips = list_trips(TRAIN_DIR)
    student_test_trips = list_trips(STUDENT_TEST_DIR)
    assert len(train_trips) == 80
    assert len(student_test_trips) == 10
    trip = load_trip(train_trips[0])
    assert list(trip.track.columns) == ["lon", "lat", "t"]
    assert list(trip.nodes.columns) == ["lon", "lat"]
    assert list(trip.arcs.columns) == ["from_node", "to_node"]
    assert trip.nodes.index.name == "node_id"
    assert trip.arcs.index.name == "arc_id"


def test_trip_id_keeps_leading_zeroes() -> None:
    """Keep folder identifiers as strings instead of losing leading zeroes."""
    trip = load_trip(list_trips(TRAIN_DIR)[0])
    assert isinstance(trip.trip_id, str)
    assert trip.trip_id == "00000001"


def test_route_can_be_absent(tmp_path: Path) -> None:
    """Allow route-independent preprocessing and inference."""
    source = list_trips(TRAIN_DIR)[0]
    target = tmp_path / source.name
    shutil.copytree(source, target)
    (target / f"{source.name}.route").unlink()
    trip = load_trip(target)
    assert trip.route is None


def test_projection_round_trip() -> None:
    """Ensure projection helpers preserve nearby geographic coordinates."""
    lon = np.array([8.58, 8.60])
    lat = np.array([49.84, 49.86])
    crs = pick_metric_crs(lon, lat)
    x, y = lonlat_to_metres(lon, lat, crs)
    restored_lon, restored_lat = metres_to_lonlat(x, y, crs)
    assert crs.to_epsg() == 32632
    np.testing.assert_allclose(restored_lon, lon, atol=1e-8)
    np.testing.assert_allclose(restored_lat, lat, atol=1e-8)


def test_validate_trip_counts_invalid_values_and_broken_route() -> None:
    """Count coordinate, timing, endpoint, and route defects without repair."""
    trip = Trip(
        trip_id="00000000",
        track=pd.DataFrame(
            {"lon": [8.5, 181.0, 8.5], "lat": [49.9, 49.9, 49.9], "t": [1.0, 1.0, 7.0]}
        ),
        nodes=pd.DataFrame(
            {"lon": [8.5, 8.6], "lat": [49.9, 49.9]},
            index=pd.Index([0, 1], name="node_id"),
        ),
        arcs=pd.DataFrame(
            {"from_node": [0, -1], "to_node": [1, 0]},
            index=pd.Index([0, 1], name="arc_id"),
        ),
        route=np.array([0, 0, 5]),
    )
    result = validate_trip(trip)
    assert result["gps_out_of_range_lon"] == 1
    assert result["non_increasing_timestamps"] == 1
    assert result["max_time_gap_s"] == 6
    assert result["arc_node_ids_out_of_range"] == 1
    assert result["route_arc_ids_out_of_range"] == 1
    assert result["route_discontinuities"] == 1
