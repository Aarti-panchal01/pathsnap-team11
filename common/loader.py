"""Load and validate one trip's GPS track and individual road network."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trip:
    """Keep trip tables together while preserving numeric, zero-based row IDs."""

    trip_id: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: np.ndarray | None


def list_trips(partition_dir: Path | str) -> list[Path]:
    """Return only trip directories whose names preserve the 8-digit ID format."""
    root = Path(partition_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Partition directory does not exist: {root}")
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and re.fullmatch(r"[0-9]{8}", path.name)),
        key=lambda path: path.name,
    )


def _read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read the headerless whitespace format with its declared column names."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns)


def load_trip(trip_dir: Path | str, load_route: bool = True) -> Trip:
    """Load a single trip and tolerate an absent optional reference route."""
    directory = Path(trip_dir)
    trip_id = directory.name
    if not re.fullmatch(r"[0-9]{8}", trip_id):
        raise ValueError(f"Trip directory must have an 8-digit name: {directory}")

    track = _read_table(directory / f"{trip_id}.track", ["lon", "lat", "t"])
    nodes = _read_table(directory / f"{trip_id}.nodes", ["lon", "lat"])
    arcs = _read_table(directory / f"{trip_id}.arcs", ["from_node", "to_node"])
    nodes.index.name = "node_id"
    arcs.index.name = "arc_id"

    route_path = directory / f"{trip_id}.route"
    route: np.ndarray | None = None
    if load_route and route_path.is_file():
        route_frame = _read_table(route_path, ["arc_id"])
        route = route_frame["arc_id"].to_numpy(dtype=np.int64)

    return Trip(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route=route)


def validate_trip(trip: Trip) -> dict[str, int | float]:
    """Count structural and temporal defects without altering any observations."""
    track = trip.track
    coords = track[["lon", "lat"]].to_numpy(dtype=float)
    times = track["t"].to_numpy(dtype=float)
    dt = np.diff(times)
    arcs = trip.arcs[["from_node", "to_node"]].to_numpy(dtype=float)
    node_count = len(trip.nodes)
    arc_count = len(trip.arcs)

    arc_ids_valid = np.zeros(0, dtype=bool)
    route_discontinuities = 0
    route_out_of_range = 0
    if trip.route is not None:
        route = trip.route
        arc_ids_valid = (route >= 0) & (route < arc_count)
        route_out_of_range = int((~arc_ids_valid).sum())
        valid_positions = np.flatnonzero(arc_ids_valid)
        if len(valid_positions) > 1:
            consecutive = np.diff(valid_positions) == 1
            pairs = np.flatnonzero(consecutive)
            route_arcs = trip.arcs[["from_node", "to_node"]].to_numpy(dtype=np.int64)
            route_discontinuities = int(
                sum(route_arcs[route[pos], 1] != route_arcs[route[pos + 1], 0] for pos in pairs)
            )

    return {
        "track_nan_values": int(track.isna().sum().sum()),
        "node_nan_values": int(trip.nodes.isna().sum().sum()),
        "arc_nan_values": int(trip.arcs.isna().sum().sum()),
        "gps_out_of_range_lon": int((~np.isfinite(coords[:, 0]) | (np.abs(coords[:, 0]) > 180)).sum()),
        "gps_out_of_range_lat": int((~np.isfinite(coords[:, 1]) | (np.abs(coords[:, 1]) > 90)).sum()),
        "non_increasing_timestamps": int((dt <= 0).sum()),
        "max_time_gap_s": float(np.max(dt[np.isfinite(dt)]))
        if np.isfinite(dt).any()
        else 0.0,
        "arc_node_ids_out_of_range": int(
            ((~np.isfinite(arcs)) | (arcs < 0) | (arcs >= node_count)).sum()
        ),
        "route_arc_ids_out_of_range": route_out_of_range,
        "route_discontinuities": route_discontinuities,
    }
