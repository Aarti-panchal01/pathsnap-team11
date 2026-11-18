"""Plot real inventory, quality, network, and route-distance findings."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from common.loader import load_trip
from common.paths import PROCESSED_DIR, RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR
from common.projection import lonlat_to_metres, pick_metric_crs


def _read_eda_table(name: str) -> pd.DataFrame:
    """Load a generated EDA table and fail clearly if an earlier step was skipped."""
    path = RESULTS_DIR / "eda" / name
    if not path.is_file():
        raise FileNotFoundError(f"Run the corresponding data-preparation step first: {path}")
    return pd.read_csv(path)


def _save(figure: Figure, name: str) -> None:
    """Save a plot consistently and close it to release memory."""
    output = RESULTS_DIR / "eda" / name
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)


def _plot_gps_rows() -> None:
    """Compare GPS sample counts across train and student-test trips."""
    trips = pd.concat(
        [_read_eda_table(f"inventory_{part}.csv") for part in ("train", "student_test")],
        ignore_index=True,
    )
    figure, axis = plt.subplots(figsize=(12, 4))
    for partition, marker in (("train", "o"), ("student_test", "s")):
        values = trips[trips["partition"] == partition]["gps_rows"].sort_values().to_numpy()
        axis.plot(np.arange(len(values)), values, marker=marker, linestyle="none", label=partition)
    axis.set(title="GPS rows per trip", xlabel="Trip rank within partition", ylabel="GPS rows")
    axis.legend()
    _save(figure, "gps_rows_per_trip.png")


def _plot_sampling_and_gaps() -> None:
    """Show positive sampling steps and large gaps from preserved processed rows."""
    intervals = []
    gaps = []
    for partition in ("train", "student_test"):
        root = PROCESSED_DIR / "preprocess" / partition
        for path in sorted(root.glob("*.csv")):
            track = pd.read_csv(path, usecols=["time_step_s"])
            values = track["time_step_s"].to_numpy(dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            intervals.extend(values[values <= 5].tolist())
            gaps.extend(values[values > 5].tolist())

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.hist(intervals, bins=50, color="#2563eb", alpha=0.85)
    axis.axvline(1, color="#b91c1c", linestyle="--", label="1 second")
    axis.set(title="Positive GPS sampling intervals (0–5 seconds)", xlabel="Interval (seconds)", ylabel="Rows")
    axis.legend()
    _save(figure, "sampling_interval_histogram.png")
    figure, axis = plt.subplots(figsize=(8, 4))
    if gaps:
        bins = np.geomspace(min(gaps), max(gaps), 35) if min(gaps) < max(gaps) else 10
        axis.hist(gaps, bins=bins, color="#ea580c", alpha=0.85)
        axis.set_xscale("log")
    axis.set(title="GPS gaps larger than 5 seconds", xlabel="Gap size (seconds, log scale)", ylabel="Rows")
    _save(figure, "gap_sizes.png")


def _plot_extents_and_lengths() -> None:
    """Relate each trip's local metric bounding-box span to its traveled length."""
    frame = _read_eda_table("quality_per_trip.csv")
    frame["bbox_diagonal_m"] = np.hypot(
        frame["bbox_max_x_m"] - frame["bbox_min_x_m"],
        frame["bbox_max_y_m"] - frame["bbox_min_y_m"],
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    for partition, color in (("train", "#2563eb"), ("student_test", "#ea580c")):
        data = frame[frame["partition"] == partition]
        axis.scatter(data["bbox_diagonal_m"], data["trajectory_length_m"], s=25, alpha=0.75, label=partition, color=color)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(title="Trip extent and trajectory length", xlabel="GPS bounding-box diagonal (m)", ylabel="Trajectory length (m)")
    axis.legend()
    _save(figure, "trip_extents_lengths.png")


def _plot_network_size() -> None:
    """Compare nodes and directed arcs for each independent trip network."""
    frame = _read_eda_table("network_per_trip.csv")
    figure, axis = plt.subplots(figsize=(7, 5))
    for partition, color in (("train", "#2563eb"), ("student_test", "#ea580c")):
        data = frame[frame["partition"] == partition]
        axis.scatter(data["node_count"], data["arc_count"], s=25, alpha=0.75, label=partition, color=color)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(title="Road-network size per trip", xlabel="Nodes", ylabel="Directed arcs")
    axis.legend()
    _save(figure, "network_size_per_trip.png")


def _plot_missing_values() -> None:
    """Summarize observed missing and invalid GPS rows by partition."""
    frame = _read_eda_table("quality_summary.csv").set_index("partition")
    figure, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(frame))
    width = 0.35
    axis.bar(x - width / 2, frame["missing_gps_rows"], width, label="Missing coordinate rows")
    axis.bar(x + width / 2, frame["invalid_gps_rows"], width, label="Invalid coordinate rows")
    axis.set_xticks(x, frame.index)
    axis.set(title="GPS coordinate quality counts", ylabel="Rows")
    axis.legend()
    _save(figure, "missing_value_summary.png")


def _plot_route_distances() -> None:
    """Plot every available GPS-to-reference-route distance in projected metres."""
    path = PROCESSED_DIR / "gps_to_route_distances.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run python -m data_processing.network_stats first: {path}")
    frame = pd.read_csv(path, usecols=["partition", "distance_to_route_m"])
    figure, axis = plt.subplots(figsize=(8, 4))
    for partition, color in (("train", "#2563eb"), ("student_test", "#ea580c")):
        values = frame.loc[frame["partition"] == partition, "distance_to_route_m"].dropna()
        axis.hist(values, bins=70, alpha=0.55, density=True, label=partition, color=color)
    axis.set(title="GPS point distance to supplied reference-route geometry", xlabel="Distance (m)", ylabel="Density")
    axis.legend()
    _save(figure, "gps_to_route_distance_histogram.png")


def _plot_trajectories() -> None:
    """Plot selected trips in their own local metric CRS so distant trips are not overlaid."""
    selected = (
        ("00000095", "train", TRAIN_DIR, "Typical train"),
        ("00000090", "train", TRAIN_DIR, "Train with large GPS gaps"),
        ("00000007", "student_test", STUDENT_TEST_DIR, "Student-test"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (trip_id, partition, root, label) in zip(axes, selected):
        trip = load_trip(root / trip_id)
        lon = trip.track["lon"].to_numpy(dtype=float)
        lat = trip.track["lat"].to_numpy(dtype=float)
        valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
        crs = pick_metric_crs(lon[valid], lat[valid])
        x, y = lonlat_to_metres(lon[valid], lat[valid], crs)
        times = trip.track["t"].to_numpy(dtype=float)
        valid_rows = np.flatnonzero(valid)
        run_x, run_y = [], []
        for index, row_index in enumerate(valid_rows):
            if index and (
                not np.isfinite(times[row_index - 1 : row_index + 1]).all()
                or times[row_index] - times[row_index - 1] <= 0
                or times[row_index] - times[row_index - 1] > 5
            ):
                if len(run_x) > 1:
                    axis.plot(run_x, run_y, color="#2563eb", linewidth=1)
                run_x, run_y = [], []
            run_x.append(float(x[index]))
            run_y.append(float(y[index]))
        if len(run_x) > 1:
            axis.plot(run_x, run_y, color="#2563eb", linewidth=1)
        axis.scatter(x, y, s=5, color="#16a34a")
        axis.set(title=f"{label}\n{partition} {trip_id}", xlabel="Easting (m)", ylabel="Northing (m)")
        axis.set_aspect("equal", adjustable="datalim")
    _save(figure, "sample_trajectories.png")


def main() -> None:
    """Generate the requested real-data EDA figures from saved outputs."""
    _read_eda_table("inventory_summary.csv")
    _plot_gps_rows()
    _plot_sampling_and_gaps()
    _plot_extents_and_lengths()
    _plot_network_size()
    _plot_missing_values()
    _plot_route_distances()
    _plot_trajectories()


if __name__ == "__main__":
    main()
