"""Write a reproducible file and row inventory for each available partition."""

from pathlib import Path

import pandas as pd

from common.loader import list_trips, load_trip
from common.paths import RESULTS_DIR, STUDENT_TEST_DIR, TRAIN_DIR

FILE_SUFFIXES = (".track", ".nodes", ".arcs", ".route")


def inventory_partition(
    partition: str, root: Path
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Count files, bytes, and GPS rows so partition completeness is measurable."""
    rows = []
    for trip_dir in list_trips(root):
        sizes = {
            suffix.removeprefix(".") + "_bytes": (
                (trip_dir / f"{trip_dir.name}{suffix}").stat().st_size
                if (trip_dir / f"{trip_dir.name}{suffix}").is_file()
                else 0
            )
            for suffix in FILE_SUFFIXES
        }
        trip = load_trip(trip_dir)
        present = [
            suffix.removeprefix(".")
            for suffix in FILE_SUFFIXES
            if (trip_dir / f"{trip_dir.name}{suffix}").is_file()
        ]
        rows.append(
            {
                "partition": partition,
                "trip_id": trip.trip_id,
                "files_present": ";".join(present),
                "file_count": len(present),
                "gps_rows": len(trip.track),
                **sizes,
            }
        )
    frame = pd.DataFrame(rows)
    summary: dict[str, int | str] = {
        "partition": partition,
        "trip_count": len(frame),
        "trips_with_all_four_files": int((frame["file_count"] == 4).sum()),
        "total_gps_rows": int(frame["gps_rows"].sum()),
        "total_file_bytes": int(frame[[column for column in frame if column.endswith("_bytes")]].sum().sum()),
        "mean_gps_rows_per_trip": float(frame["gps_rows"].mean()),
    }
    return frame, summary


def main() -> None:
    """Persist per-trip and partition inventory tables under results/eda."""
    output_dir = RESULTS_DIR / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for partition, root in (("train", TRAIN_DIR), ("student_test", STUDENT_TEST_DIR)):
        trips, summary = inventory_partition(partition, root)
        trips.to_csv(output_dir / f"inventory_{partition}.csv", index=False)
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "inventory_summary.csv", index=False)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
