"""Verify route-independent candidates and opt-in reference labels."""

import shutil
from pathlib import Path

from common.paths import STUDENT_TEST_DIR
from features.candidates import build_candidates


def test_candidate_generation_works_without_route(tmp_path: Path) -> None:
    """Ensure inference candidates require no reference-route file or labels."""
    source = STUDENT_TEST_DIR / "00000007"
    copied_trip = tmp_path / source.name
    shutil.copytree(source, copied_trip)
    (copied_trip / f"{source.name}.route").unlink()

    candidates = build_candidates(
        copied_trip, load_route=False, radius_m=100.0, max_candidates=16
    )
    assert list(candidates.columns) == ["trip_id", "obs_idx", "cand_arc_id", "dist_m"]
    assert candidates["trip_id"].eq("00000007").all()
    assert candidates["dist_m"].between(0, 100).all()
    assert candidates.groupby("obs_idx").size().max() <= 16


def test_reference_labels_are_added_only_when_requested() -> None:
    """Keep the inference schema label-free and add route labels explicitly."""
    trip_dir = STUDENT_TEST_DIR / "00000007"
    unlabelled = build_candidates(
        trip_dir, load_route=False, radius_m=100.0, max_candidates=16
    )
    labelled = build_candidates(
        trip_dir, load_route=True, radius_m=100.0, max_candidates=16
    )
    assert "label" not in unlabelled.columns
    assert list(labelled.columns) == [
        "trip_id",
        "obs_idx",
        "cand_arc_id",
        "dist_m",
        "label",
    ]
    assert set(labelled["label"].unique()) <= {0, 1}
    assert int(labelled["label"].sum()) > 0
