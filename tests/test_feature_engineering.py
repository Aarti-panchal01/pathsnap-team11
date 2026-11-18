"""Check candidate features use metric geometry and local temporal context."""

import numpy as np
import pandas as pd

from common.loader import Trip
from features.feature_engineering import (
    CANDIDATE_FEATURE_COLUMNS,
    engineer_candidate_features,
)


def test_candidate_features_rank_arcs_and_flag_motion_across_gaps() -> None:
    """Ensure route labels remain targets and long gaps do not imply a speed or heading."""
    trip = Trip(
        trip_id="00000001",
        track=pd.DataFrame(
            {
                "lon": [8.5805, 8.5806],
                "lat": [49.8401, 49.8401],
                "t": [100.0, 110.0],
            }
        ),
        nodes=pd.DataFrame(
            {
                "lon": [8.58, 8.581],
                "lat": [49.84, 49.84],
            },
            index=pd.RangeIndex(2, name="node_id"),
        ),
        arcs=pd.DataFrame(
            {"from_node": [0, 1], "to_node": [1, 0]},
            index=pd.RangeIndex(2, name="arc_id"),
        ),
        route=None,
    )
    candidates = pd.DataFrame(
        {
            "trip_id": ["00000001", "00000001", "00000001"],
            "obs_idx": [0, 0, 1],
            "cand_arc_id": [1, 0, 0],
            "dist_m": [2.0, 1.0, 1.0],
            "label": [0, 1, 1],
        }
    )

    result = engineer_candidate_features(trip, candidates)

    assert result.loc[result["obs_idx"] == 0, "cand_arc_id"].tolist() == [0, 1]
    assert result.loc[result["obs_idx"] == 0, "candidate_rank"].tolist() == [1, 2]
    assert result["arc_length_m"].gt(50).all()
    assert result["projection_fraction"].between(0, 1).all()
    assert result["label"].tolist() == [1, 0, 1]
    assert not set(CANDIDATE_FEATURE_COLUMNS).difference(result.columns)
    after_gap = result["obs_idx"] == 1
    assert result.loc[after_gap, "time_step_s"].eq(10).all()
    assert result.loc[after_gap, "is_gap_point"].all()
    assert result.loc[after_gap, "gps_speed_mps"].isna().all()
    assert result.loc[after_gap, "gps_heading_rad"].isna().all()
    assert np.isfinite(result.loc[result["obs_idx"] == 0, "signed_cross_track_m"]).all()
