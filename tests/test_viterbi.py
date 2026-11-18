"""Unit tests for the HMM Viterbi route reconstruction decoder."""

import numpy as np
import pandas as pd
from common.loader import load_trip
from common.paths import TRAIN_DIR
from decoder.viterbi import HMMConfig, ViterbiDecoder
from features.candidates import build_candidates
from features.feature_engineering import engineer_candidate_features


class DummyModel:
    """Mock classifier assigning score based on candidate distance rank."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Give higher probability to candidate_rank == 1
        probs_pos = np.clip(1.0 / (X[:, 1] + 1e-5), 0.05, 0.95)
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])


def test_viterbi_config():
    cfg = HMMConfig(beta_m=6.0, disconnection_log_penalty=-20.0)
    d = cfg.to_dict()
    assert d["beta_m"] == 6.0
    assert d["disconnection_log_penalty"] == -20.0


def test_viterbi_decode_trip():
    # Use real train trip 00000001 with mock model
    trip_dir = TRAIN_DIR / "00000001"
    trip = load_trip(trip_dir, load_route=False)
    
    # Generate small slice of candidates
    candidates = build_candidates(trip_dir, radius_m=50.0, max_candidates=10)
    # Take first 10 observations to keep unit test fast
    candidates_slice = candidates[candidates["obs_idx"] < 10].copy()
    track_slice = trip.track.iloc[:10].copy()
    trip_slice = trip
    trip_slice.track = track_slice

    features_df = engineer_candidate_features(trip_slice, candidates_slice)
    
    decoder = ViterbiDecoder()
    pred_arcs = decoder.decode_trip(trip_slice, features_df, DummyModel())

    assert len(pred_arcs) == 10
    assert pred_arcs.dtype == np.int64
    assert (pred_arcs >= 0).any()
