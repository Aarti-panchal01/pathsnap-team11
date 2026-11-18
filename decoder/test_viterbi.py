import numpy as np
import pandas as pd
import pytest

from decoder import decode


def candidates(*rows):
    return pd.DataFrame(rows, columns=["obs_idx", "cand_arc_id", "score"])


def test_future_candidate_changes_earlier_prediction():
    graph = np.array([[0, 1], [1, 2], [7, 8]])
    rows = candidates((1, 1, 0.95), (0, 2, 0.90), (0, 0, 0.70))
    assert decode(rows, graph, [0, 1]).tolist() == [0, 1]


def test_can_cross_one_bridge_arc_in_the_correct_direction():
    graph = np.array([[0, 1], [1, 2], [2, 3], [8, 9]])
    rows = candidates((0, 3, 0.91), (0, 0, 0.70), (1, 2, 0.90))
    assert decode(rows, graph, [0, 1]).tolist() == [0, 2]
    backwards = candidates((0, 2, 0.90), (1, 3, 0.91), (1, 0, 0.70))
    assert decode(backwards, graph, [0, 1]).tolist() == [2, 3]


def test_missing_candidates_and_large_gap_restart_sequences():
    graph = np.array([[0, 1], [1, 2], [7, 8]])
    rows = candidates((0, 2, 0.9), (0, 0, 0.7), (2, 1, 0.95), (3, 2, 0.9), (3, 1, 0.5))
    assert decode(rows, graph, [0, 1, 2, 10]).tolist() == [2, -1, 1, 2]
    assert decode(candidates(), graph, [0, 1]).tolist() == [-1, -1]


def test_disconnected_states_fall_back_and_zero_scores_are_finite():
    graph = np.array([[0, 1], [7, 8]])
    rows = candidates((0, 0, 0.0), (1, 1, 0.0))
    assert decode(rows, graph, [0, 1]).tolist() == [0, 1]


def test_invalid_arc_is_rejected():
    with pytest.raises(ValueError, match="invalid directed arc"):
        decode(candidates((0, 1, 0.5)), np.array([[0, 1]]), [0])
