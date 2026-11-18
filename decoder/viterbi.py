"""Decode one directed candidate arc per observation with log-space Viterbi."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ViterbiConfig:
    max_candidates: int = 16
    max_gap_seconds: float = 5.0
    emission_floor: float = 1e-8
    same_arc_weight: float = 1.0
    adjacent_arc_weight: float = 0.85
    one_bridge_weight: float = 0.35
    unlinked_weight: float = 0.02

    def __post_init__(self):
        if type(self.max_candidates) is not int or self.max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer.")
        if not np.isfinite(self.max_gap_seconds) or self.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be finite and positive.")
        if not 0 < self.emission_floor < 1:
            raise ValueError("emission_floor must be between zero and one.")
        if not 0 < self.unlinked_weight <= self.one_bridge_weight <= self.adjacent_arc_weight <= self.same_arc_weight <= 1:
            raise ValueError("Transition weights must be positive, ordered, and at most one.")

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _transition_logs(previous, current, endpoints, edge_keys, stride, config):
    previous_end = endpoints[previous, 1, None]
    current_start = endpoints[current, 0]
    queries = previous_end * stride + current_start
    locations = np.searchsorted(edge_keys, queries)
    bridged = edge_keys[np.minimum(locations, len(edge_keys) - 1)] == queries
    weights = np.full(queries.shape, config.unlinked_weight)
    weights[bridged] = config.one_bridge_weight
    weights[previous_end == current_start] = config.adjacent_arc_weight
    weights[previous[:, None] == current] = config.same_arc_weight
    # The unused mass belongs to states outside this layer. Keep the same
    # denominator across rows so even a singleton layer preserves connectivity.
    return np.log(weights) - np.log(len(current) * config.same_arc_weight)


def _backtrack(layers, scores, output):
    state = int(np.argmax(scores))
    for observation, arcs, parents in reversed(layers):
        output[observation] = arcs[state]
        if parents is not None:
            state = parents[state]


def decode(candidates, arc_endpoints, timestamps, config=None):
    """Return arc IDs in GPS order, with -1 for observations without candidates.

    Candidates are a DataFrame with ``obs_idx``, ``cand_arc_id`` and ``score``.
    Endpoint rows map arc IDs to directed ``(from_node, to_node)`` pairs.
    Timestamps are in seconds. Large, missing or non-increasing time steps
    start a fresh sequence, as do observations without any candidates.
    """
    cfg = ViterbiConfig() if config is None else config
    times = np.asarray(timestamps, dtype=float)
    endpoints = np.asarray(arc_endpoints)
    if times.ndim != 1 or endpoints.ndim != 2 or endpoints.shape[1] != 2:
        raise ValueError("Expected 1-D timestamps and an (arc_count, 2) endpoint array.")
    if not np.issubdtype(endpoints.dtype, np.integer):
        raise ValueError("Road endpoints must contain integer node IDs.")
    output = np.full(len(times), -1, dtype=np.int64)
    rows = candidates.loc[:, ["obs_idx", "cand_arc_id", "score"]]
    if rows.empty:
        return output
    obs_ids = rows["obs_idx"].to_numpy()
    arc_ids = rows["cand_arc_id"].to_numpy()
    probabilities = rows["score"].to_numpy(dtype=float)
    if not np.issubdtype(obs_ids.dtype, np.integer) or not np.issubdtype(arc_ids.dtype, np.integer):
        raise ValueError("Candidate observation and arc IDs must be integers.")
    if (obs_ids < 0).any() or (obs_ids >= len(times)).any():
        raise ValueError("An observation index is outside the GPS track.")
    if (arc_ids < 0).any() or (arc_ids >= len(endpoints)).any() or (endpoints[arc_ids] < 0).any():
        raise ValueError("A candidate references an invalid directed arc.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Candidate scores must be finite numbers in [0, 1].")

    order = np.lexsort((arc_ids, -probabilities, obs_ids))
    obs_ids, arc_ids, probabilities = obs_ids[order], arc_ids[order], probabilities[order]
    cuts = np.r_[0, np.flatnonzero(obs_ids[1:] != obs_ids[:-1]) + 1, len(obs_ids)]
    valid_endpoints = endpoints[np.all(endpoints >= 0, axis=1)].astype(np.int64, copy=False)
    stride = int(valid_endpoints.max()) + 1
    if stride > 3_037_000_499:
        raise ValueError("Node IDs exceed the integer range of the transition index.")
    endpoints = endpoints.astype(np.int64, copy=False)
    edge_keys = np.unique(valid_endpoints[:, 0] * stride + valid_endpoints[:, 1])

    previous_obs = None
    layers = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        observation = int(obs_ids[start])
        # Duplicate arcs retain their highest score. Preserve score order for pruning.
        _, unique_indices = np.unique(arc_ids[start:end], return_index=True)
        retained = start + np.sort(unique_indices)[:cfg.max_candidates]
        arcs = arc_ids[retained]
        emissions = np.log(np.maximum(probabilities[retained], cfg.emission_floor))
        elapsed = times[observation] - times[previous_obs] if previous_obs is not None else np.nan
        linked_in_time = previous_obs == observation - 1 and 0 < elapsed <= cfg.max_gap_seconds
        if linked_in_time:
            transitions = _transition_logs(layers[-1][1], arcs, endpoints, edge_keys, stride, cfg)
            paths = scores[:, None] + transitions
            parents = paths.argmax(axis=0)
            scores = paths[parents, np.arange(len(arcs))] + emissions
        else:
            if layers:
                _backtrack(layers, scores, output)
            layers = []
            parents = None
            scores = emissions
        layers.append((observation, arcs, parents))
        previous_obs = observation
    _backtrack(layers, scores, output)
    return output



# ---------------------------------------------------------
# Backwards-compatibility wrapper for test suites / callers
# ---------------------------------------------------------
@dataclass
class HMMConfig:
    beta_m: float = 6.0
    disconnection_log_penalty: float = -20.0
    max_gap_seconds: float = 5.0

    def to_dict(self):
        return {
            "beta_m": self.beta_m,
            "disconnection_log_penalty": self.disconnection_log_penalty,
            "max_gap_seconds": self.max_gap_seconds
        }


class ViterbiDecoder:
    def __init__(self, config=None):
        self.config = config or ViterbiConfig()

    def decode_trip(self, trip, features_df, model):
        """Wrapper bridging Trip object + features DataFrame + Model into decode()."""
        # Feature columns excluding metadata
        meta_cols = {"trip_id", "obs_idx", "cand_arc_id", "label", "partition"}
        feature_cols = [c for c in features_df.columns if c not in meta_cols]
        X = features_df[feature_cols].to_numpy()

        # Score candidates using model
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        elif hasattr(model, "predict"):
            probs = model.predict(X)
        else:
            probs = np.full(len(X), 0.5)

        cand_df = features_df[["obs_idx", "cand_arc_id"]].copy()
        cand_df["score"] = probs

        arc_endpoints = trip.arcs[["from_node", "to_node"]].to_numpy()
        timestamps = trip.track["t"].to_numpy()

        cfg = self.config if isinstance(self.config, ViterbiConfig) else ViterbiConfig()
        return decode(cand_df, arc_endpoints, timestamps, cfg)
