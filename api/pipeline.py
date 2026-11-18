"""
pipeline.py - Production Map-Matching Pipeline for Team 11 (Role D).

Integrates:
- Role A: Data loader (common.loader), candidates (features.candidates), features (features.feature_engineering)
- Role B: Tuned Bayesian LightGBM classifier (artifacts/final_model.joblib, artifacts/feature_list.json)
- Role C: Log-space HMM / Viterbi route decoder (decoder.viterbi, artifacts/hmm_config.json)
- Role D: High-speed server-side evaluation & latency benchmarking
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Ensure repository root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default dataset path if environment variable not set
DEFAULT_DATA_DIR = Path(r"C:\Users\ADARSH TELI\OneDrive\Desktop\ML HACKATHON\Student folder")
if "PATHSNAP_DATA_DIR" not in os.environ and DEFAULT_DATA_DIR.exists():
    os.environ["PATHSNAP_DATA_DIR"] = str(DEFAULT_DATA_DIR)

from common.loader import load_trip, list_trips, Trip
import common.paths as paths
from features.candidates import build_candidates
from features.feature_engineering import engineer_candidate_features
from decoder.viterbi import ViterbiConfig, decode

logger = logging.getLogger("pathsnap.pipeline")
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


class MapMatchingPipeline:
    def __init__(self):
        self.data_dir = paths.STUDENT_TEST_DIR if hasattr(paths, "STUDENT_TEST_DIR") and paths.STUDENT_TEST_DIR.exists() else DEFAULT_DATA_DIR / "student_test"
        self.classifier = None
        self.feature_cols = []
        self.hmm_config = None
        self._load_production_models()

    def _load_production_models(self):
        """Loads trained artifacts delivered by Role B and Role C."""
        model_path = ARTIFACTS_DIR / "final_model.joblib"
        feat_path = ARTIFACTS_DIR / "feature_list.json"
        hmm_path = ARTIFACTS_DIR / "hmm_config.json"

        if model_path.exists():
            try:
                import joblib
                self.classifier = joblib.load(model_path)
                logger.info(f"Loaded production classifier from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load classifier: {e}")

        if feat_path.exists():
            try:
                with open(feat_path, "r", encoding="utf-8") as f:
                    self.feature_cols = json.load(f)
                logger.info(f"Loaded {len(self.feature_cols)} feature definitions.")
            except Exception as e:
                logger.warning(f"Could not load feature list: {e}")

        if hmm_path.exists():
            try:
                self.hmm_config = ViterbiConfig.from_json(hmm_path)
                logger.info(f"Loaded HMM Viterbi configuration from {hmm_path}")
            except Exception as e:
                self.hmm_config = ViterbiConfig()
                logger.warning(f"Could not load HMM config, using defaults: {e}")
        else:
            self.hmm_config = ViterbiConfig()

    def list_available_trips(self) -> List[str]:
        """List the 10 student/evaluator test trips."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist!")
            return []
        
        trip_dirs = list_trips(self.data_dir)
        return [td.name for td in trip_dirs]

    def evaluate_trip(self, trip_id: str) -> Dict[str, Any]:
        """
        Execute full end-to-end production map-matching pipeline for trip_id:
        1. Load raw data (.track, .nodes, .arcs, .route)
        2. Generate spatial candidates
        3. Extract engineered features
        4. Score with Tuned Bayesian LightGBM classifier
        5. Reconstruct sequence with HMM Viterbi decoder
        6. Compute server-side metrics
        """
        trip_dir = self.data_dir / trip_id
        if not trip_dir.exists():
            raise FileNotFoundError(f"Trip directory {trip_dir} not found.")

        start_time = time.perf_counter()
        trip = load_trip(trip_dir, load_route=True)

        # 1. Candidate Generation (Role A)
        candidates = build_candidates(trip_dir, load_route=False, radius_m=205.82, max_candidates=57)

        # 2. Feature Engineering (Role A)
        features_df = engineer_candidate_features(trip, candidates)

        # 3. Model Scoring (Role B)
        if self.classifier is not None and self.feature_cols:
            X = features_df[self.feature_cols].to_numpy()
            scores = self.classifier.predict_proba(X)[:, 1]
        else:
            # Fallback distance ranking if model not yet loaded
            scores = 1.0 / (features_df["dist_m"].to_numpy() + 1.0)

        cand_scores = features_df[["obs_idx", "cand_arc_id"]].copy()
        cand_scores["score"] = scores

        # 4. Pointwise Classifier-Only Route (Baseline comparison)
        best_cands = cand_scores.sort_values(["obs_idx", "score"], ascending=[True, False]).groupby("obs_idx").first()
        classifier_only_route = np.full(len(trip.track), -1, dtype=np.int64)
        classifier_only_route[best_cands.index] = best_cands["cand_arc_id"].to_numpy()

        # 5. HMM Viterbi Route Reconstruction (Role C)
        arc_endpoints = trip.arcs[["from_node", "to_node"]].to_numpy()
        timestamps = trip.track["t"].to_numpy()

        predicted_route = decode(cand_scores, arc_endpoints, timestamps, self.hmm_config)
        predicted_route_list = [int(a) for a in predicted_route]

        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # 6. Compute Server-Side Metrics
        metrics: Dict[str, Any] = {
            "inference_time_ms": inference_time_ms,
            "observations_count": len(trip.track),
            "predicted_route_length": len(predicted_route_list),
            "candidates_evaluated": len(candidates),
        }

        # Route continuity calculation
        arcs_arr = trip.arcs[["from_node", "to_node"]].to_numpy()
        continuity_matches = 0
        total_transitions = max(1, len(predicted_route_list) - 1)
        for i in range(total_transitions):
            curr_arc = predicted_route_list[i]
            next_arc = predicted_route_list[i + 1]
            if curr_arc == next_arc:
                continuity_matches += 1
            elif curr_arc >= 0 and next_arc >= 0:
                curr_to = arcs_arr[curr_arc, 1] if curr_arc < len(arcs_arr) else -1
                next_from = arcs_arr[next_arc, 0] if next_arc < len(arcs_arr) else -2
                if curr_to == next_from:
                    continuity_matches += 1

        metrics["route_continuity"] = round(float(continuity_matches / total_transitions), 4)

        # Ground truth evaluation
        if trip.route is not None:
            metrics["has_ground_truth"] = True
            gt_set = set(trip.route)
            
            # Viterbi Accuracy
            vit_matches = sum(1 for a in predicted_route_list if a in gt_set and a >= 0)
            metrics["top1_accuracy"] = round(float(vit_matches / max(1, len(predicted_route_list))), 4)
            
            # Classifier-only Accuracy (Before HMM)
            pt_matches = sum(1 for a in classifier_only_route if a in gt_set and a >= 0)
            metrics["classifier_only_accuracy"] = round(float(pt_matches / max(1, len(classifier_only_route))), 4)

            # Accuracy improvement from Viterbi
            metrics["viterbi_gain"] = round(metrics["top1_accuracy"] - metrics["classifier_only_accuracy"], 4)
        else:
            metrics["has_ground_truth"] = False
            metrics["top1_accuracy"] = None
            metrics["classifier_only_accuracy"] = None

        # 7. Coordinates for UI Rendering
        track_vals = trip.track[["lat", "lon"]].to_numpy()
        step = max(1, len(track_vals) // 300)
        gps_display = [
            {"lat": float(track_vals[i, 0]), "lon": float(track_vals[i, 1]), "step": i}
            for i in range(0, len(track_vals), step)
        ]

        nodes_arr = trip.nodes[["lat", "lon"]].to_numpy()
        
        # Viterbi route coords
        route_coords = []
        for arc_id in predicted_route_list[::max(1, len(predicted_route_list) // 500)]:
            if 0 <= arc_id < len(arcs_arr):
                u, v = int(arcs_arr[arc_id, 0]), int(arcs_arr[arc_id, 1])
                if u < len(nodes_arr) and v < len(nodes_arr):
                    route_coords.append([float(nodes_arr[u, 0]), float(nodes_arr[u, 1])])
                    route_coords.append([float(nodes_arr[v, 0]), float(nodes_arr[v, 1])])

        # Classifier-only route coords (for before/after comparison toggle)
        classifier_coords = []
        for arc_id in classifier_only_route[::max(1, len(classifier_only_route) // 500)]:
            if 0 <= arc_id < len(arcs_arr):
                u, v = int(arcs_arr[arc_id, 0]), int(arcs_arr[arc_id, 1])
                if u < len(nodes_arr) and v < len(nodes_arr):
                    classifier_coords.append([float(nodes_arr[u, 0]), float(nodes_arr[u, 1])])
                    classifier_coords.append([float(nodes_arr[v, 0]), float(nodes_arr[v, 1])])

        return {
            "trip_id": trip_id,
            "metrics": metrics,
            "predicted_route": predicted_route_list,
            "gps_display": gps_display,
            "route_coords": route_coords,
            "classifier_coords": classifier_coords,
        }


pipeline_instance = MapMatchingPipeline()
