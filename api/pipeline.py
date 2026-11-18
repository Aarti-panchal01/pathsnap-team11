"""
pipeline.py - Role D: Map-Matching Pipeline integration for Team 11.

Integrates:
- Common loader (from Role A: common.loader)
- Common paths (from Role A: common.paths)
- Classifier model (from Role B: artifacts/final_model.joblib)
- HMM Decoder (from Role C: artifacts/hmm_config.json)
- Fast fallback baseline matcher for continuous testing & demonstration
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

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

logger = logging.getLogger("pathsnap.pipeline")
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


class MapMatchingPipeline:
    def __init__(self):
        self.data_dir = paths.STUDENT_TEST_DIR if hasattr(paths, "STUDENT_TEST_DIR") else DEFAULT_DATA_DIR / "student_test"
        self.classifier = None
        self.hmm_decoder = None
        self._load_production_models()

    def _load_production_models(self):
        """Loads trained artifacts when delivered by Role B and Role C."""
        model_path = ARTIFACTS_DIR / "final_model.joblib"
        hmm_path = ARTIFACTS_DIR / "hmm_config.json"

        if model_path.exists():
            try:
                import joblib
                self.classifier = joblib.load(model_path)
                logger.info(f"Loaded production classifier from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load classifier: {e}")

        if hmm_path.exists():
            try:
                logger.info(f"Found HMM configuration at {hmm_path}")
            except Exception as e:
                logger.warning(f"Could not load HMM: {e}")

    def list_available_trips(self) -> List[str]:
        """List the 10 student/evaluator test trips."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist!")
            return []
        
        trip_dirs = list_trips(self.data_dir)
        return [td.name for td in trip_dirs]

    def _run_baseline_decoder(self, trip: Trip) -> List[int]:
        """
        Fast spatial baseline matcher (nearest arc midpoint) used when
        Role B + Role C models are being trained.
        Guarantees the API can be tested and demoed anytime.
        """
        track_df = trip.track
        nodes_df = trip.nodes
        arcs_df = trip.arcs

        num_obs = len(track_df)
        from_nodes = arcs_df["from_node"].to_numpy()
        to_nodes = arcs_df["to_node"].to_numpy()
        
        node_coords = nodes_df[["lon", "lat"]].to_numpy()
        
        valid_mask = (from_nodes < len(node_coords)) & (to_nodes < len(node_coords))
        valid_arc_indices = np.where(valid_mask)[0]
        
        midpoints_lon = (node_coords[from_nodes[valid_mask], 0] + node_coords[to_nodes[valid_mask], 0]) / 2.0
        midpoints_lat = (node_coords[from_nodes[valid_mask], 1] + node_coords[to_nodes[valid_mask], 1]) / 2.0
        
        gps_coords = track_df[["lon", "lat"]].to_numpy()
        
        predicted_arcs: List[int] = []
        for i in range(num_obs):
            gps_lon, gps_lat = gps_coords[i, 0], gps_coords[i, 1]
            d2 = (midpoints_lon - gps_lon)**2 + (midpoints_lat - gps_lat)**2
            nearest_idx = np.argmin(d2)
            predicted_arcs.append(int(valid_arc_indices[nearest_idx]))

        return predicted_arcs

    def evaluate_trip(self, trip_id: str) -> Dict[str, Any]:
        """
        Execute prediction pipeline for trip_id, compute metrics server-side,
        and package response for the UI.
        """
        trip_dir = self.data_dir / trip_id
        if not trip_dir.exists():
            raise FileNotFoundError(f"Trip directory {trip_dir} not found.")

        start_time = time.perf_counter()
        trip = load_trip(trip_dir, load_route=True)

        # Run Map-Matching
        if self.classifier is not None and self.hmm_decoder is not None:
            # Future integration with Role B & Role C components
            predicted_route = self._run_baseline_decoder(trip)
        else:
            predicted_route = self._run_baseline_decoder(trip)

        inference_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Server-Side Metrics Calculation (Ground Truth never exposed)
        metrics: Dict[str, Any] = {
            "inference_time_ms": inference_time_ms,
            "observations_count": len(trip.track),
            "predicted_route_length": len(predicted_route),
        }

        # Route continuity calculation
        arcs_arr = trip.arcs[["from_node", "to_node"]].to_numpy()
        continuity_matches = 0
        total_transitions = max(1, len(predicted_route) - 1)
        for i in range(total_transitions):
            curr_arc = predicted_route[i]
            next_arc = predicted_route[i + 1]
            if curr_arc == next_arc:
                continuity_matches += 1
            else:
                curr_to = arcs_arr[curr_arc, 1] if curr_arc < len(arcs_arr) else -1
                next_from = arcs_arr[next_arc, 0] if next_arc < len(arcs_arr) else -2
                if curr_to == next_from:
                    continuity_matches += 1

        metrics["route_continuity"] = round(float(continuity_matches / total_transitions), 4)

        if trip.route is not None:
            metrics["has_ground_truth"] = True
            gt_set = set(trip.route)
            
            # Pointwise matching: how many predicted arcs appear in reference route
            matches = sum(1 for a in predicted_route if a in gt_set)
            metrics["top1_accuracy"] = round(float(matches / max(1, len(predicted_route))), 4)
            
            min_len = min(len(predicted_route), len(trip.route))
            exact_pos_matches = sum(1 for i in range(min_len) if predicted_route[i] == trip.route[i])
            metrics["exact_step_accuracy"] = round(float(exact_pos_matches / max(1, min_len)), 4)
        else:
            metrics["has_ground_truth"] = False
            metrics["top1_accuracy"] = None

        # Prepare downsampled GPS points for UI map
        track_vals = trip.track[["lat", "lon"]].to_numpy()
        step = max(1, len(track_vals) // 300)
        gps_display = [
            {"lat": float(track_vals[i, 0]), "lon": float(track_vals[i, 1]), "step": i}
            for i in range(0, len(track_vals), step)
        ]

        # Prepare predicted route coordinates
        nodes_arr = trip.nodes[["lat", "lon"]].to_numpy()
        route_coords = []
        for arc_id in predicted_route[::max(1, len(predicted_route) // 400)]:
            if arc_id < len(arcs_arr):
                u, v = int(arcs_arr[arc_id, 0]), int(arcs_arr[arc_id, 1])
                if u < len(nodes_arr) and v < len(nodes_arr):
                    route_coords.append([float(nodes_arr[u, 0]), float(nodes_arr[u, 1])])
                    route_coords.append([float(nodes_arr[v, 0]), float(nodes_arr[v, 1])])

        return {
            "trip_id": trip_id,
            "metrics": metrics,
            "predicted_route": predicted_route,
            "gps_display": gps_display,
            "route_coords": route_coords
        }


pipeline_instance = MapMatchingPipeline()
