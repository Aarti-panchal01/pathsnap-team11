"""Role C candidate classifiers with a scikit-learn-compatible interface."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features.feature_engineering import CANDIDATE_FEATURE_COLUMNS


FEATURE_COLUMNS = tuple(CANDIDATE_FEATURE_COLUMNS)


def model_factories(random_state=11):
    """Return constructors so the shared harness can fit a fresh model per fold."""
    return {
        "model_4_hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=70, early_stopping=False, random_state=random_state,
        ),
        "model_5_mlp": lambda: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16), learning_rate_init=0.002,
                batch_size=1024, max_iter=60, tol=0.0003, n_iter_no_change=5,
                random_state=random_state,
            ),
        ),
    }


def score_candidates(model, candidates, feature_columns=FEATURE_COLUMNS):
    """Return canonical decoder columns using the probability of class 1."""
    scored = candidates.loc[:, ["obs_idx", "cand_arc_id"]].copy()
    if scored.empty:
        scored["score"] = np.empty(0, dtype=float)
        return scored
    classes = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(classes) != 1:
        raise ValueError("The classifier must expose positive candidate class 1.")
    inputs = candidates.loc[:, list(feature_columns)].astype("float32")
    scored["score"] = model.predict_proba(inputs)[:, classes[0]]
    return scored
