# PathSnap Role C

## Reproduce the Role C experiments

From the repository root, use the existing Role A `processed/candidate_features` and `processed/labels` directories. Point `--data-dir` to their parent dataset directory, which also contains `train` and `student_test`.

```powershell
python -m pip install numpy pandas scikit-learn joblib threadpoolctl networkx pyproj "shapely>=2"
python -m models.evaluate_role_c --data-dir $dataDir --jobs 6
```

The run saves a manifest, per-trip metrics, summary CSVs, and the two fitted estimators in `artifacts/role_c/`. The exported estimators were fitted using scikit-learn 1.9.1. For the shared team comparison, instantiate the estimator factories from `models.role_c` inside Role B's evaluation harness. The Role C runner can also accept a split manifest via `--folds-json`. Its format is `{"folds": [{"fit": ["00000001"], "validation": ["00000002"]}]}`, with the actual training trip IDs for each fold. Supplying the manifest aligns the splits. The team harness remains the source of the shared preprocessing and metric protocol.

## §3.2 Models 4 and 5

**Model 4 — Histogram Gradient Boosting.** The trees capture nonlinear relationships between point-to-arc distance, heading alignment, projected position, GPS motion, and graph degree. The estimator accepts missing values in the motion features and uses 70 boosting iterations. Its positive-class output scores each candidate arc.

**Model 5 — MLP.** A median imputer and standard scaler precede two hidden layers of 32 and 16 units. The imputer and scaler are fitted separately inside each training fold. The network uses a batch size of 1,024, learning rate of 0.002, and a maximum of 60 training iterations.

Both models use Role A's 18 candidate features. In the Role C verification, training keeps every positive, the closest negative per observation, and up to two additional randomly sampled negatives per positive within each trip. Evaluation retains the full candidate set. Candidate macro F1 and log loss use each labeled candidate row. Observation Top-1 counts missing reference candidates as misses. Metrics in the summary tables are averages across trips. The upstream candidate-generation settings were fitted by Role A using the full training partition, so the verification folds hold out classifier training only.

| Model | Provisional CV Top-1 | Provisional CV macro F1 | Student-test Top-1 | Student-test macro F1 | Student-test log loss | Scoring and ranking per student-test trip |
|---|---:|---:|---:|---:|---:|---:|
| Histogram Gradient Boosting | 76.82% | 85.58% | 81.82% | 88.11% | 0.0364 | 73.9 ms |
| MLP | 74.03% | 83.59% | 79.65% | 84.68% | 0.0461 | 38.1 ms |

Histogram Gradient Boosting had higher candidate macro F1 and Top-1 accuracy in this verification. The MLP took about half as long to score and rank candidates on the same machine. It reached the 60-iteration limit in two validation folds, so the cross-validation scores describe the fixed training budget. Timings exclude loading files and generating candidates. These results use the same 10 student-test trips for both models. The shared five-model comparison requires the Role B harness.

## §5 HMM and Viterbi route reconstruction

At each GPS observation, the hidden states are the highest-scoring directed candidate arcs. The default cap is 16 states per observation. The model's positive-class score, floored before taking its logarithm, serves as an emission likelihood proxy. The transition weights favor staying on the same arc, taking a directly connected arc, or taking an arc reachable through one intervening directed edge, in that order. A small weight handles links missing from this bounded graph check. States outside the retained candidate layer can account for unused transition mass.

Viterbi adds log emissions and log transitions, stores the best predecessor for each arc, and backtracks to find the highest-scoring sequence. Missing candidates yield `-1`. Missing, repeated or reversed timestamps and time steps longer than five seconds restart the sequence. The returned array contains one entry per GPS observation in its original order. Intermediate connector arcs influence transitions between observations.

In the comparison, pointwise prediction selects the highest-scoring candidate at each observation. Direct continuity measures the fraction of adjacent, valid-time GPS pairs assigned to the same arc or two directly connected arcs. Unmatched pairs count as breaks. Valid one-edge bridges can be inferred by the decoder and still count as a break under this direct-continuity metric.

| Model | Pointwise Top-1 | Viterbi Top-1 | Pointwise continuity | Viterbi continuity | Added decode time per trip |
|---|---:|---:|---:|---:|---:|
| Histogram Gradient Boosting | 81.82% | 87.48% | 87.20% | 98.26% | 106.8 ms |
| MLP | 79.65% | 86.09% | 85.87% | 98.64% | 93.0 ms |

Viterbi improved reference-arc accuracy on all 10 student-test trips for both models. Every one of the 19,160 observations in those trips received a candidate arc. The unmatched-state behavior was verified separately with a synthetic missing-candidate case.

### Decoder handoff

```python
import joblib

from decoder import ViterbiConfig, decode
from models.role_c import score_candidates

bundle = joblib.load(model_path)
scored = score_candidates(bundle["model"], candidate_features, bundle["feature_columns"])
config = ViterbiConfig.from_json("artifacts/hmm_config.json")
arc_ids = decode(scored, arc_endpoints, timestamps, config)
```

`arc_endpoints` contains `(from_node, to_node)` in `.arcs` row order, and `timestamps` contains the GPS timestamps in seconds. `candidate_features` has the Role A keys `obs_idx` and `cand_arc_id` plus the columns required by the fitted model. The decoder can accept scores from any team classifier that supplies `obs_idx`, `cand_arc_id`, and `score`. Retain `-1` in the per-observation response and filter it before indexing the road graph to draw arc geometry.
