# Role A — Dataset preparation, route labels, and candidate features

## Scope and reproducibility

Work was limited to `common/`, `data_processing/`, `features/`, `visualisation/`, `tests/`, `notes/`, and `results/`. The project branch was `feature/data-features`. All source files are read through `PATHSNAP_DATA_DIR`; processed row-level outputs were written to its `processed/` directory. Route labeling, candidate generation, and candidate-feature tables are implemented; no models or predicted routes were generated.

The inventory, quality, network, labeling, candidate, and visualization tables explicitly identify `train` and `student_test`. Only `train` was used to derive far-distance cutoffs, candidate radius, and candidate cap. The reference route is treated as reference data, not guaranteed truth.

## Dataset inventory (actual run)

- `train`: 80 eight-digit trip directories; 80/80 had `.track`, `.nodes`, `.arcs`, and `.route`; 208,417 GPS rows; 89,737,452 total source-file bytes.
- `student_test`: 10 eight-digit trip directories; 10/10 had all four files; 19,160 GPS rows; 4,368,591 total source-file bytes.
- Train GPS rows per trip: min 361, median 1,529, max 19,313. Student-test: min 262, median 1,420, max 3,958.
- IDs remain strings, including leading zeroes. Networks are trip-specific; their sizes vary substantially.

## Data-quality findings and CRS

The real coordinates contradict the supplied approximate Mannheim-area extent. Across train tracks and nodes, the observed bounding extent was approximately longitude `-100.428454` to `137.247005` and latitude `-34.650477` to `61.277181`. GPS-track-only train bounds were `-100.418538` to `137.237029` longitude and `-34.640291` to `61.267258` latitude. Student-test tracks and nodes also span multiple regions; GPS-track-only bounds were `-93.424337` to `77.271115` longitude and `42.645529` to `61.699590` latitude.

This world-spanning extent makes one global UTM zone, including EPSG:32632, inappropriate for distances across all trips. The train trips use 25 distinct local UTM zones; EPSG:32632 applies to only 8 train trips. Distance calculations therefore use a trip-local UTM zone selected from that trip's real GPS coordinate extent. The student-test analysis uses the same local-zone rule, without contributing to any learned cutoff. Coordinates are always longitude-first, and no Euclidean distance in degrees is used.

Observed quality results, reported separately by partition:

- Missing track values, missing GPS coordinates, missing timestamps, out-of-range GPS coordinates, suspected swapped pairs, non-increasing timestamps, out-of-range arc endpoint IDs, and out-of-range route arc IDs: all zero in both partitions.
- Train: 201,048 of 208,337 positive time intervals were exactly 1 second (96.5013%); 1,216 intervals exceeded 5 seconds. The largest was 153,064 seconds (42 hours, 31 minutes, 4 seconds).
- Student-test: 18,886 of 19,150 positive intervals were exactly 1 second (98.6214%); 44 intervals exceeded 5 seconds. The largest was 210 seconds.
- The quality report flags 49 train trips and 2 student-test trips for at least one gap over 5 seconds or broken route. Broken-route checks found 13 discontinuities across 10 train trips and 1 discontinuity in 1 student-test trip. No route arc IDs were out of range.
- Train trajectory lengths ranged from 6,338.5 m to 169,552.4 m (median 24,046.3 m). Student-test ranged from 5,559.1 m to 71,778.5 m (median 20,964.4 m). Length sums use adjacent valid GPS rows in each trip's local UTM projection.

## Network and reference-route findings

- Train networks contain 1,933,485 nodes and 3,876,904 directed arcs in total; median trip sizes are 9,608 nodes and 19,526.5 arcs. Student-test contains 90,689 nodes and 185,041 arcs in total.
- There were no isolated nodes. The median train trip had 20 weakly connected components; the median student-test trip had 11. Strong-component medians were 127 and 53.5, respectively.
- The mean per-trip reverse-twin share was 92.17% for train and 92.69% for student-test, so the networks are mostly bidirectional but retain directed-only arcs.
- Routes contain 31,336 arc references across train (median 280 per trip) and 3,477 across student-test (median 250 per trip). Median route geometry lengths were 14,313.8 m and 14,763.2 m, respectively.
- Point-to-reference-route distances use projected metres and the route's valid arc geometry. Across all GPS rows, train median/p95/max distances were 5.61/140.27/569.66 m; student-test median/p95/max were 5.16/16.78/105.91 m. The long train tail and detected discontinuities are additional reasons not to regard the route as error-free truth.

## Preprocessing decisions

- Kept every source GPS row. `preprocess_trip` adds coordinate/timestamp validity, time-step, non-increasing-time, gap-before, gap-point, nearest-network-distance, and far-from-network columns. No row was dropped, interpolated, or silently repaired.
- A gap point is flagged when its positive preceding time step is greater than 5 seconds. This fixed data-quality reporting boundary is not fitted from student-test.
- The far-from-network distance is measured to the nearest valid road-arc geometry in the trip-local projected CRS. Its cutoff is the 99th percentile of the 208,417 train GPS-to-network distances: 97.4095 m (train median 4.9701 m, p95 83.4339 m). It is fit on train only and applied unchanged to both partitions.
- Result: 2,085 train points and 0 student-test points were flagged far from the network. Missing/out-of-range coordinates and non-increasing timestamps affected zero points/steps in both partitions. The gap flag marked 1,216 train and 44 student-test rows. Broken routes were listed separately and not removed.
- The preprocessing runner wrote 80 train and 10 student-test row CSVs outside the repository to `PATHSNAP_DATA_DIR\processed\preprocess\`, plus its threshold and summary files.

## Draft for report Section 2.2

We loaded each trip's headerless whitespace-separated files independently, retained its eight-digit ID as a string, and preserved the original GPS rows and trip-specific network indices. GPS coordinates were interpreted as longitude/latitude. Because the measured training extent spans many continents and 25 local UTM zones, all distance calculations use a local UTM CRS selected for the individual trip rather than treating degrees as metres or applying EPSG:32632 globally. We checked train and student-test separately. Both partitions were nearly 1 Hz, but the train partition contained 1,216 gaps longer than 5 seconds (maximum 153,064 seconds), while student-test contained 44 (maximum 210 seconds). No GPS coordinates were missing or out of range, and no timestamps were non-increasing. Preprocessing therefore retained all rows and added explicit flags for coordinate validity, timestamp order, time gaps, and distance to the nearest road arc. The far-from-network cutoff (97.4095 m) was calibrated at the 99th percentile using train only and then applied unchanged to student-test. The supplied routes were retained as references; 10 train trips and 1 student-test trip had route discontinuities, so these trips were flagged rather than removed. The data's actual global coordinate extent contradicts the supplied approximate Mannheim-area description and should be confirmed before subsequent model design.

## Stage 3 — route-labeling investigation and method

The route-labeling analysis used the supplied route files in `train` first. Every train and student-test GPS row was assignable to a valid route arc. Train has 31,336 route-arc occurrences for 208,417 GPS observations; student-test has 3,477 occurrences for 19,160 observations. Counts are by route occurrence, so repeated arc IDs at different route positions remain distinct. The per-occurrence counts are saved in `results/eda/route_observations_per_occurrence.csv`: 3,069 train and 368 student-test route occurrences received no observation label; the train occurrence count median/p95 was 3/19 observations, with a maximum of 16,609 on one occurrence. In train trip `00000054`, one route occurrence received 16,609 of 19,313 GPS labels. That extreme concentration should be reviewed rather than interpreted as clean truth.

The labeler computes all GPS-to-route-segment distances in the trip-local projected CRS, then uses dynamic programming to minimize total squared distance subject to a non-decreasing route position in GPS row order. This avoids the large error from a simple “never move backward” greedy clamp. Exact nearest-distance ties within 1e-6 m are flagged as ambiguous; ties are resolved toward the earliest route occurrence consistent with the global assignment. A point is flagged far when its nearest-route distance exceeds the train-only p95 threshold of 140.2723 m. These are diagnostic flags, not confidence guarantees or corrections to the reference route. Rows with invalid coordinates would remain unlabeled rather than be removed.

Measured results:

- Train: median/p95/max nearest-route distance 5.61/140.27/569.66 m; 14,123 exact-tie observations (6.78%); 11,122 labels differ from the unconstrained nearest occurrence; 10,421 points exceed the train-derived far threshold.
- Student-test: median/p95/max nearest-route distance 5.16/16.78/105.91 m; 3,417 exact-tie observations (17.84%); 2,056 labels differ from the unconstrained nearest occurrence; none exceeds the train-derived threshold.
- Median trip-level p95 assigned-label distance was 14.28 m for train and 15.80 m for student-test. The global p95 assigned-label distances were 142.10 m and 16.87 m, respectively, reflecting a long tail in a small set of train trips.
- There are 13 route discontinuities across 10 train trips and 1 across 1 student-test trip. Labels preserve route order but do not repair broken route segments.

## Stage 4 — candidate-generation precheck

`features/candidates.py` uses each trip's own projected arc lines and a Shapely STRtree `dwithin` query. The query is batched in 256 GPS observations to bound temporary memory, candidates are ranked by metric distance (then arc ID for deterministic ties), and the cap is applied per observation. The inference default loads no route and emits no label; route labels are added only when `load_route=True`.

Candidate settings were fit using train only: radius = p99 of assigned-label-arc distance, 205.8227 m; cap = p99 of the label arc's rank among arcs inside that radius, 57 candidates per observation. The rank distribution inside the radius had p95 22, p99 57, maximum 181. Settings are saved in `PATHSNAP_DATA_DIR\processed\candidate_config.json`.

At this checkpoint, the candidate precheck (without engineered features or persisted full candidate tables) measured:

- Train: 8,000,052 candidate rows; 38.3848 average candidates per GPS point; 4,031/208,417 labels missed (1.9341%), comprising 2,085 radius misses and 1,946 cap misses.
- Student-test comparison: 777,686 candidate rows; 40.5890 average candidates per GPS point; 0/19,160 labels missed. Student-test did not contribute to choosing radius or cap.
- Eight train trips had at least one miss. The largest trip-level miss rate was 13.5352% in trip `00000033` (791 misses, all due to the cap); trip `00000054` had 2,426 misses (1,477 radius, 949 cap). This uneven train recall is a material limitation of the pooled quantile settings and should be considered before feature-table generation.
- An inference-style `build_candidates(load_route=False)` run on train trip `00000001` took 0.436 s and returned 66,109 rows for 1,315 GPS observations (50.273 candidates per point). Runtime is machine- and load-dependent.

Per-trip candidate precheck results are saved in `results/tables/candidate_precheck_summary.csv`; partition totals are in `results/tables/candidate_precheck_partition_summary.csv`.

## Stage 5 — candidate feature engineering

`features/feature_engineering.py` adds 18 route-independent candidate features. Geometric features include the existing nearest-arc distance, per-observation candidate rank (distance, then arc ID), arc length/bearing, clamped projection fraction, signed cross-track distance, and distances to both endpoints. Motion context includes GPS heading from the immediately preceding observation, wrapped heading difference and cosine alignment, preceding time step, speed, and a gap flag. Heading and speed are left missing when the previous step is invalid, non-positive, longer than 5 seconds, or stationary. Topological features are incoming/outgoing arc counts at both endpoints. All geometric quantities use the trip-local projected CRS.

The optional `label` column is retained only as a target for offline evaluation and is not in `CANDIDATE_FEATURE_COLUMNS`; no route position, route sequence, or reference-route distance is used as an input feature. Candidate rank, label, and route labels are never computed from student-test statistics.

## Stage 6 — full candidate-feature tables

`python -m data_processing.build_feature_tables` writes one gzip CSV per trip under `PATHSNAP_DATA_DIR\processed\candidate_features\<partition>\`, using the Stage 4 train-fitted radius/cap and cached Stage 3 labels. It writes no row-level candidate data into the repository. The per-trip, partition, and feature-missing summaries are saved in `results/tables/`; `candidate_feature_sample.csv` contains 50 deterministic sample rows per trip, 4,500 rows total, and includes a partition column.

Measured output:

- Train: 80 files, 8,000,052 candidate rows, 38.3848 candidates per GPS point, 4,031 reference labels missed (2,085 radius and 1,946 cap), and 408,267,399 compressed bytes. Per-trip build time summed to 267.24 seconds.
- Student-test: 10 files, 777,686 candidate rows, 40.5890 candidates per GPS point, no reference labels missed, and 40,337,176 compressed bytes. Per-trip build time summed to 25.87 seconds. Settings remained train-fitted.
- The candidate-feature schema is consistent across all 90 files and contains 23 columns: partition, trip/observation/candidate identifiers, reference label, and the 18 features listed above.
- The deterministic sample contains 4,000 train and 500 student-test rows (4,500 total).
- All candidate rows have non-missing distance, arc length, candidate rank, projection fraction, signed cross-track distance, endpoint distances, bearing, gap flag, and endpoint degree features. The time step is missing for 0.0475% of train and 0.0585% of student-test candidate rows, corresponding to first observations. Heading/alignment are missing for 0.6406% of train and 0.3810% of student-test candidate rows; speed is missing for 0.6385% and 0.3810%, respectively. These motion features are intentionally unavailable when a valid short-step movement cannot be computed.

## Open questions for later

- **Reference labels:** How should broken or highly concentrated reference routes be treated in model validation? Keep route labels distinct from guaranteed ground truth.
- **Candidate radius/cap:** The approved pooled train settings have 1.9341% train misses overall but uneven per-trip misses (up to 13.5352%). Should later validation adopt a per-trip or more conservative strategy, and what candidate-volume ceiling is acceptable? Do not fit settings from student-test.
- **Features:** Which additional features, if any, are justified after testing these geometry, motion, and topology features? Preserve the route-independent input schema and avoid reference-route leakage.