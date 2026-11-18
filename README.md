# PathSnap: Team 11

Dataset preparation, route-reference labeling, and candidate-feature tables for the PES University ML course project.

## Reproduce the analysis

From the repository root in Windows PowerShell, point `PATHSNAP_DATA_DIR` to the external dataset directory. The source data and large per-trip preprocessed/candidate-feature tables are not stored in this repository.

```powershell
$env:PATHSNAP_DATA_DIR = 'C:\path\to\pathsnap-data'
$python = '.\.venv\Scripts\python.exe'

& $python -m pytest -q
& $python -m data_processing.inventory
& $python -m data_processing.quality_checks
& $python -m data_processing.network_stats
& $python -m data_processing.preprocess
& $python -m features.labeling
& $python -m data_processing.candidate_analysis
& $python -m data_processing.build_feature_tables
& $python -m visualisation.eda_plots
& $python -m visualisation.folium_maps
```

Run the modules in this order: candidate analysis requires the route labels, and full candidate-feature generation requires both the labels and train-fitted candidate settings. Candidate settings and all learned thresholds are fit on `train`; `student_test` is only transformed or compared.

Generated row-level preprocessing and candidate-feature CSVs are written under `PATHSNAP_DATA_DIR\processed\`. EDA, summary tables, plots, and HTML maps are written under `results\`. The candidate-feature `label` column is a route-reference target for offline checks, not an input feature; do not use it as a prediction input.
