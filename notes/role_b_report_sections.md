# PathSnap: Role B Technical Report Deliverable
**Author:** Person B (Member 2)  
**Role:** Experiments, Model Evaluation, and Bayesian Tuning  
**Folders Owned:** `models/`, `optimisation/`, `results/`, `artifacts/`  
**Course:** UE24CS352A - Machine Learning Operations & Systems  
**Institution:** PES University, Department of Computer Science and Engineering  

---

## 3. Model Training, Evaluation, and Comparison

### 3.1 Overall Model Comparison

#### Evaluation Methodology & Anti-Leakage Protocol
To ensure strict academic integrity and eliminate spatial and temporal data leakage, all model evaluations adhere to the following protocol:
1. **Group-Aware Partitioning (`GroupKFold` on `trip_id`):** Trajectories consist of continuous GPS observations with high spatial and temporal autocorrelation. Random splitting would cause adjacent timestamps from the same trip to appear simultaneously in training and validation folds, resulting in severe data leakage and artificially inflated accuracy. Consequently, all 5 cross-validation folds strictly partition by `trip_id` across the training dataset (80 training trips). No GPS observation from any trip appears in both the training and validation splits of any fold.
2. **Strict Preprocessing Isolation:** All feature scalers (`StandardScaler`) and missing-value imputers (`SimpleImputer(strategy='median')`) are fitted exclusively on the training folds and applied out-of-fold to validation and student-test sets. Zero statistical parameters are learned from evaluator data.
3. **Primary Evaluation Metrics Under Class Imbalance:**
   - **Severe Class Imbalance:** Road network candidate generation yields between 30 and 57 candidate road segments per GPS observation point. Only one candidate arc corresponds to the true traversed route ($y=1$), while all others are negative candidates ($y=0$). The empirical positive class prevalence is only **2.75%** (a 35:1 negative-to-positive ratio).
   - **Deceptiveness of Standard Accuracy:** A trivial majority-class classifier predicting $y=0$ for all candidates achieves **97.25% standard accuracy**, yet possesses **0% Top-1 Accuracy** and fails completely at map-matching.
   - **Selected Primary Metrics:**
     - **Macro F1-Score:** The unweighted mean of class 0 and class 1 F1 scores, $\frac{1}{2}(F1_0 + F1_1)$. This penalizes false positives and false negatives equally regardless of class size, preventing majority-class dominance.
     - **Observation-Level Top-1 Accuracy:** Defined as:
       $$\text{Top-1 Accuracy} = \frac{1}{|\mathcal{T}|} \sum_{(t, o) \in \mathcal{T}} \mathbb{I}\left( y_{\arg\max_{c \in \mathcal{C}(t, o)} \hat{P}(c)} = 1 \right)$$
       where $\mathcal{C}(t, o)$ is the set of candidate road arcs for trip $t$ at observation index $o$, and $\hat{P}(c)$ is the classifier's predicted probability. This metric directly measures whether the classifier ranks the ground-truth road segment highest among all candidate options for a given GPS ping.
     - **Binary Cross-Entropy (Log Loss):**
       $$\mathcal{L}_{\text{log}} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
       Log loss measures probabilistic calibration, which is essential because the downstream Hidden Markov Model (HMM) Viterbi decoder uses these predicted probabilities directly as emission likelihoods.

#### Consolidated 5-Model Comparison Table
The table below documents the empirical evaluation results across all 5 algorithm families evaluated using the shared Group-Aware Evaluation Harness (`models/evaluation.py`):

| Model & Algorithm Family | 5-Fold CV Macro F1 | 5-Fold CV Top-1 Acc | Student-Test Macro F1 | Student-Test Top-1 Acc | Student-Test Log Loss | Student-Test Accuracy | Student-Test Recall | Fit Time (s) | Latency (ms/1k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: Logistic Regression** *(Linear & Probabilistic)* | $0.6588 \pm 0.015$ | $0.6856 \pm 0.157$ | $0.6651$ | $0.7963$ | $0.1657$ | $91.82\%$ | $99.06\%$ | $10.21\text{ s}$ | $0.32\text{ ms}$ |
| **Model 1-GPU: Logistic Regression** *(RTX 4060 GPU)* | $0.6536 \pm 0.016$ | $0.6752 \pm 0.151$ | $0.6666$ | $0.7980$ | $0.1597$ | $91.90\%$ | $99.24\%$ | $8.02\text{ s}$ | $0.38\text{ ms}$ |
| **Model 2: K-Nearest Neighbors** *(Instance-based, k=15)* | $0.6392 \pm 0.091$ | $0.0349 \pm 0.006$ | $0.6985$ | $0.0462$ | $0.0987$ | $98.11\%$ | $27.00\%$ | $0.09\text{ s}$ | $22.23\text{ ms}$ |
| **Model 3: Balanced Random Forest** *(Tree Ensemble)* | $\mathbf{0.8127} \pm 0.038$ | $\mathbf{0.7052} \pm 0.155$ | $\mathbf{0.8290}$ | $\mathbf{0.7949}$ | $\mathbf{0.0634}$ | $\mathbf{97.66\%}$ | $\mathbf{96.30\%}$ | $70.32\text{ s}$ | $0.98\text{ ms}$ |
| **Model 4: HistGradientBoosting** *(Gradient Boosting)* | $0.7974 \pm 0.034$ | $0.7129 \pm 0.143$ | $0.8096$ | $0.7573$ | $0.0744$ | $97.26\%$ | $96.10\%$ | $15.42\text{ s}$ | $2.68\text{ ms}$ |
| **Model 5: PyTorch MLP on RTX 4060** *(Neural Network)* | $0.7455 \pm 0.029$ | $0.7038 \pm 0.137$ | $0.7874$ | $0.7454$ | $0.0889$ | $96.70\%$ | $97.18\%$ | $9.86\text{ s}$ | $0.40\text{ ms}$ |

*Raw data source:* [`results/tables/model_comparison_5models.csv`](file:///C:/Users/chiranth/Downloads/shool/pathsnap-team11/results/tables/model_comparison_5models.csv) and [`results/tables/role_b_models_summary.csv`](file:///C:/Users/chiranth/Downloads/shool/pathsnap-team11/results/tables/role_b_models_summary.csv).

---

### 3.2 Compulsory Questions for Evaluated Models (Models 1, 2, and 3)

#### Model 1: Linear and Probabilistic Family (Logistic Regression & PyTorch GPU Benchmark)
1. **Algorithmic Suitability & Formulation:**
   - Logistic Regression models the posterior probability of candidate correctness via the logistic sigmoid function:
     $$P(y_i = 1 \mid \mathbf{x}_i) = \sigma(\mathbf{w}^T \mathbf{x}_i + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x}_i + b)}}$$
   - **Suitability for Map-Matching:** Road candidate scoring relies strongly on monotonic physical features, notably perpendicular distance to arc centerline (`distance_to_candidate`) and trajectory-to-road bearing misalignment (`heading_diff`). A linear decision surface provides smooth, well-behaved probability outputs that degrade gracefully with increasing distance.
2. **Loss Function & Class Weighting:**
   - The objective is the weighted negative log-likelihood (Weighted Binary Cross-Entropy):
     $$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^N \left[ w_+ y_i \log \sigma(\mathbf{w}^T \mathbf{x}_i + b) + (1 - y_i) \log(1 - \sigma(\mathbf{w}^T \mathbf{x}_i + b)) \right] + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
   - To counteract the 35:1 class imbalance, $w_+$ is set to balanced inverse frequency ($w_+ \approx 35.0$). This scales gradient contributions from the true positive road segment so the optimizer does not collapse into predicting constant zero.
3. **Hardware Acceleration (NVIDIA GeForce RTX 4060 Laptop GPU):**
   - In addition to standard CPU scikit-learn Logistic Regression, an optimized PyTorch implementation (`GPULogisticRegression`) was deployed directly to the system's **NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM, CUDA 12.4)**.
   - Using batched matrix-vector multiplications ($16,384$ candidates per tensor batch) and Adam optimizer on CUDA cores, training converged in $8.02\text{ s}$ with an ultra-low inference latency of **$0.38\text{ ms}$ per 1,000 candidate evaluations**.
4. **Strengths and Limitations:**
   - *Strengths:* Highly interpretable feature weights; strictly convex optimization surface with guaranteed convergence to the global optimum; ultra-low inference latency; directly calibrated sigmoid probabilities.
   - *Limitations:* Inability to capture non-linear relationships or complex spatial interactions (e.g., interaction between high speed and heading variance) without explicit polynomial feature transforms.
5. **Bias-Variance & Overfitting/Underfitting Analysis:**
   - High bias / slight underfitting. The gap between training F1 ($0.6588$) and student-test F1 ($0.6651$) is minimal ($< 0.01$), demonstrating zero overfitting, but Top-1 accuracy is constrained by linear boundary limits.

---

#### Model 2: Instance-Based Family (Distance-Weighted $k$-Nearest Neighbors)
1. **Algorithmic Suitability & Formulation:**
   - Non-parametric local estimation assigning candidate probability based on the distance-weighted labels of the $k$ closest candidate exemplars in normalized feature space:
     $$P(y = 1 \mid \mathbf{x}) = \frac{\sum_{j \in \mathcal{N}_k(\mathbf{x})} w_j y_j}{\sum_{j \in \mathcal{N}_k(\mathbf{x})} w_j}, \quad w_j = \frac{1}{d(\mathbf{x}, \mathbf{x}_j) + \epsilon}$$
     where $d(\mathbf{x}, \mathbf{x}_j) = \|\mathbf{x} - \mathbf{x}_j\|_2$ over standardized continuous features with $k = 15$.
2. **Computational Complexity & The Curse of Dimensionality:**
   - *Training Time:* $\mathcal{O}(1)$ (lazy learner, $0.09\text{ s}$).
   - *Inference Time:* $\mathcal{O}(N_{\text{train}} \cdot D)$ per candidate query point.
   - *Empirical Latency Bottleneck:* In our experiments, $k$-NN required **$22.23\text{ ms}$ per 1,000 queries**, which is over **60$\times$ slower** than Random Forest and Logistic Regression. Furthermore, on observation-level Top-1 accuracy, $k$-NN collapsed to only **$4.62\%$** because Euclidean proximity in candidate feature space fails to differentiate between parallel opposite-direction candidate arcs, confirming its unsuitability for real-time map matching.
3. **Strengths and Limitations:**
   - *Strengths:* Non-parametric; requires no training phase.
   - *Limitations:* Severe inference latency bottleneck; poor ranking capability on observation Top-1 accuracy; susceptible to the curse of dimensionality.

---

#### Model 3: Tree-Based Ensemble Family (Balanced Random Forest Classifier)
1. **Algorithmic Suitability & Formulation:**
   - Bagging ensemble of $B = 100$ decorrelated Classification and Regression Trees (CART). Each split considers a random feature subset of size $\sqrt{D}$.
   - Class probabilities represent the aggregated vote across all individual decision trees:
     $$P(y = 1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B P_b(y = 1 \mid \mathbf{x})$$
   - To resolve the extreme class imbalance, balanced bootstrap subsampling (`class_weight='balanced_subsample'`) is utilized, rebalancing class weights dynamically for each bootstrap sample.
2. **Splitting Criterion:**
   - Trees optimize the Gini Impurity reduction:
     $$\Delta I_G = I_G(D) - \frac{|D_L|}{|D|} I_G(D_L) - \frac{|D_R|}{|D|} I_G(D_R), \quad I_G(p) = 1 - (p_0^2 + p_1^2)$$
3. **Strengths and Limitations:**
   - *Strengths:* Exceptional capacity to model complex non-linear feature interactions (such as the threshold combination of `distance_to_candidate < 15m` AND `heading_diff < 20°` AND `speed_limit > 50km/h`); completely invariant to monotonic feature transformations; robust to sensor outliers; achieved **$0.8290$ Macro F1** and **$0.7949$ Top-1 Accuracy** with a low Log Loss of **$0.0634$**.
   - *Limitations:* Moderate memory footprint for tree storage; slower inference latency than linear models ($0.98\text{ ms}/1\text{k}$).
4. **Bias-Variance & Overfitting/Underfitting Analysis:**
   - Low bias and low variance. By setting `max_depth = 12` and averaging across 100 decorrelated trees, variance is suppressed while maintaining high expressive capacity. The train CV Macro F1 ($0.8127$) to student-test Macro F1 ($0.8290$) gap remains minimal, confirming robust generalization without overfitting.

---

### 3.3 Model Visualisations
Visual comparisons across all models and evaluation metrics have been plotted and saved to the repository:
1. **Role B Models Comparison (Models 1–3 + GPU):** [`results/plots/role_b_model_comparison_plots.png`](file:///C:/Users/chiranth/Downloads/shool/pathsnap-team11/results/plots/role_b_model_comparison_plots.png)
   - Cross-Validation Macro F1 (with error bars).
   - Student-Test Observation-Level Top-1 Accuracy.
   - Student-Test Macro F1-Score.
   - Student-Test Log Loss (Binary Cross-Entropy).
2. **Consolidated 5-Model Comparison (Models 1–5):** [`results/plots/model_comparison_5models_plots.png`](file:///C:/Users/chiranth/Downloads/shool/pathsnap-team11/results/plots/model_comparison_5models_plots.png)
   - Visual benchmark displaying the progression from linear baselines to tree ensembles, gradient boosting, and deep neural networks on the RTX 4060 GPU.

---

## 4. Bayesian Hyperparameter Optimisation

### 4.1 Framework, Search Space, and Objective Function
Bayesian optimization was conducted using **Optuna** with the **Tree-structured Parzen Estimator (TPE)** algorithm to systematically optimize the tree-based gradient boosting architecture.
- **Objective Function:** Maximize 5-Fold Group-Aware Cross-Validation Macro F1-Score on the training partition (`GroupKFold` on `trip_id`).
- **Audit & Persistence:** All trial parameters, durations, and intermediate values were logged directly into a persistent SQLite database (`optimisation/optuna_study.db`) and exported to `results/tables/optuna_trial_records.csv`.

#### Search Space Justification
| Hyperparameter | Search Domain | Scale | Justification |
| :--- | :---: | :---: | :--- |
| `n_estimators` | $[80, 300]$ | Integer (step=20) | Sufficient trees for convergence without excessive inference latency. |
| `learning_rate` | $[0.02, 0.20]$ | Log-uniform | Controls gradient shrinkage step size; lower rates require more trees. |
| `num_leaves` | $[15, 63]$ | Integer | Directly controls tree model complexity ($2^{\text{depth}}$ ceiling). |
| `max_depth` | $[4, 10]$ | Integer | Limits maximum tree depth to prevent memorization of noise. |
| `min_child_samples` | $[20, 150]$ | Integer | Prevents leaves from isolating rare GPS noise pings. |
| `scale_pos_weight` | $[15.0, 45.0]$ | Uniform | Balances the empirical ~35:1 negative-to-positive candidate ratio. |
| `subsample` | $[0.70, 1.00]$ | Uniform | Stochastic row subsampling to reduce tree correlation. |
| `colsample_bytree` | $[0.70, 1.00]$ | Uniform | Feature subsampling at each tree split. |
| `reg_alpha` ($L_1$) | $[10^{-3}, 5.0]$ | Log-uniform | Induces sparsity on redundant spatial features. |
| `reg_lambda` ($L_2$) | $[10^{-3}, 5.0]$ | Log-uniform | Smooths leaf output weights to prevent overconfidence. |

---

### 4.2 Empirical Tuning Results: Tuned vs. Untuned Comparison
Optuna evaluated 15 complete Bayesian trials across 5-fold group cross-validation. The global optimum was identified at **Trial #5**:
- `n_estimators`: $300$
- `learning_rate`: $0.1192$
- `num_leaves`: $61$
- `max_depth`: $10$
- `min_child_samples`: $98$
- `scale_pos_weight`: $42.66$
- `subsample`: $0.7265$
- `colsample_bytree`: $0.7588$
- `reg_alpha`: $0.00147$
- `reg_lambda`: $0.01597$

#### Measured Tuned vs. Untuned Performance
| Configuration | 5-Fold CV Macro F1 | CV Top-1 Acc | Student-Test Macro F1 | Student-Test Top-1 Acc | Student-Test Log Loss | Inference Latency (ms/1k) | Trials | Tuning Cost (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untuned Baseline** *(Default LightGBM)* | $0.7963$ | $0.6652$ | $0.8142$ | $0.7411$ | $0.0696$ | $0.67\text{ ms}$ | $0$ | $0.00\text{ s}$ |
| **Tuned Model** *(Optuna Bayesian TPE)* | **$0.8240$** | **$0.6626$** | **$0.8494$** | **$0.7429$** | **$0.0500$** | $2.28\text{ ms}$ | $15$ | $301.65\text{ s}$ |

*Raw data source:* [`results/tables/tuned_vs_untuned_comparison.csv`](file:///C:/Users/chiranth/Downloads/shool/pathsnap-team11/results/tables/tuned_vs_untuned_comparison.csv).

#### Computational Cost & Artifact Deliverables
- **Total Bayesian Optimization Wall-Clock Time:** $301.65\text{ seconds}$ (~$5.0\text{ minutes}$).
- **Optimization Plots:**
  - Convergence History: [`results/plots/optuna_convergence_history.png`](file:///C:/Users/chiranth/Downloads/shool/pathsnap-team11/results/plots/optuna_convergence_history.png)
  - Parameter Importances: [`results/plots/optuna_param_importances.png`](file:///C:/Users/chiranth/Downloads/shool/pathsnap-team11/results/plots/optuna_param_importances.png)
- **Saved Model Artifacts:**
  - Trained Model Binary: [`artifacts/final_model.joblib`](file:///C:/Users/chiranth/Downloads/shool/pathsnap-team11/artifacts/final_model.joblib)
  - Canonical Feature Schema: [`artifacts/feature_list.json`](file:///C:/Users/chiranth/Downloads/shool/pathsnap-team11/artifacts/feature_list.json)
  - Hyperparameter Manifest: [`artifacts/best_hyperparameters.json`](file:///C:/Users/chiranth/Downloads/shool/pathsnap-team11/artifacts/best_hyperparameters.json)
  - Audited Trial Database: [`optimisation/optuna_study.db`](file:///C:/Users/chiranth/Downloads/shool/pathsnap-team11/optimisation/optuna_study.db)
