# ✈️ Turbofan Engine Health-Stage Classification System
## NASA C-MAPSS Fleet Predictive Maintenance Documentation

---

## Table of Contents
1. [Executive Summary & Problem Formulation](#1-executive-summary--problem-formulation)
2. [Dataset Specification & Operating Regimes](#2-dataset-specification--operating-regimes)
3. [Health Stage Derivation & Labeling Strategy](#3-health-stage-derivation--labeling-strategy)
4. [System Architecture & Pipeline Design](#4-system-architecture--pipeline-design)
5. [Feature Engineering & Causal Constraints](#5-feature-engineering--causal-constraints)
6. [Machine Learning Models & Cross-Validation](#6-machine-learning-models--cross-validation)
7. [Empirical Evaluation & Performance Benchmarks](#7-empirical-evaluation--performance-benchmarks)
8. [Interactive Fleet Dashboard (Streamlit)](#8-interactive-fleet-dashboard-streamlit)
9. [CLI & API Reference Guide](#9-cli--api-reference-guide)
10. [Configuration Reference (`config.yaml`)](#10-configuration-reference-configyaml)
11. [Testing & Quality Assurance](#11-testing--quality-assurance)
12. [Production Deployment & Operational Best Practices](#12-production-deployment--operational-best-practices)

---

## 1. Executive Summary & Problem Formulation

Predictive maintenance of modern commercial aircraft turbofan engines requires continuous health state assessment from high-frequency, multivariate sensor telemetry. Traditional approaches to the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset formulate the problem as continuous Remaining Useful Life (RUL) regression. However, operational maintenance dispatch teams require discrete, actionable health stages to schedule inspections, supply-chain logistics, and component overhauls.

This project implements an end-to-end, production-grade machine learning system that transforms continuous run-to-failure sensor streams into actionable ordinal health stages:
$$\text{Healthy} \longrightarrow \text{Degrading} \longrightarrow \text{Warning} \longrightarrow \text{Critical}$$

### Core Design Principles
* **Strict Temporal Causality:** Feature extraction utilizes only current and historical cycles ($t' \le t$). Unit-tested causal transformations ensure zero lookahead bias during online streaming inference.
* **Leakage-Free Validation:** Grouped $K$-fold cross-validation by engine identifier (`unit`) ensures cycles from the same engine never contaminate validation folds.
* **Regime-Aware Normalization:** Unsupervised $k$-means clustering on operating parameters normalizes sensors within distinct flight conditions, enabling effective modeling of complex multi-regime subsets (`FD002` and `FD004`).
* **Ordinal-Aware Evaluation:** Standard accuracy and macro-F1 metrics are augmented with *Adjacent Accuracy*, accounting for the ordinal nature of degradation transitions.

---

## 2. Dataset Specification & Operating Regimes

The NASA C-MAPSS dataset consists of four distinct sub-datasets (`FD001` through `FD004`), simulating varying environmental conditions, operating regimes, and degradation modes.

| Dataset | Operating Conditions | Fault Modes | Train Units | Test Units | Condition Regimes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FD001** | 1 (Sea Level) | 1 (High Pressure Compressor) | 100 | 100 | Steady state (1 cluster) |
| **FD002** | 6 Operating Regimes | 1 (High Pressure Compressor) | 260 | 259 | Dynamic (Altitude 0–42k ft, Mach 0–0.84) |
| **FD003** | 1 (Sea Level) | 2 (HPC + Fan Degradation) | 100 | 100 | Steady state (1 cluster) |
| **FD004** | 6 Operating Regimes | 2 (HPC + Fan Degradation) | 249 | 248 | Dynamic (Multi-regime + Multi-fault) |

### Raw Feature Schema (26 Columns)
Each row represents an operational cycle of a specific turbofan unit:
1. `unit`: Engine serial identification number (integer).
2. `cycle`: Operational cycle index ($1, 2, \dots, T$).
3. `op1`, `op2`, `op3`: Operational settings (Altitude, Mach number, Throttle resolver angle).
4. `s1` to `s21`: 21 internal sensor measurements (Temperatures, Pressures, Rotor Speeds, Flow Ratios, Bleed Enthalpy).

---

## 3. Health Stage Derivation & Labeling Strategy

Because C-MAPSS provides only raw run-to-failure trajectories without discrete state annotations, targets are mathematically derived.

### 3.1 Remaining Useful Life (RUL) Ground Truth
* **Training Fleet:** Engines operate until functional failure:
  $$\text{RUL}_{\text{train}}(u, t) = T_{\max}(u) - t$$
* **Testing Fleet:** Time series are truncated prior to failure. True final RUL values are supplied via `RUL_FD00X.txt`:
  $$\text{RUL}_{\text{test}}(u, t) = \text{RUL}_{\text{ground\_truth}}(u) + T_{\max}(u) - t$$

### 3.2 Discrete Stage Mapping
The system maps continuous RUL into four distinct ordinal stages via configurable thresholding:

| Stage ID | Stage Label | Operational Definition | Default RUL Boundary | Maintenance Action |
| :---: | :--- | :--- | :--- | :--- |
| **0** | **Healthy** | Nominal baseline performance | $\text{RUL} > 125$ cycles | Routine pre-flight check |
| **1** | **Degrading** | Early onset wear detected | $75 < \text{RUL} \le 125$ cycles | Schedule depot inspection |
| **2** | **Warning** | Accelerated component degradation | $30 < \text{RUL} \le 75$ cycles | Order replacement parts |
| **3** | **Critical** | Impending failure / overhaul needed | $\text{RUL} \le 30$ cycles | Ground aircraft; immediate overhaul |

*Alternative Labeling:* A life-fraction option is also supported ($\tau = \frac{t}{t + \text{RUL}}$) using percentage boundaries (e.g., $50\%$, $75\%$, $90\%$).

---

## 4. System Architecture & Pipeline Design

The processing pipeline is encapsulated within the `HealthPipeline` class, providing a self-contained, serializable artifact (`.joblib`):

```
+--------------------------------------------------------------------------------+
|                             RAW TELEMETRY INPUT                                |
|                 (unit, cycle, op1, op2, op3, s1, ..., s21)                     |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                        STAGE 1: CONDITION SCALER                               |
|   - Multi-regime datasets (FD002/FD004): k-means clustering (k=6) on ops       |
|   - Per-regime sensor z-score standardization: z = (s - mu_regime) / sigma     |
|   - Constant sensor pruning (variance threshold = 1e-6)                        |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                        STAGE 2: CAUSAL FEATURE BUILDER                         |
|   - Strictly causal rolling windows: W = [10, 30, 50, 80] cycles               |
|   - Rolling Mean: mu_W(t) = (1/W) sum_{k=0}^{W-1} s(t - k)                     |
|   - Rolling Std:  sigma_W(t)                                                   |
|   - Rolling Delta: Delta_W(t) = s(t) - s(t - W + 1)                            |
|   - Engine operational cycle counter                                           |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                        STAGE 3: CLASSIFIER / INFERENCE                         |
|   - Standard Scaler + Regularized Logistic Regression (Balanced Weights)       |
|   - Multiclass Softmax Probabilities: [p_Healthy, p_Degrading, p_Warn, p_Crit] |
|   - Discrete Argmax Prediction + Risk Metrics                                  |
+--------------------------------------------------------------------------------+
```

---

## 5. Feature Engineering & Causal Constraints

### 5.1 Operating Condition Normalization (`ConditionScaler`)
In multi-regime datasets (`FD002`, `FD004`), flight conditions dominate sensor variances by several orders of magnitude over mechanical wear. Applying a global standard scaler obscures degradation trends.
* The `ConditionScaler` clusters the 3 operational settings using $k$-means ($k=6$).
* For each cluster $c \in \{1, \dots, 6\}$, sensor means $\mu_{s, c}$ and standard deviations $\sigma_{s, c}$ are calculated strictly on training data.
* Low-variance sensors ($\sigma_{s, c} < 10^{-6}$) are eliminated.

### 5.2 Multi-Scale Rolling Statistics (`FeatureBuilder`)
Engine wear develops over hundreds of flight cycles. High-frequency sensor noise is filtered by expanding features across four window sizes:
$$W \in \{10, 30, 50, 80\} \text{ cycles}$$

For each surviving sensor $s$:
1. **Instantaneous Value:** Normalized sensor reading at cycle $t$.
2. **Rolling Mean ($\mu_{s, W}$):** Long-term smoothed operating trajectory.
3. **Rolling Standard Deviation ($\sigma_{s, W}$):** Telemetry volatility indicator.
4. **Window Delta ($\Delta_{s, W}$):** Direct measure of degradation slope over window $W$.
5. **Cycle Counter:** Current operating age of the engine.

This yields approximately $180$ to $200$ engineered features, providing robust linear separation across wear states.

---

## 6. Machine Learning Models & Cross-Validation

### 6.1 Model Candidates
The framework implements and benchmarks four model architectures:
1. **Regularized Logistic Regression (Production Standard):**
   * Multi-class multinomial formulation with $L_2$ regularization ($C=1.0$).
   * Inverse frequency class weighting (`class_weight='balanced'`).
   * Scaled via standard scaling pipeline.
2. **Random Forest Classifier:**
   * 200 estimators, minimum leaf size 3, balanced sub-sampling.
3. **Histogram-Based Gradient Boosting (`HistGradientBoostingClassifier`):**
   * 300 iterations, learning rate 0.06, maximum 31 leaf nodes, balanced sample weights.
4. **XGBoost Classifier (Optional):**
   * 400 trees, learning rate 0.05, max depth 6, histogram tree method.

### 6.2 GroupKFold Cross-Validation Strategy
To prevent temporal and engine-level data leakage, validation is structured via `GroupKFold` grouped by `unit`:
* 5 folds for `FD001` and `FD003`.
* 3 folds for `FD002` and `FD004` (due to computational size).
* The `ConditionScaler`, `FeatureBuilder`, and classifier are re-fitted from scratch inside every fold.

---

## 7. Empirical Evaluation & Performance Benchmarks

### 7.1 Cross-Validation Model Selection
Cross-validation macro-F1 scores across model architectures:

| Subset | Logistic Regression | Random Forest | HistGradientBoosting | Selected Model |
| :--- | :---: | :---: | :---: | :---: |
| **FD001** | **0.830** | 0.781 | 0.787 | `logistic_regression` |
| **FD002** | **0.814** | — | 0.751 | `logistic_regression` |
| **FD003** | **0.836** | 0.806 | 0.811 | `logistic_regression` |
| **FD004** | **0.777** | — | 0.774 | `logistic_regression` |

*Insight:* Logistic Regression consistently outperforms non-linear tree ensembles. When long-window causal trend features are introduced, health stage boundaries become quasi-linear, while linear models remain immune to tree over-partitioning on noisy cycles.

### 7.2 Independent Test Set Benchmark

| Dataset | Model | CV Macro-F1 | Test Acc (Last Cycle) | Test Macro-F1 (Last Cycle) | Test Acc (All Cycles) | Test Macro-F1 (All Cycles) | Adjacent Acc (All Cycles) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** | Logistic Regression | **0.830** | 0.810 | 0.787 | 0.825 | 0.828 | **0.999** |
| **FD002** | Logistic Regression | **0.814** | 0.788 | 0.795 | 0.767 | 0.774 | **0.997** |
| **FD003** | Logistic Regression | **0.836** | 0.830 | 0.822 | 0.836 | 0.809 | **0.997** |
| **FD004** | Logistic Regression | **0.777** | 0.746 | 0.754 | 0.831 | 0.771 | **0.994** |

*Adjacent Accuracy ($\ge 99.4\%$):* Over $99.4\%$ of model misclassifications occur strictly between adjacent stages (e.g., Warning vs. Critical near cycle boundary 30). Catastrophic misclassifications (e.g., classifying a Critical unit as Healthy) are effectively zero.

---

## 8. Interactive Fleet Dashboard (Streamlit)

The web dashboard ([`app/streamlit_app.py`](file:///Users/ashish/Downloads/cmapss-health-classifier/app/streamlit_app.py)) provides operational intelligence for maintenance control centers:

### Dashboard Capabilities
1. **Fleet Health Status Overview:**
   * Executive KPI cards: Total Units, Critical Count, Warning Count, Operational Health Index.
   * Fleet Stage Distribution donut chart.
   * Priority Maintenance Table sorted by critical probability and cycle age.
2. **Single Engine Deep Dive:**
   * Selectable engine unit dropdown.
   * Temporal Health Stage Trajectory: probability timeline over cycles ($p_{\text{Healthy}}, \dots, p_{\text{Critical}}$).
   * Multi-sensor comparative degradation traces with interactive zoom.
3. **Sensor Analytics & Operating Regimes:**
   * 2D/3D scatter plots of operational regimes (Altitude vs. Mach vs. TRA).
   * Feature correlation heatmaps and condition-specific sensor drift.
4. **Model Performance & Explainability:**
   * Interactive Confusion Matrices (Last Cycle and All Cycles).
   * Global Feature Importance rankings (Permutation Importance / Coefficients).
   * Direct CSV export of fleet predictions.

---

## 9. CLI & API Reference Guide

### 9.1 Command-Line Interface

#### Training & Comparison
```bash
# Train on FD001 with default model and generate evaluation reports
python3 -m cmapss_health.train --subset FD001

# Run full cross-validation comparison across candidate models
python3 -m cmapss_health.train --subset FD001 --compare

# Train on synthetic data for rapid CI/CD smoke testing
python3 -m cmapss_health.train --synthetic --subset FD001

# Train all subsets sequentially
./run_all.sh
```

#### Batch Inference / Prediction
```bash
python3 -m cmapss_health.predict \
  --subset FD001 \
  --input data/raw/test_FD001.txt \
  --output reports/predictions.csv
```

#### Launching the Dashboard
```bash
python3 -m streamlit run app/streamlit_app.py --server.port 8501
```

### 9.2 Python API (`HealthPipeline`)

```python
from cmapss_health.config import load_config
from cmapss_health.pipeline import HealthPipeline
from cmapss_health.data import read_engine_file

# 1. Load configuration and model
cfg = load_config()
pipeline = HealthPipeline.load("models/health_model_FD001.joblib")

# 2. Ingest sensor data
df = read_engine_file("data/raw/test_FD001.txt")

# 3. Predict per-cycle health states
cycle_predictions = pipeline.predict_frame(df)

# 4. Predict latest status per engine
fleet_status = pipeline.predict_last(df)
print(fleet_status[["unit", "cycle", "pred_stage", "p_Critical"]])
```

---

## 10. Configuration Reference (`config.yaml`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `seed` | `int` | `42` | Random seed for reproducibility across splits and models |
| `data.dir` | `string` | `data/raw` | Path containing raw C-MAPSS text files |
| `data.subset` | `string` | `FD001` | Active dataset identifier (`FD001`, `FD002`, `FD003`, `FD004`) |
| `labeling.method` | `string` | `rul` | Labeling method: `rul` or `life_fraction` |
| `labeling.rul_thresholds`| `list[int]` | `[30, 75, 125]`| Thresholds separating Critical ($\le 30$), Warning ($\le 75$), Degrading ($\le 125$), Healthy ($> 125$) |
| `labeling.stage_names` | `list[str]` | `[Healthy, Degrading, Warning, Critical]` | Ordinal category names |
| `preprocessing.n_conditions` | `string/int` | `auto` | Operating regimes count (`auto` detects 1 or 6 regimes) |
| `features.windows` | `list[int]` | `[10, 30, 50, 80]` | Rolling window cycles for mean, std, and delta calculations |
| `features.include_cycle` | `bool` | `true` | Include raw cycle counter as a feature |
| `model.name` | `string` | `logistic_regression` | Active classification algorithm |
| `model.candidates` | `list[str]` | `[random_forest, hist_gradient_boosting, logistic_regression]` | Models evaluated during `--compare` |
| `cv.n_splits` | `int` | `5` | Number of GroupKFold partitions |

---

## 11. Testing & Quality Assurance

Unit and regression tests are implemented under `tests/`:
* `tests/test_pipeline.py`: Comprehensive end-to-end tests covering data loading, causal rolling transforms, model training, persistence, and inference integrity.
* **Strict Causal Feature Verification:** Ensures that modifying future cycles ($t' > t$) has zero numerical effect on features computed at cycle $t$.

Run the automated test suite:
```bash
python3 -m pytest -q tests
```

---

## 12. Production Deployment & Operational Best Practices

1. **Model Persistence:** Models are serialized using `joblib`. Preprocessor parameters (cluster centroids, regime means, variance masks) and classifier coefficients are stored in a single immutable artifact.
2. **Online Streaming:** When streaming data arrives per flight cycle, maintain a rolling memory buffer of 80 cycles per active engine to compute all temporal features without querying historical cold storage.
3. **Drift Detection:** Monitor the distribution of predicted stages over 30-day windows. Shifts in the ratio of Healthy to Degrading units indicate seasonal flight pattern variations or sensor calibration drift.
4. **Maintenance Action Protocol:**
   * Any transition into `Critical` with $p_{\text{Critical}} > 0.70$ must trigger an automated maintenance dispatch ticket.
   * Engines residing in `Warning` for $> 15$ cycles should undergo boroscopic inspection.
