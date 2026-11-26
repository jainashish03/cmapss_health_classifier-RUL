# ✈️ Turbofan Engine Health-Stage Classification (NASA C-MAPSS)

Classify the **health stage** of an aircraft turbofan engine — `Healthy → Degrading → Warning → Critical` —
from multivariate sensor time series of the NASA C-MAPSS run-to-failure dataset.

C-MAPSS has no health label; it only provides run-to-failure trajectories. This project therefore
**derives the target** from Remaining Useful Life (RUL) and trains a classifier on top of causal
time-series features.

## 1. Problem definition

| Stage | Default rule (RUL in cycles) |
|-------|------------------------------|
| 0 Healthy   | RUL > 125 |
| 1 Degrading | 75 < RUL ≤ 125 |
| 2 Warning   | 30 < RUL ≤ 75 |
| 3 Critical  | RUL ≤ 30 |

* **Train engines** run to failure → `RUL = last_cycle − cycle`.
* **Test engines** are cut before failure → `RUL = RUL_FD00X[engine] + last_cycle − cycle`, so *every* test row gets a true label.
* Thresholds, number of stages and names are editable in `config.yaml`. An alternative labelling by
  *fraction of life consumed* is available (`labeling.method: life_fraction`).

## 2. Pipeline

```
raw 26 columns
  └─ ConditionScaler   k-means on 3 op-settings → z-score each sensor per operating regime (FD002/FD004: 6 regimes),
  │                    drop constant sensors (fit on train only)
  └─ FeatureBuilder    per engine & strictly causal: normalised sensors + rolling mean / std / delta
  │                    over windows 10, 30, 50, 80 cycles + cycle counter  (~200 features)
  └─ Classifier        logistic regression | random forest | hist. gradient boosting | xgboost (optional)
```

Design decisions that matter:

* **No leakage.** Cross-validation is a `GroupKFold` **by engine**; the scaler and models are re-fitted inside every fold.
* **Causal features.** A cycle's features only use that cycle and earlier ones (unit-tested), so the model can run online.
* **Long windows win.** Going from windows (5, 15) to (10, 30, 50, 80) raised CV macro-F1 on FD001 from 0.72 to 0.83 –
  health is a slow trend hidden in noisy sensors.
* **Operating-condition normalisation** is what makes FD002/FD004 workable at all.
* **Ordinal-aware metrics.** Besides accuracy / macro-F1 we report *adjacent accuracy* (prediction within one stage), because
  confusing Warning with Critical is less severe than confusing Healthy with Critical.

## 3. Quick start

```bash
pip install -r requirements.txt
# data is already in data/raw/ (copy the Kaggle .txt files there if you start from scratch)

python -m cmapss_health.train --subset FD001 --compare      # CV model selection → train → evaluate → save
python -m cmapss_health.train --subset FD002 --model logistic_regression   # skip the comparison
./run_all.sh                                                # all four subsets

python -m cmapss_health.predict --subset FD001 --input data/raw/test_FD001.txt --output reports/predictions.csv
streamlit run app/streamlit_app.py                          # fleet dashboard
python -m pytest -q tests                                   # unit + end-to-end tests (synthetic data)
```

Useful flags: `--models a b c` and `--folds N` (for `--compare`), `--synthetic` (generate a fake C-MAPSS-format
dataset for smoke tests), `--out path.joblib`.

## 4. Results

Selected model per subset by 5-fold (3-fold for FD002/FD004) grouped CV macro-F1, then evaluated **once** on the
official test set. "Last cycle" is the official C-MAPSS protocol (one row per test engine — small *n*, noisy);
"all cycles" scores every test row.

| Subset | Model | CV macro-F1 | Test acc (last cycle) | Test macro-F1 (last cycle) | Test acc (all cycles) | Test macro-F1 (all cycles) | Adjacent acc (all cycles) |
|---|---|---|---|---|---|---|---|
| FD001 | logistic_regression | 0.830 | 0.810 | 0.787 | 0.825 | 0.828 | 0.999 |
| FD002 | logistic_regression | 0.814 | 0.788 | 0.795 | 0.767 | 0.774 | 0.997 |
| FD003 | logistic_regression | 0.836 | 0.830 | 0.822 | 0.836 | 0.809 | 0.997 |
| FD004 | logistic_regression | 0.777 | 0.746 | 0.754 | 0.831 | 0.771 | 0.994 |

Model comparison inside CV (macro-F1): FD001 logreg 0.830 / RF 0.781 / HGB 0.787; FD003 logreg 0.836 / RF 0.806 / HGB 0.811;
FD002 logreg 0.814 / HGB 0.751; FD004 logreg 0.777 / HGB 0.774 (FD002/FD004 compared with 3 folds and without random forest to save time).
Plain logistic regression wins everywhere: with long-window trend features the stage boundaries are close to linear.

Full numbers: `reports/*_metrics.json`, `reports/*_model_comparison.csv`; plots in `reports/figures/`
(confusion matrices, feature importance, class balance, sensor trends, per-engine probability timeline).

**Reading the results.** Most errors are between *neighbouring* stages near a threshold — an engine at RUL 76 and one at
RUL 74 look alike by construction, and each engine starts with an unknown initial wear. The `Critical` stage — the one that
matters operationally — is detected best.

## 5. Project layout

```
config.yaml            all settings (labelling, windows, models, CV)
cmapss_health/
  data.py              loading C-MAPSS txt files
  labeling.py          RUL → health stage
  preprocessing.py     operating-condition-aware scaling
  features.py          causal rolling features
  modeling.py          model factory
  pipeline.py          HealthPipeline (fit / predict / save / load)
  evaluate.py          metrics + plots
  train.py  predict.py CLI entry points
  synthetic.py         fake data generator for tests
app/streamlit_app.py   dashboard
tests/                 pytest suite
data/raw/              dataset            models/   trained pipelines
reports/               metrics, predictions, figures
```

## 6. Ideas for extension

* Sequence models (LSTM / 1D-CNN / Transformer) on sliding windows of normalised sensors.
* Piece-wise-linear RUL cap (≈125) and a joint RUL-regression + stage-classification model.
* Probability calibration + hysteresis/smoothing of stage predictions over time to avoid flicker.
* Cost-sensitive training: penalise predicting "Healthy" for a truly "Critical" engine more heavily.
* Predict the *next* stage / time-to-stage-change instead of only the current one.

## 7. Reference

A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*,
PHM08 (PDF included in `data/raw/`).
