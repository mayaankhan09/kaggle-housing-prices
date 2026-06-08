# Kaggle Housing Prices — ML Pipeline

> **Top ~30% leaderboard** | Scikit-learn · XGBoost · GridSearchCV

Predicts residential sale prices using a full ML pipeline: EDA → missing-value imputation → feature engineering (20+ features) → model comparison → XGBoost hyperparameter tuning.

---

## Pipeline Architecture

```
Data Ingestion (train.csv, test.csv)
        ↓
Exploratory Data Analysis
  • Target distribution (log transform)
  • Correlation heatmap
  • Missing value map
  • Outlier detection
        ↓
Preprocessing
  • Semantic NaN imputation (categorical)
  • Group-median imputation (LotFrontage)
  • Ordinal encoding (quality columns)
  • One-hot encoding (nominal columns)
        ↓
Feature Engineering  (20+ new features)
  • TotalSF, HouseAge, RemodAge
  • TotalBathrooms, HasPool, HasGarage
  • Quality interaction terms (QualSF, QualAge)
  • Skewed feature log transforms
        ↓
Feature Scaling
  • RobustScaler (fit on train only)
        ↓
Model Selection  (5-fold cross-validation)
  • Linear Regression  — baseline
  • Ridge Regression
  • Random Forest
  • Gradient Boosting
  • XGBoost            ← best
        ↓
Hyperparameter Tuning
  • GridSearchCV on XGBoost
  • ~18% RMSE reduction over baseline
        ↓
Final Submission
  • Predict on test set → expm1() → submission.csv
```

---

## Benchmark Results

| Model | CV RMSE (log) | CV R² | Notes |
|---|---|---|---|
| Linear Regression | — | — | Baseline |
| Ridge | — | — | |
| Random Forest | — | — | |
| Gradient Boosting | — | — | |
| **XGBoost (tuned)** | **—** | **—** | **Best** |

*Table will be filled after Phase 5–6.*

---

## Key Features Engineered

| Feature | Formula | Rationale |
|---|---|---|
| `TotalSF` | BsmtSF + 1stFlrSF + 2ndFlrSF | Total living area |
| `HouseAge` | YrSold − YearBuilt | Age at sale |
| `RemodAge` | YrSold − YearRemodAdd | Remodel recency |
| `TotalBathrooms` | FullBath + 0.5×Half + Bsmt baths | Composite bathroom count |
| `QualSF` | OverallQual × TotalSF | Quality-weighted area |
| `HasGarage` | GarageArea > 0 | Binary flag |
| `HasPool` | PoolArea > 0 | Binary flag |
| `HasFireplace` | Fireplaces > 0 | Binary flag |

---

## EDA Visualisations

*(Screenshots added after Phase 2)*

- `target_distribution.png` — raw vs log-transformed SalePrice
- `correlation_heatmap.png` — top correlated features
- `missing_values.png` — missing value map
- `feature_importances.png` — XGBoost top 20 features

---

## Leaderboard

*(Screenshot added after Phase 7)*

Rank: — / ~4000+  
RMSE: —

---

## How to Reproduce

```bash
git clone https://github.com/<your-username>/kaggle-housing-prices
cd kaggle-housing-prices
pip install -r requirements.txt
jupyter notebook notebooks/housing_pipeline.ipynb
```

Run all cells top to bottom. Requires `kaggle.json` in `~/.kaggle/`.

---

## Tech Stack

- Python 3.10
- pandas, numpy, scipy
- scikit-learn (preprocessing, models, cross-validation)
- XGBoost
- matplotlib, seaborn, missingno
