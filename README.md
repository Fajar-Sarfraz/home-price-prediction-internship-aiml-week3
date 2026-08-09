# House Price Prediction — ASL Internship Week 3 (AI/ML Track)

## Project Overview
This project builds an end-to-end supervised machine learning pipeline to predict 
residential house prices in the Seattle, WA area using a public housing dataset. 
It covers data cleaning, train/test splitting, preprocessing with scikit-learn 
pipelines, model training, evaluation, and comparison — following the standard 
supervised learning workflow.

**Problem type:** Regression (predicting a continuous target: `price`)

## Dataset
- **Source:** Kaggle — House Price Prediction dataset (`modified_data.csv`)
- **Rows:** 4,600 (original) → 4,462 after cleaning
- **Target column:** `price`
- **Features:** bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, 
  view, condition, sqft_above, sqft_basement, yr_built, yr_renovated, city, sale_year

## Setup Instructions
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `week3_house_price_prediction.ipynb` in Jupyter or Google Colab
4. Run all cells in order (Runtime → Run all in Colab)

## Project / Folder Structure
```
asl-internship-aiml-week3-<yourname>/
├── README.md
├── requirements.txt
├── data/
│   └── modified_data.csv
├── notebooks/
│   └── week3_house_price_prediction.ipynb
├── models/
│   └── house_price_model.joblib
├── outputs/
│   └── predicted_vs_actual.png
├── report/
│   └── week3_report.pdf
└── screenshots/
    ├── data_quality_check.png
    ├── model_comparison.png
    ├── predicted_vs_actual.png
    └── overfitting_check.png
```

## Data Cleaning & Quality Decisions
- **Removed `price_per_sqft`** — this column was directly derived from the 
  target (`price / sqft_living`) and would have caused data leakage.
- **Removed `street`** — too high cardinality (nearly unique per row) to be 
  a useful categorical feature.
- **Removed `statezip`** — redundant with `city`, and would have added many 
  low-value categorical columns.
- **Dropped 49 rows** with `price == 0` (invalid/placeholder data).
- **Extracted `sale_year`** from the `date` column, then dropped the raw date.
- **Removed 89 outlier rows** with price above ~$1.65M (IQR method, 3× IQR 
  upper bound) — these were distorting model training and causing an unstable 
  train/test split (see Challenges section in report).
- No missing values were present in any remaining column.

## Preprocessing Pipeline
Built using `scikit-learn`'s `ColumnTransformer` and `Pipeline`:
- **Numeric features** (13 columns): scaled with `StandardScaler`
- **Categorical feature** (`city`, 44 unique values): encoded with `OneHotEncoder(handle_unknown='ignore')`
- Preprocessing is fit only on training data (inside the Pipeline) to prevent data leakage.

## Models Trained
| Model | Description |
|---|---|
| Linear Regression | Baseline model |
| Random Forest (default) | Comparative model, `n_estimators=100` |
| Random Forest (constrained) | Final model — `max_depth=10, min_samples_leaf=5` to reduce overfitting |

## Results

| Model | MAE | RMSE | Test R² | Train R² | Train/Test Gap |
|---|---|---|---|---|---|
| Linear Regression | $100,622 | $147,616 | 0.676 | 0.711 | 0.035 |
| Random Forest (default) | $97,795 | $147,805 | 0.675 | 0.957 | 0.282 |
| **Random Forest (constrained)** | $103,820 | $151,704 | 0.658 | 0.799 | 0.141 |

**Final model chosen: Random Forest (constrained)** — it offers the best 
balance between predictive accuracy and generalization, with a much smaller 
overfitting gap than the default Random Forest, at only a small cost to raw 
test accuracy.

## Features Implemented
- ✅ Public dataset with 500+ rows and a clear regression target
- ✅ Data quality check documented (missing values, leakage, outliers)
- ✅ Train/test split performed before preprocessing
- ✅ ColumnTransformer + Pipeline preprocessing (no data leakage)
- ✅ Baseline model (Linear Regression) + comparative model (Random Forest)
- ✅ Evaluated with MAE, RMSE, and R² (3 metrics)
- ✅ Predicted-vs-actual visualization
- ✅ Model comparison table with reasoning for final choice
- ✅ Final model saved with `joblib`
- ✅ Overfitting identified and addressed (train/test R² gap analysis)

## How to Reproduce
```python
import joblib
model = joblib.load("models/house_price_model.joblib")
predictions = model.predict(X_new)
```

## screenshot

## Project Screenshot

![Project Screenshot](https://raw.githubusercontent.com/Fajar-Sarfraz/home-price-prediction-internship-aiml-week3/4e18e8e98ed556d238c1c71b5c547b9c1a46afee/week%203%20sreenshots/Screenshot%20%2870%29.png)
