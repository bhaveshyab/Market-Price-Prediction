# AI-Assisted Market Price Prediction for Small-Scale Farmers

This repository contains an ML pipeline for learning from agricultural market price history across **multiple crops and markets**, plus a simple Streamlit interface.

## What it predicts

The model estimates the **next available recorded Modal_Price** for a selected State, District, Market, Commodity, Variety and Grade series. Market records are not necessarily daily, so this is not described as an exact 7-day forecast.

## Dataset

The training script accepts CSV or Excel data with these columns:

- State, District, Market, Commodity, Variety, Grade
- Arrival_Date
- Modal_Price

By default, it looks for `2025 cleaned.xlsx` in the repository root. You can pass another dataset path. Keep an untouched backup of your original data. The pipeline excludes rows with invalid dates or non-positive modal prices and aggregates duplicate series/date records by their mean modal price. It does not delete records merely because Min_Price and Max_Price look unusual.

## Setup in VS Code (Windows PowerShell)

Open the repository folder in VS Code, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use this in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Train and evaluate on the full dataset

```powershell
python train_multi_crop.py --input "2025 cleaned.xlsx"
```

For a CSV:

```powershell
python train_multi_crop.py --input "data\market_prices_all_crops.csv"
```

Generated files are saved in `outputs/`:

- `multi_crop_xgboost.joblib` — trained preprocessing + XGBoost pipeline
- `evaluation_metrics.csv` — MAE, RMSE and R² for a persistence baseline and XGBoost
- `test_predictions.csv` — chronological test predictions
- `feature_importance.csv` — model feature-importance report

The test set is later in time than the training set. Review the metrics; do not assume the model is accurate until it has run successfully on your dataset.

## Launch the interface

Train the model first, then run:

```powershell
streamlit run app.py
```

Choose crop, state, district, market, variety and grade to inspect historical prices and estimate the next recorded modal price.

## Important limitations

- The model is trained on the dataset supplied; missing future data, new markets and rare crop/variety combinations may reduce reliability.
- Modal prices are in the source dataset's units/currency; verify the official source's units before presenting prices to farmers.
- A one-year dataset may not represent seasonality across multiple years.
- Predictions are decision support, not guaranteed prices. Evaluate errors by crop and market before relying on them.
