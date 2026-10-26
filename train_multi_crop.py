"""Train a multi-crop, multi-market next-observation modal-price model.

Usage:
  python train_multi_crop.py --input "2025 cleaned.xlsx"
  python train_multi_crop.py --input data/market_prices_all_crops.csv

The target is the next available recorded modal price within the same
state/district/market/commodity/variety/grade series. It is NOT guaranteed
to be exactly seven calendar days later.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

GROUPS = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
REQUIRED = GROUPS + ["Arrival_Date", "Modal_Price"]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPass the dataset path using --input."
        )
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError("Input must be a .csv, .xlsx, or .xls file.")

    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Found columns: {list(frame.columns)}"
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="2025 cleaned.xlsx",
                        help="Path to full dataset (.csv or .xlsx)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Folder for reports and trained model")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_data(Path(args.input))
    print(f"Loaded rows: {len(raw):,}")
    print(f"Crop names before cleaning: {raw['Commodity'].nunique(dropna=True):,}")

    df = raw.copy()
    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")

    for col in GROUPS:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA).fillna("Unknown")

    # Keep only rows with a valid date and a positive target price.
    # Do not use Min_Price/Max_Price as predictors because their availability
    # at the time of a future prediction may not be known.
    before = len(df)
    df = df.dropna(subset=["Arrival_Date", "Modal_Price"])
    df = df[df["Modal_Price"] > 0].copy()
    print(f"Rows excluded for invalid date/target price: {before - len(df):,}")

    # Resolve multiple records for the same series and date by averaging
    # modal prices. This is explicit aggregation, not arbitrary row deletion.
    df = (
        df.groupby(GROUPS + ["Arrival_Date"], as_index=False, dropna=False)
        .agg(Modal_Price=("Modal_Price", "mean"))
    )
    df = df.sort_values(GROUPS + ["Arrival_Date"]).reset_index(drop=True)

    grouped = df.groupby(GROUPS, dropna=False)["Modal_Price"]
    df["Previous_Price"] = grouped.shift(1)
    df["Rolling_Mean_3"] = grouped.transform(
        lambda s: s.shift(1).rolling(3, min_periods=3).mean()
    )
    df["Rolling_Mean_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=7).mean()
    )
    # The label is the next recorded observation in this exact series.
    df["Target_Next_Observed_Price"] = grouped.shift(-1)
    df["Year"] = df["Arrival_Date"].dt.year
    df["Month"] = df["Arrival_Date"].dt.month
    df["Day"] = df["Arrival_Date"].dt.day
    df["DayOfWeek"] = df["Arrival_Date"].dt.dayofweek

    model_data = df.dropna(
        subset=["Previous_Price", "Rolling_Mean_3", "Rolling_Mean_7",
                "Target_Next_Observed_Price"]
    ).copy()
    if model_data.empty or model_data["Arrival_Date"].nunique() < 2:
        raise ValueError(
            "Not enough historical observations to build lag features. "
            "Use a dataset with more dates per market/crop/variety/grade."
        )

    print(f"Usable training examples: {len(model_data):,}")
    print(f"Date range: {model_data['Arrival_Date'].min().date()} to "
          f"{model_data['Arrival_Date'].max().date()}")
    print(f"Distinct crops used: {model_data['Commodity'].nunique():,}")
    print(f"Distinct market names used: {model_data['Market'].nunique():,}")

    # Chronological split prevents training on later dates than the test set.
    dates = np.array(sorted(model_data["Arrival_Date"].unique()))
    cutoff = dates[max(1, min(len(dates) - 1, int(len(dates) * 0.8)))]
    train = model_data[model_data["Arrival_Date"] < cutoff].copy()
    test = model_data[model_data["Arrival_Date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train/test set.")

    categorical = GROUPS
    numeric = [
        "Year", "Month", "Day", "DayOfWeek", "Previous_Price",
        "Rolling_Mean_3", "Rolling_Mean_7"
    ]
    features = categorical + numeric
    target = "Target_Next_Observed_Price"

    preprocess = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical)
    ], remainder="passthrough")

    model = Pipeline([
        ("preprocessor", preprocess),
        ("xgboost", XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, n_jobs=-1
        ))
    ])

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    baseline = X_test["Previous_Price"].to_numpy()

    print(f"Training rows: {len(train):,}; testing rows: {len(test):,}")
    print(f"Training dates end: {train['Arrival_Date'].max().date()}")
    print(f"Testing dates start: {test['Arrival_Date'].min().date()}")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = pd.DataFrame([
        {"model": "Previous-price baseline",
         "MAE": mean_absolute_error(y_test, baseline),
         "RMSE": np.sqrt(mean_squared_error(y_test, baseline)),
         "R2": r2_score(y_test, baseline)},
        {"model": "Multi-crop XGBoost",
         "MAE": mean_absolute_error(y_test, pred),
         "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
         "R2": r2_score(y_test, pred)}
    ])
    metrics.to_csv(output / "evaluation_metrics.csv", index=False)

    results = test[GROUPS + ["Arrival_Date", "Modal_Price"]].copy()
    results["Actual_Next_Observed_Price"] = y_test.to_numpy()
    results["Previous_Price_Baseline"] = baseline
    results["Predicted_Next_Observed_Price"] = np.round(pred, 2)
    results.to_csv(output / "test_predictions.csv", index=False)
    joblib.dump(model, output / "multi_crop_xgboost.joblib")

    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
        values = model.named_steps["xgboost"].feature_importances_
        pd.DataFrame({"feature": names, "importance": values}).sort_values(
            "importance", ascending=False
        ).to_csv(output / "feature_importance.csv", index=False)
    except Exception as exc:
        print(f"Feature-importance export skipped: {exc}")

    print("\n=== EVALUATION (price units as provided by dataset) ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved reports and model in: {output.resolve()}")
    print("Note: target is next available recorded price, not a fixed 7-day forecast.")


if __name__ == "__main__":
    main()
