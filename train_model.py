"""Train and evaluate an exact seven-calendar-day price model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from market_price import DATE, DIMENSIONS, chron_split, clean_data, make_features, read_sources, save_clean_report

ROOT = Path(__file__).resolve().parent
CAT = DIMENSIONS
NUM = ["Year", "Month", "Day", "Weekday", "DayOfYear", "Lag_1", "Lag_7", "Rolling_Mean_7", "Rolling_Std_7"]
FEATURES = CAT + NUM

def build_model(estimator):
    """Bundle compact categorical preprocessing with the estimator."""
    return Pipeline([("encode", ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=2), CAT),
        ("numeric", SimpleImputer(strategy="median"), NUM),
    ], remainder="drop")), ("regressor", estimator)])

def score(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else None}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--data-file", type=Path, help="Use one explicit CSV/XLSX source file.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    excluded = {"tomato_clean.csv", "tomato_clean.csv.xlsx", "xgboost_predictions.csv",
                "xgboost_lag_predictions.csv", "feature_importance_report.csv",
                "monthly_model_evaluation.csv", "market_model_evaluation.csv"}
    candidates = ([args.data_file] if args.data_file else
                  [p for p in args.data_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
                   and p.name.lower() not in excluded and not any(part.lower() in {"artifacts", ".venv", "venv"} for part in p.parts)])
    # When raw files are alongside a cleaned workbook, prefer the raw sources;
    # this prevents importing the same year twice through its derived workbook.
    raw_candidates = [p for p in candidates if "clean" not in p.stem.lower()]
    sources = raw_candidates or candidates
    if not sources:
        raise FileNotFoundError(f"No source CSV/XLSX files found under {args.data_dir}.")
    raw = read_sources(sources)
    data, report = clean_data(raw)
    report["source_files"] = [p.name for p in sources]
    report["source_rows_loaded"] = int(len(raw))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "cleaned_prices.csv", index=False)
    save_clean_report(report, args.output_dir / "cleaning_report.json")
    feat = make_features(data)
    eligible = feat.dropna(subset=["Target_Price", "Lag_1", "Rolling_Mean_7"]).copy()
    train_m, valid_m, test_m = chron_split(eligible)
    train, valid, test = eligible.loc[train_m], eligible.loc[valid_m], eligible.loc[test_m]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Insufficient exact seven-day targets for chronological train/validation/test splits.")
    try:
        from xgboost import XGBRegressor
        estimator = XGBRegressor(n_estimators=350, max_depth=6, learning_rate=0.04, subsample=.85,
                                 colsample_bytree=.85, objective="reg:squarederror", random_state=42,
                                 n_jobs=4, tree_method="hist")
        model_name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, max_features=.8, n_jobs=4, random_state=42)
        model_name = "RandomForest (XGBoost unavailable)"
    model = build_model(estimator)
    model.fit(train[FEATURES], train["Target_Price"])
    pred = model.predict(test[FEATURES])
    results = test[DIMENSIONS + [DATE, "Target_Date", "Lag_1", "Target_Price"]].copy()
    results = results.rename(columns={"Lag_1": "Persistence_Prediction", "Target_Price": "Actual_Price"})
    results["Model_Prediction"] = pred
    results.to_csv(args.output_dir / "test_predictions.csv", index=False)
    def grouped_scores(column):
        rows = []
        for key, group in results.groupby(column, dropna=False):
            if len(group) >= 20:
                rows.append({column: str(key), "test_rows": len(group),
                             "baseline": score(group.Actual_Price, group.Persistence_Prediction),
                             "model": score(group.Actual_Price, group.Model_Prediction)})
        return rows
    metrics = {"target": "exact modal price seven calendar days after observation", "model": model_name,
               "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test),
               "eligible_rows": len(eligible), "excluded_rows": int(len(feat)-len(eligible)),
               "train_start": str(train[DATE].min().date()), "train_end": str(train[DATE].max().date()),
               "test_start": str(test[DATE].min().date()), "test_end": str(test[DATE].max().date()),
               "exact_target_coverage": float(len(eligible)/max(1,len(data))),
               "baseline": score(test["Target_Price"], test["Lag_1"]), "model_metrics": score(test["Target_Price"], pred),
               "by_commodity_min_20": grouped_scores("Commodity"), "by_market_min_20": grouped_scores("Market"),
               "cleaning": report}
    (args.output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "features": FEATURES, "target_horizon_days": 7, "series_dimensions": DIMENSIONS,
                 "model_name": model_name, "metrics": metrics}, args.output_dir / "price_model.joblib")
    try:
        names = model.named_steps["encode"].get_feature_names_out()
        imp = model.named_steps["regressor"].feature_importances_
        pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).to_csv(args.output_dir / "feature_importance.csv", index=False)
    except (AttributeError, ValueError):
        pass
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
