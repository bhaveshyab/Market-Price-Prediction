
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load saved test predictions
df = pd.read_csv("xgboost_lag_predictions.csv")
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

df["Abs_Error"] = (
    df["Actual_Price"] - df["Predicted_Price"]
).abs()

df["Baseline_Error"] = (
    df["Actual_Price"] - df["Previous_Price"]
).abs()

# 1. Overall evaluation
mae = mean_absolute_error(df["Actual_Price"], df["Predicted_Price"])
rmse = np.sqrt(mean_squared_error(df["Actual_Price"], df["Predicted_Price"]))
r2 = r2_score(df["Actual_Price"], df["Predicted_Price"])

print("===== OVERALL MODEL PERFORMANCE =====")
print(f"Test records: {len(df)}")
print(f"MAE: Rs. {mae:.2f}")
print(f"RMSE: Rs. {rmse:.2f}")
print(f"R2: {r2:.4f}")

# 2. Monthly performance
df["Month"] = df["Arrival_Date"].dt.strftime("%Y-%m")

monthly = df.groupby("Month").apply(
    lambda g: pd.Series({
        "Records": len(g),
        "XGBoost_MAE": g["Abs_Error"].mean(),
        "Baseline_MAE": g["Baseline_Error"].mean(),
        "XGBoost_RMSE": np.sqrt(
            np.mean((g["Actual_Price"] - g["Predicted_Price"]) ** 2)
        )
    }),
    include_groups=False
).round(2)

print("\n===== PERFORMANCE BY MONTH =====")
print(monthly.to_string())

monthly.to_csv("monthly_model_evaluation.csv")

# 3. Performance by market
market_results = df.groupby("Market").agg(
    Records=("Actual_Price", "size"),
    XGBoost_MAE=("Abs_Error", "mean"),
    Baseline_MAE=("Baseline_Error", "mean")
)

market_results = market_results[market_results["Records"] >= 20]
market_results = market_results.sort_values("XGBoost_MAE")

print("\n===== 10 BEST-PERFORMING MARKETS =====")
print(market_results.head(10).round(2).to_string())

print("\n===== 10 MARKETS WITH HIGHEST MAE =====")
print(market_results.tail(10).sort_values(
    "XGBoost_MAE", ascending=False
).round(2).to_string())

market_results.to_csv("market_model_evaluation.csv")

print("\nSaved: monthly_model_evaluation.csv")
print("Saved: market_model_evaluation.csv")

