
import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load and clean the data
df = pd.read_csv("tomato_clean.csv")

df["Arrival_Date"] = pd.to_datetime(
    df["Arrival_Date"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

df["Modal_Price"] = pd.to_numeric(
    df["Modal_Price"], errors="coerce"
)

df = df.dropna(subset=["Arrival_Date", "Modal_Price"])
df = df[df["Modal_Price"] > 0].copy()

group_cols = ["State", "District", "Market", "Variety", "Grade"]

for col in group_cols:
    df[col] = df[col].fillna("Unknown").astype(str)

# 2. One record per market, variety, grade and date
df = (
    df.groupby(
        group_cols + ["Arrival_Date"],
        as_index=False,
        dropna=False
    )
    .agg(Modal_Price=("Modal_Price", "mean"))
)

# 3. Sort chronologically within each market series
df = df.sort_values(
    group_cols + ["Arrival_Date"]
).reset_index(drop=True)

grouped = df.groupby(group_cols)["Modal_Price"]

# Only use prices recorded BEFORE the date being predicted
df["Previous_Price"] = grouped.shift(1)

df["Rolling_Mean_3"] = grouped.transform(
    lambda s: s.shift(1).rolling(3, min_periods=3).mean()
)

df["Rolling_Mean_7"] = grouped.transform(
    lambda s: s.shift(1).rolling(7, min_periods=7).mean()
)

# 4. Create date features
df["Year"] = df["Arrival_Date"].dt.year
df["Month"] = df["Arrival_Date"].dt.month
df["Day"] = df["Arrival_Date"].dt.day
df["DayOfWeek"] = df["Arrival_Date"].dt.dayofweek

# Require enough historical observations
df = df.dropna(
    subset=["Previous_Price", "Rolling_Mean_3", "Rolling_Mean_7"]
).copy()

print("Usable market-date records:", len(df))
print("Date range:", df["Arrival_Date"].min().date(),
      "to", df["Arrival_Date"].max().date())

# 5. Split by date, not randomly
unique_dates = df["Arrival_Date"].drop_duplicates().sort_values().tolist()
cutoff = unique_dates[int(len(unique_dates) * 0.8)]

train_mask = df["Arrival_Date"] < cutoff
test_mask = df["Arrival_Date"] >= cutoff

print("Training ends:", df.loc[train_mask, "Arrival_Date"].max().date())
print("Testing starts:", df.loc[test_mask, "Arrival_Date"].min().date())

categorical = group_cols
numeric = [
    "Year", "Month", "Day", "DayOfWeek",
    "Previous_Price", "Rolling_Mean_3", "Rolling_Mean_7"
]
features = categorical + numeric

X_train = df.loc[train_mask, features]
X_test = df.loc[test_mask, features]
y_train = df.loc[train_mask, "Modal_Price"]
y_test = df.loc[test_mask, "Modal_Price"]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 6. Previous-price baseline
baseline = X_test["Previous_Price"].to_numpy()

print("\n===== PREVIOUS-PRICE BASELINE =====")
print(f"MAE:  Rs. {mean_absolute_error(y_test, baseline):.2f}")
print(f"RMSE: Rs. {np.sqrt(mean_squared_error(y_test, baseline)):.2f}")
print(f"R2:   {r2_score(y_test, baseline):.4f}")

# 7. Build and train XGBoost
preprocessor = ColumnTransformer(
    transformers=[
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical)
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("xgboost", XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining enhanced XGBoost...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n===== ENHANCED XGBOOST RESULTS =====")
print(f"MAE:  Rs. {mean_absolute_error(y_test, predictions):.2f}")
print(f"RMSE: Rs. {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
print(f"R2:   {r2_score(y_test, predictions):.4f}")

# 8. Save the model and test predictions
results = df.loc[test_mask, group_cols + ["Arrival_Date"]].copy()
results["Actual_Price"] = y_test.to_numpy()
results["Previous_Price"] = baseline
results["Predicted_Price"] = predictions.round(2)

results.to_csv("xgboost_lag_predictions.csv", index=False)
joblib.dump(model, "xgboost_lag_model.joblib")

print("\nSaved: xgboost_lag_predictions.csv")
print("Saved: xgboost_lag_model.joblib")
print("\nFirst 10 predictions:")
print(results[[
    "Arrival_Date", "Market", "Actual_Price",
    "Previous_Price", "Predicted_Price"
]].head(10).to_string(index=False))
