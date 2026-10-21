
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load the dataset
df = pd.read_csv("tomato_clean.csv")

print("Original dataset:", df.shape)

# 2. Prepare the date and target columns
df["Arrival_Date"] = pd.to_datetime(
    df["Arrival_Date"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)
df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")

df = df.dropna(subset=["Arrival_Date", "Modal_Price"])
df = df[df["Modal_Price"] > 0].copy()

# 3. Create useful date features
df["Year"] = df["Arrival_Date"].dt.year
df["Month"] = df["Arrival_Date"].dt.month
df["Day"] = df["Arrival_Date"].dt.day
df["DayOfWeek"] = df["Arrival_Date"].dt.dayofweek

df = df.sort_values("Arrival_Date").reset_index(drop=True)

print("Usable rows:", len(df))
print("Date range:", df["Arrival_Date"].min().date(),
      "to", df["Arrival_Date"].max().date())

# 4. Choose the input features and prediction target
categorical_features = [
    "State", "District", "Market", "Variety", "Grade"
]

numeric_features = ["Year", "Month", "Day", "DayOfWeek"]

for column in categorical_features:
    df[column] = df[column].fillna("Unknown").astype(str)

X = df[categorical_features + numeric_features]
y = df["Modal_Price"]


# 5. Split by date so records from the same day stay together
cutoff_date = df["Arrival_Date"].quantile(0.8)

train_mask = df["Arrival_Date"] <= cutoff_date
test_mask = df["Arrival_Date"] > cutoff_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training ends:", df.loc[train_mask, "Arrival_Date"].max().date())
print("Testing starts:", df.loc[test_mask, "Arrival_Date"].min().date())


# 6. Convert text categories into numerical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        )
    ],
    remainder="passthrough",
)

# 7. Build the XGBoost model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "xgboost",
            XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# 8. Train the model
print("\nTraining XGBoost... please wait!")
model.fit(X_train, y_train)

# 9. Evaluate on unseen, later records
predictions = model.predict(X_test)

# Simple baseline: always predict the training-set median
baseline_predictions = np.full(len(y_test), y_train.median())

print("\n===== BASELINE RESULTS =====")
print(f"MAE:  Rs. {mean_absolute_error(y_test, baseline_predictions):.2f}")
print(f"RMSE: Rs. {np.sqrt(mean_squared_error(y_test, baseline_predictions)):.2f}")
print(f"R2 Score: {r2_score(y_test, baseline_predictions):.4f}")


mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n===== XGBOOST RESULTS =====")
print(f"MAE:  Rs. {mae:.2f}")
print(f"RMSE: Rs. {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# 10. Save predictions and trained model
results = pd.DataFrame({
    "Actual_Price": y_test.to_numpy(),
    "Predicted_Price": predictions.round(2),
})

results.to_csv("xgboost_predictions.csv", index=False)
joblib.dump(model, "xgboost_model.joblib")

print("\nSaved: xgboost_predictions.csv")
print("Saved: xgboost_model.joblib")
print("\nFirst 10 predictions:")
print(results.head(10).to_string(index=False))
