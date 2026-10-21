
import pandas as pd
import joblib

# Load the enhanced model
model = joblib.load("xgboost_lag_model.joblib")

# Extract the trained XGBoost model and feature names
preprocessor = model.named_steps["preprocessor"]
xgb = model.named_steps["xgboost"]

feature_names = preprocessor.get_feature_names_out()
importance = xgb.feature_importances_

results = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

results = results.sort_values(
    "Importance", ascending=False
).reset_index(drop=True)

print("===== TOP 20 MOST IMPORTANT FEATURES =====")
print(results.head(20).to_string(index=False))

# Save the full report
results.to_csv("feature_importance_report.csv", index=False)

print("\nSaved: feature_importance_report.csv")
