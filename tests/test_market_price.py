import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.ensemble import RandomForestRegressor

from market_price import clean_data, make_features, parse_dates, chron_split
from train_model import FEATURES, build_model


def sample():
    dates = pd.date_range("2025-01-01", periods=28, freq="D")
    return pd.DataFrame({
        "State": ["S"] * len(dates), "District": ["D"] * len(dates),
        "Market": ["M"] * len(dates), "Commodity": ["Crop"] * len(dates),
        "Variety": ["V"] * len(dates), "Grade": ["G"] * len(dates),
        "Arrival_Date": dates.strftime("%Y-%m-%d"),
        "Modal_Price": [str(100 + i).replace(".", "") for i in range(len(dates))],
        "Min_Price": [90] * len(dates), "Max_Price": [200] * len(dates),
    })


def test_mixed_dates():
    got = parse_dates(pd.Series(["2025-01-02", "03/01/2025", "bad"]))
    assert got.notna().tolist() == [True, True, False]


def test_cleaning_and_duplicate_report():
    raw = sample()
    raw = pd.concat([raw, raw.iloc[[0]], raw.iloc[[1]].assign(Modal_Price="bad")], ignore_index=True)
    data, report = clean_data(raw)
    assert len(data) == 28
    assert report["duplicate_records"] == 1
    assert report["invalid_modal_prices"] == 1


def test_exact_seven_day_target_and_historical_features():
    data, _ = clean_data(sample())
    features = make_features(data)
    row = features.loc[features["Arrival_Date"] == pd.Timestamp("2025-01-10")].iloc[0]
    assert row["Target_Price"] == data.loc[data["Arrival_Date"] == pd.Timestamp("2025-01-17"), "Modal_Price"].iloc[0]
    assert row["Lag_1"] == 108
    assert row["Target_Date"] == pd.Timestamp("2025-01-17")
    assert np.isnan(features.iloc[0]["Lag_1"])


def test_missing_exact_target_is_not_filled_from_next_row():
    data, _ = clean_data(sample().drop(index=10))
    features = make_features(data)
    row = features.loc[features["Arrival_Date"] == pd.Timestamp("2025-01-04")].iloc[0]
    assert pd.isna(row["Target_Price"])


def test_chronological_split():
    data, _ = clean_data(sample())
    features = make_features(data).dropna(subset=["Target_Price", "Lag_1", "Rolling_Mean_7"])
    train, valid, test = chron_split(features)
    assert features.loc[train, "Arrival_Date"].max() < features.loc[valid, "Arrival_Date"].min()
    assert features.loc[valid, "Arrival_Date"].max() < features.loc[test, "Arrival_Date"].min()


def test_missing_required_column():
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_data(pd.DataFrame({"Market": ["M"]}))


def test_train_and_reload_small_model(tmp_path):
    data, _ = clean_data(sample())
    features = make_features(data).dropna(subset=["Target_Price", "Lag_1", "Rolling_Mean_7"])
    model = build_model(RandomForestRegressor(n_estimators=8, random_state=42, n_jobs=1))
    model.fit(features[FEATURES], features["Target_Price"])
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "features": FEATURES}, path)
    saved = joblib.load(path)
    estimate = saved["model"].predict(features[FEATURES].iloc[[-1]])
    assert len(estimate) == 1
    assert np.isfinite(estimate[0])
