"""Streamlit dashboard for market prices and forecasts."""
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from market_price import DATE, DIMENSIONS, PRICE, make_features

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

@st.cache_data(show_spinner=False)
def load_data():
    path = ARTIFACTS / "cleaned_prices.csv"
    return pd.read_csv(path, parse_dates=[DATE]) if path.exists() else None

@st.cache_resource(show_spinner=False)
def load_model():
    path = ARTIFACTS / "price_model.joblib"
    return joblib.load(path) if path.exists() else None

st.set_page_config(page_title="Farm Price Outlook", layout="wide")
st.title("Farm Price Outlook")
st.caption("Explore reported market prices and exact seven-calendar-day modal-price forecasts.")
df = load_data()
if df is None:
    st.warning("Prepared data is missing. Run `python train_model.py` from the project folder first.")
    st.stop()
report_path, metric_path = ARTIFACTS / "cleaning_report.json", ARTIFACTS / "model_metrics.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
metrics = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else None
dash, explore, forecast = st.tabs(["Dashboard", "Explore prices", "Forecast"])
with dash:
    cols = st.columns(4)
    for col, label, val in zip(cols, ["Valid records", "Commodities", "Markets", "Date range"],
                               [f"{len(df):,}", f"{df.Commodity.nunique():,}", f"{df.Market.nunique():,}", f"{df[DATE].min():%Y-%m-%d} to {df[DATE].max():%Y-%m-%d}"]):
        col.metric(label, val)
    st.subheader("Data quality")
    st.write({k: report.get(k) for k in ["input_rows", "retained_rows", "invalid_rows_removed", "duplicate_records", "price_relationship_violations"]})
    if metrics:
        st.subheader("Chronological test results | exact +7 day target")
        st.write(f"{metrics['model']} | {metrics['test_rows']:,} test records | {metrics['test_start']} to {metrics['test_end']}")
        left, right = st.columns(2)
        for col, key, label in [(left,"baseline","Persistence baseline"),(right,"model_metrics","Trained model")]:
            col.markdown(f"**{label}**")
            for name in ["mae", "rmse", "r2"]:
                value = metrics[key][name]
                col.metric(name.upper(), "n/a" if value is None else f"{value:.3f}")
        st.caption("R-squared can be negative. These historical test results are estimates, not guarantees.")
    else:
        st.info("Model evaluation is unavailable.")
with explore:
    subset = df
    for dim in DIMENSIONS:
        opts = sorted(subset[dim].fillna("Unknown").astype(str).unique())
        choice = st.selectbox(dim, opts, key="explore_" + dim)
        subset = subset[subset[dim].astype(str) == choice]
    subset = subset.sort_values(DATE)
    st.caption("Modal price (units as recorded in the source)")
    st.line_chart(subset.set_index(DATE)[PRICE])
    st.dataframe(subset, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", subset.to_csv(index=False), "market_prices.csv", "text/csv")
with forecast:
    bundle = load_model()
    if bundle is None:
        st.warning("No saved model. Run `python train_model.py` to train it.")
    else:
        if metrics:
            st.caption(f"Held-out test MAE {metrics['model_metrics']['mae']:.2f} | RMSE {metrics['model_metrics']['rmse']:.2f} | R-squared {metrics['model_metrics']['r2'] if metrics['model_metrics']['r2'] is not None else 'n/a'}")
        subset = df
        for dim in DIMENSIONS:
            opts = sorted(subset[dim].fillna("Unknown").astype(str).unique())
            choice = st.selectbox(dim, opts, key="forecast_" + dim)
            subset = subset[subset[dim].astype(str) == choice]
        subset = subset.sort_values(DATE)
        if len(subset) < 8:
            st.error("This series has too little history for the saved model's historical features.")
        else:
            last = subset.iloc[-1]
            st.write(f"Latest observation: **{last[DATE].date()}** | modal price **{last[PRICE]:,.2f}** (source units).")
            history_features = make_features(subset)
            if history_features["Target_Price"].notna().sum() < 1:
                st.error("This market series has no observed exact seven-day target pairs, so its forecast horizon is unsupported.")
            else:
                row = history_features.iloc[[-1]]
                if row["Rolling_Mean_7"].isna().any():
                    st.error("Required historical features are unavailable for this series.")
                else:
                    try:
                        estimate = float(bundle["model"].predict(row[bundle["features"]])[0])
                        st.metric("Forecast modal price", f"{estimate:,.2f}")
                        st.write(f"Forecast date: **{(last[DATE] + pd.Timedelta(days=7)).date()}** (7 calendar days after the latest observation).")
                        st.caption("Estimate based on historical records; reporting gaps mean the forecast is supported only by series and dates represented in the exact seven-day training targets.")
                    except (KeyError, ValueError) as exc:
                        st.error(f"Could not produce a forecast: {exc}")

