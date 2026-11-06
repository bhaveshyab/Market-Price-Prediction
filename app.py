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
    return pd.read_csv(path, parse_dates=[DATE], low_memory=False) if path.exists() else None

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
coverage_path = ARTIFACTS / "commodity_coverage.csv"
coverage = pd.read_csv(coverage_path).set_index("Commodity") if coverage_path.exists() else None
dash, explore, forecast = st.tabs(["Dashboard", "Explore prices", "Forecast"])
with dash:
    cols = st.columns(4)
    for col, label, val in zip(cols, ["Valid records", "Commodities", "Markets", "Date range"],
                               [f"{len(df):,}", f"{df.Commodity.nunique():,}", f"{df.Market.nunique():,}", f"{df[DATE].min():%Y-%m-%d} to {df[DATE].max():%Y-%m-%d}"]):
        col.metric(label, val)
    st.subheader("Data quality")
    st.write({k: report.get(k) for k in ["input_rows", "retained_rows", "invalid_rows_removed", "duplicate_records", "price_relationship_violations"]})
    if coverage is not None:
        st.subheader("Commodity coverage")
        st.dataframe(coverage.reset_index(), width="stretch", hide_index=True)
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
    for dim in ["Commodity", "State", "District", "Market", "Variety", "Grade"]:
        opts = sorted(subset[dim].fillna("Unknown").astype(str).unique())
        choice = st.selectbox(dim, opts, key="explore_" + dim)
        subset = subset[subset[dim].astype(str) == choice]
    subset = subset.sort_values(DATE)
    st.caption("Modal price (units as recorded in the source)")
    st.line_chart(subset.set_index(DATE)[PRICE])
    st.dataframe(subset, width="stretch", hide_index=True)
    st.download_button("Download filtered CSV", subset.to_csv(index=False), "market_prices.csv", "text/csv")
with forecast:
    bundle = load_model()
    if bundle is None:
        st.warning("No saved model. Run `python train_model.py` to train it.")
    else:
        if metrics:
            st.caption(f"Held-out test MAE {metrics['model_metrics']['mae']:.2f} | RMSE {metrics['model_metrics']['rmse']:.2f} | R-squared {metrics['model_metrics']['r2'] if metrics['model_metrics']['r2'] is not None else 'n/a'}")
        subset = df
        for dim in ["Commodity", "State", "District", "Market", "Variety", "Grade"]:
            opts = sorted(subset[dim].fillna("Unknown").astype(str).unique())
            choice = st.selectbox(dim, opts, key="forecast_" + dim)
            subset = subset[subset[dim].astype(str) == choice]
        subset = subset.sort_values(DATE)
        commodity = subset["Commodity"].iloc[0] if len(subset) else None
        trained_rows = int(coverage.loc[commodity, "train_rows"]) if coverage is not None and commodity in coverage.index else 0
        if trained_rows < 20:
            st.error(f"Insufficient reliable training history for {commodity}: only {trained_rows} exact seven-day training targets are available; forecasts are disabled below 20.")
        elif len(subset) < 8:
            st.error(f"This market/variety/grade series has only {len(subset)} observations; at least 8 are required for historical features.")
        else:
            last = subset.iloc[-1]
            st.write(f"Latest observation: **{last[DATE].date()}** | modal price **{last[PRICE]:,.2f}** (source units).")
            history_features = make_features(subset)
            row = history_features.iloc[[-1]]
            if row["Rolling_Mean_7"].isna().any():
                st.error("Required historical features are unavailable for this series.")
            else:
                try:
                    estimate = float(bundle["model"].predict(row[bundle["features"]])[0])
                    st.metric("Forecast modal price", f"{estimate:,.2f}")
                    st.write(f"Forecast date: **{(last[DATE] + pd.Timedelta(days=7)).date()}** (7 calendar days after the latest observation).")
                    st.caption("Estimate uses the selected series history and the pooled model trained on eligible observations for all supported commodities.")
                except (KeyError, ValueError) as exc:
                    st.error(f"Could not produce a forecast: {exc}")

