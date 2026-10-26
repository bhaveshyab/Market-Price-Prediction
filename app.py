"""Streamlit interface for the multi-crop market-price model."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

DATA_CANDIDATES = [Path("2025 cleaned.xlsx"), Path("data/2025 cleaned.xlsx"), Path("data/market_prices_all_crops.csv")]
MODEL_PATH = Path("outputs/multi_crop_xgboost.joblib")
GROUPS = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
NUMERIC = ["Year", "Month", "Day", "DayOfWeek", "Previous_Price", "Rolling_Mean_3", "Rolling_Mean_7"]
FEATURES = GROUPS + NUMERIC

st.set_page_config(page_title="Market Price Prediction", page_icon="🌾", layout="wide")
st.title("AI-Assisted Market Price Prediction")
st.caption("Explore historical agricultural market prices and estimate the next recorded modal price.")

@st.cache_data(show_spinner="Loading dataset…")
def load_history(path_text):
    path = Path(path_text)
    frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path, low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    needed = GROUPS + ["Arrival_Date", "Modal_Price"]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    frame["Arrival_Date"] = pd.to_datetime(frame["Arrival_Date"], format="mixed", dayfirst=True, errors="coerce")
    frame["Modal_Price"] = pd.to_numeric(frame["Modal_Price"], errors="coerce")
    for col in GROUPS:
        frame[col] = frame[col].astype("string").str.strip().replace("", pd.NA).fillna("Unknown")
    frame = frame.dropna(subset=["Arrival_Date", "Modal_Price"])
    frame = frame[frame["Modal_Price"] > 0].copy()
    frame = frame.groupby(GROUPS + ["Arrival_Date"], as_index=False, dropna=False).agg(Modal_Price=("Modal_Price", "mean"))
    return frame.sort_values(GROUPS + ["Arrival_Date"]).reset_index(drop=True)

path = next((p for p in DATA_CANDIDATES if p.exists()), None)
if path is None:
    st.error("Dataset not found. Put 2025 cleaned.xlsx in the project root, or put a CSV at data/market_prices_all_crops.csv.")
    st.code('python train_multi_crop.py --input "C:/full/path/to/your/dataset.xlsx"')
    st.stop()

try:
    history = load_history(str(path))
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

left, right = st.columns(2)
left.metric("Valid market-date records", f"{len(history):,}")
right.metric("Crop types", f"{history['Commodity'].nunique():,}")
st.caption(f"Dataset: {path} · {history['Arrival_Date'].min().date()} to {history['Arrival_Date'].max().date()}")

st.subheader("Find a crop and market")
a, b, c = st.columns(3)
crop = a.selectbox("Crop / commodity", sorted(history["Commodity"].unique()))
crop_df = history[history["Commodity"] == crop]
state = b.selectbox("State", sorted(crop_df["State"].unique()))
state_df = crop_df[crop_df["State"] == state]
district = c.selectbox("District", sorted(state_df["District"].unique()))
district_df = state_df[state_df["District"] == district]
d, e, f = st.columns(3)
market = d.selectbox("Market", sorted(district_df["Market"].unique()))
market_df = district_df[district_df["Market"] == market]
variety = e.selectbox("Variety", sorted(market_df["Variety"].unique()))
variety_df = market_df[market_df["Variety"] == variety]
grade = f.selectbox("Grade", sorted(variety_df["Grade"].unique()))
series = variety_df[variety_df["Grade"] == grade].sort_values("Arrival_Date").copy()

st.subheader("Historical prices")
st.line_chart(series.set_index("Arrival_Date")["Modal_Price"])
st.dataframe(series.tail(10).sort_values("Arrival_Date", ascending=False), use_container_width=True, hide_index=True)

st.subheader("Estimate the next recorded price")
st.warning("This estimates the next available recorded price for the selected series. It does not guarantee a forecast exactly 7 days ahead.")
if len(series) < 8:
    st.info("This series has fewer than 8 dated records, so there is not enough history for the 7-observation rolling feature.")
else:
    series["Previous_Price"] = series["Modal_Price"].shift(1)
    series["Rolling_Mean_3"] = series["Modal_Price"].shift(1).rolling(3, min_periods=3).mean()
    series["Rolling_Mean_7"] = series["Modal_Price"].shift(1).rolling(7, min_periods=7).mean()
    latest = series.iloc[-1]
    if pd.isna(latest["Previous_Price"]) or pd.isna(latest["Rolling_Mean_7"]):
        st.info("Not enough valid lag history for this series.")
    else:
        st.write(f"Latest recorded date: **{latest['Arrival_Date'].date()}**")
        st.write(f"Latest observed modal price: **₹{latest['Modal_Price']:,.2f}** (currency/unit as supplied by source data)")
        if not MODEL_PATH.exists():
            st.info("The model has not been trained in this local folder yet. Run the training command below, then refresh this page.")
            st.code('python train_multi_crop.py --input "2025 cleaned.xlsx"')
        else:
            model = joblib.load(MODEL_PATH)
            row = {col: str(latest[col]) for col in GROUPS}
            date = latest["Arrival_Date"]
            row.update({"Year": date.year, "Month": date.month, "Day": date.day, "DayOfWeek": date.dayofweek,
                        "Previous_Price": float(latest["Previous_Price"]), "Rolling_Mean_3": float(latest["Rolling_Mean_3"]),
                        "Rolling_Mean_7": float(latest["Rolling_Mean_7"])})
            predicted = max(0.0, float(model.predict(pd.DataFrame([row], columns=FEATURES))[0]))
            st.metric("Predicted next recorded modal price", f"₹{predicted:,.2f}")
            st.caption("Use this as a decision-support estimate, not a guaranteed market price.")

st.divider()
st.caption("Model quality should be judged using the saved chronological test metrics in outputs/evaluation_metrics.csv.")
