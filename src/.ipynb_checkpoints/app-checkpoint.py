import joblib
import lstm_functions as lf
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import streamlit as st

from pathlib import Path
from datetime import datetime, timedelta
from tensorflow import keras

HERE = Path(__file__).resolve().parent          # .../src
REPO_ROOT = HERE.parent                         # .../congestion_zone_traffic

MODEL_PATH = REPO_ROOT / "lstm_model.keras"
SCALER_PATH = REPO_ROOT / "scaler.pkl"

st.set_page_config(page_title="NYC Congestion Pricing - Traffic Volume", layout="wide")

# Helper functions/variables

COL_TIMESTAMP = "toll_timestamp"
COL_PRED = "traffic_volume_pred"

@st.cache_data(show_spinner=True, ttl=60 * 10) # cache for 10 minutes

def get_predictions(last_n_days: int):
    """
    Returns:
      pred_df_small: DataFrame with [toll_timestamp, traffic_volume_pred] for filtering/ranking
      y_true: np.array of true values aligned to predictions (for diagnostic plot)
      y_pred: np.array of predicted values (for diagnostic plot)
      mae, rmse: floats
    """
    raw = lf.load_last_n_days_brooklyn(n_days=last_n_days)
    feat = lf.aggr_data(raw)

    # ---- robust file loading (works on Streamlit Cloud) ----
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    if not SCALER_PATH.exists():
        st.error(f"Scaler file not found at: {SCALER_PATH}")
        st.stop()

    mdl = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    pred_df_full, mae, rmse = lf.mdl_pipe(mdl, scaler, feat, SEQ_LEN=48)

    # Ensure datetime
    pred_df_full = pred_df_full.copy()
    pred_df_full[COL_TIMESTAMP] = pd.to_datetime(pred_df_full[COL_TIMESTAMP])

    # Pull arrays for diagnostic plot
    y_true = pred_df_full["traffic_volume_true"].to_numpy()
    y_pred = pred_df_full["traffic_volume_pred"].to_numpy()

    # Small DF for filtering/ranking
    pred_df_small = pred_df_full[[COL_TIMESTAMP, "traffic_volume_pred"]].sort_values(COL_TIMESTAMP)

    return pred_df_small, y_true, y_pred, mae, rmse

st.title("NYC Congestion Pricing - Predicted Traffic Volume Coming From Brooklyn")

with st.sidebar:
    st.header("Controls")
    last_n_days = st.slider("Pull last N days of raw data",1,60,14)
    
    st.divider()
    st.header("Filter View")
    
    # Simple data filtering
    start_date = st.date_input("Start Date", value=(datetime.now() - timedelta(days=7)).date())
    end_date = st.date_input("End date", value=datetime.now().date())
    
    st.divider()
    st.header("Ranking (within selected range)")
    st.caption("Ranks timestamps by predicted volume (Lowest -> Highest).")
    top_n = st.slider("Show N lowest / N highest", 3, 25, 10)
    
with st.spinner("Loading data and generating predictions..."):
    pred, y_true, y_pred, mae, rmse = get_predictions(last_n_days)
    
ts = pd.to_datetime(pred[COL_TIMESTAMP], errors="coerce")

# If tz-aware -> convert to NY then drop tz
if getattr(ts.dt, "tz", None) is not None:
    pred[COL_TIMESTAMP] = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
else:
    # If tz-naive -> assume UTC, localize then convert
    pred[COL_TIMESTAMP] = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    
# validate
needed = {COL_TIMESTAMP, COL_PRED}
if not needed.issubset(pred.columns):
    st.error(f"Prediction output must include columns: {sorted(list(needed))}")
    st.stop()
    
# filter by date range
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

pred_f = pred[(pred[COL_TIMESTAMP] >= start_dt) & (pred[COL_TIMESTAMP] <= end_dt)].copy()


if pred_f.empty:
    
    
    st.warning("No data in the selected date range.")
    st.stop()
    
    
# --- Main Layout ---
left, right = st.columns([2,1])

with left:
    st.subheader("Predicted traffic volume over time")

    st.metric("MAE", f"{mae:,.2f}")
    st.metric("RMSE", f"{rmse:,.2f}")

    n_points = st.slider("Number of points to display", 50, 500, 200, step=50)

    fig = lf.mdl_plt_streamlit(y_true, y_pred, n=n_points)
    st.pyplot(fig, clear_figure=True)
    
    with st.expander("Show prediction data"):
        
#         st.dataframe(pred_f,use_container_width=True)
        st.text(pred_f.head(200).to_string(index=False))
        
with right:
    st.subheader("Summary")
    st.metric("Rows in view", f"{len(pred_f):,}")
    st.metric("Range",
             f"{pred_f[COL_TIMESTAMP].min():%Y-%m-%d %H:%M} → {pred_f[COL_TIMESTAMP].max():%Y-%m-%d %H:%M}"
             )
    st.metric("Avg predicted volume", f"{pred_f[COL_PRED].mean():,.0f}")
    st.metric("Min predicted volume", f"{pred_f[COL_PRED].min():,.0f}")
    st.metric("Max predicted volume", f"{pred_f[COL_PRED].max():,.0f}")

st.divider()
st.subheader("Lowest → Highest predicted timestamps (within selected range)")

ranked = pred_f.sort_values(COL_PRED, ascending=True).reset_index(drop=True)

lowest = ranked.head(top_n).copy()
lowest.insert(0, "Rank (lowest)", range(1, len(lowest) + 1))

highest = ranked.tail(top_n).sort_values(COL_PRED, ascending=False).copy()
highest.insert(0, "Rank (highest)", range(1, len(highest) + 1))

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Lowest")
#     st.dataframe(lowest, use_container_width=True)
    st.text(lowest.to_string(index=False))

with c2:
    st.markdown("### Highest")
#     st.dataframe(highest, use_container_width=True)
    st.text(highest.to_string(index=False))